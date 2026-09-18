"""The two launch paths: interactive REPL and --print (Ch 2 + Ch 13, minimal).

Both converge on run_turn(), which drives the query loop and renders its
events. The REPL keeps one event loop alive for the whole session so the
HTTP client's connection pool (and the prompt cache) survive between turns,
and blocks on input() *outside* the loop so Ctrl+C at the prompt exits
cleanly while Ctrl+C mid-turn just sets the abort flag.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tinyorbit.permissions import PermissionMode, PermissionPolicy
from tinyorbit.providers import StatusEvent, TextDelta, ThinkingDelta, resolve_provider
from tinyorbit.query import AssistantTurn, Done, QueryParams, ToolFinished, ToolStarted, query
from tinyorbit.state import SessionState
from tinyorbit.tools import ToolUseContext

if TYPE_CHECKING:
    from tinyorbit.bootstrap import Config

DIM, BOLD, CYAN, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"


def _color() -> bool:
    return sys.stdout.isatty()


def _paint(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if _color() else text


class Renderer:
    """Turns query events into terminal output."""

    def __init__(self) -> None:
        self._wrote_text = False

    def handle(self, event: object) -> None:
        if isinstance(event, TextDelta):
            sys.stdout.write(event.text)
            sys.stdout.flush()
            self._wrote_text = True
        elif isinstance(event, ThinkingDelta):
            sys.stdout.write(_paint(DIM, event.text))
            sys.stdout.flush()
        elif isinstance(event, StatusEvent):
            self._newline()
            print(_paint(DIM, f"  · {event.message}"))
        elif isinstance(event, ToolStarted):
            self._newline()
            print(_paint(CYAN, f"  ⚙ {event.call.name}") + " " + _paint(DIM, event.summary[:100]))
        elif isinstance(event, ToolFinished):
            content = str(event.result.get("content", ""))
            first = content.strip().splitlines()[0] if content.strip() else ""
            lines = content.count("\n") + 1
            color = RED if event.result.get("is_error") else DIM
            suffix = f" (+{lines - 1} lines)" if lines > 1 else ""
            print(_paint(color, f"    └ {first[:110]}{suffix}"))
        elif isinstance(event, AssistantTurn):
            self._newline()
        elif isinstance(event, Done):
            if event.reason not in ("completed",):
                print(_paint(YELLOW, f"  [turn ended: {event.reason}]"))

    def _newline(self) -> None:
        if self._wrote_text:
            print()
            self._wrote_text = False


def _open_session(config: "Config") -> SessionState:
    """Fresh session, or one restored from disk when --resume was given.

    fork keeps the loaded history but takes a new id, so the original file is
    never overwritten — the two timelines diverge from here. A resume id that
    does not exist is a soft miss: warn and start clean rather than abort.
    """
    from tinyorbit.sessions import load_session, restore_into

    if not config.resume_session:
        return SessionState()
    saved = load_session(config.data_dir, config.resume_session)
    if saved is None:
        print(_paint(YELLOW, f"  · no saved session {config.resume_session!r}; starting fresh"))
        return SessionState()
    session = restore_into(saved, fork=config.fork_session)
    where = f"forked {saved.session_id} → {session.session_id}" if config.fork_session \
        else f"resumed {saved.session_id}"
    print(_paint(DIM, f"  · {where} ({saved.turns} turn(s), {len(session.messages)} message(s))"))
    return session


def _make_context(config: "Config", policy: PermissionPolicy, provider=None,
                  session: "SessionState | None" = None) -> ToolUseContext:
    # The subagent runtime is attached only when agents were discovered and a
    # provider exists (i.e. a live run, not a unit test). It carries the pieces
    # the Task tool needs but a plain tool never does: provider, model, cost.
    subagent = None
    agents = config.capabilities.get("agents") or []
    if agents and provider is not None and session is not None:
        from tinyorbit.agents import SubagentContext

        subagent = SubagentContext(
            config=config, provider=provider, cost=session.cost, agents=agents,
        )
    return ToolUseContext(
        cwd=config.cwd,
        abort=asyncio.Event(),
        permissions=policy,
        data_dir=config.data_dir,
        subagent=subagent,
        hooks=config.capabilities.get("hooks") or [],
    )


async def _start_mcp(config: "Config"):
    """Connect any configured MCP servers and merge their tools into the pool.

    Runs in the REPL's event loop because MCP discovery is async while
    bootstrap.setup() is not. Returns the live session (to close on shutdown),
    or None when no server is configured — the common, zero-cost case.
    """
    servers = config.capabilities.get("mcp_servers") or []
    if not servers:
        return None
    from tinyorbit.mcp import connect_mcp_servers

    session = await connect_mcp_servers(servers)
    if session.tools:
        config.capabilities["tools"] = config.capabilities["tools"] + session.tools
        print(_paint(DIM, f"  · MCP: {len(session.tools)} tool(s) from "
                          f"{len(session.clients)} server(s)"))
    for err in session.errors:
        print(_paint(YELLOW, f"  · MCP server unavailable — {err}"))
    return session


def _apply_tool_filter(config: "Config") -> None:
    """Narrow the advertised tool pool per --allowed-tools/--disallowed-tools.

    Runs after MCP tools are merged, so it is the single choke point over the
    whole pool (built-ins, Task/Skill, MCP). A no-op when neither is set.
    """
    if not config.allowed_tools and not config.disallowed_tools:
        return
    from tinyorbit.tools import select_tools

    before = config.capabilities.get("tools") or []
    kept = select_tools(before, config.allowed_tools or None, config.disallowed_tools or None)
    config.capabilities["tools"] = kept
    if len(kept) != len(before):
        dropped = sorted({t.name for t in before} - {t.name for t in kept})
        print(_paint(DIM, f"  · tools: {len(kept)}/{len(before)} enabled "
                          f"(hidden: {', '.join(dropped)})"))


async def run_turn(config: "Config", session: SessionState, ctx: ToolUseContext, provider, prompt: str,
                   sink=None) -> str:
    """Run one user turn to completion. Returns the final assistant text.

    sink is the event consumer: the terminal Renderer by default, but an
    embedder (see runtime.Runtime) passes its own callback to observe the same
    event stream without any terminal output. Session bookkeeping (history,
    turn count, persistence) happens here regardless of who is watching.
    """
    session.messages.append({"role": "user", "content": prompt})
    params = QueryParams(
        messages=session.messages,
        system=config.capabilities["system_prompt"],
        tools=config.capabilities["tools"],
        ctx=ctx,
        model=config.model,
        provider=provider,
        cost=session.cost,
        source="print" if config.print_mode else "repl",
        max_turns=config.max_turns,
        effort=config.effort,
        fallbacks=config.fallbacks,
        thinking_display=config.thinking_display,
    )
    handle = sink if sink is not None else Renderer().handle
    final_text = ""
    async for event in query(params):
        handle(event)
        if isinstance(event, AssistantTurn):
            final_text = "".join(b.text for b in event.message.content if getattr(b, "type", "") == "text")
        elif isinstance(event, Done):
            session.messages[:] = event.messages
            session.turns += 1
    # Persist after every turn so the session is resumable even if the process
    # is killed mid-conversation. Best-effort: a write failure never fails a turn.
    from tinyorbit.sessions import save_session

    save_session(config.data_dir, session, model=config.model)
    return final_text


# ── --print ─────────────────────────────────────────────────────────────────

def run_print_mode(config: "Config") -> int:
    if not config.initial_prompt:
        print("--print needs a prompt", file=sys.stderr)
        return 2
    policy = PermissionPolicy(mode=config.permission_mode, ask=None, allow_rules=config.allow_rules)
    session = _open_session(config)

    async def main() -> str:
        provider = resolve_provider(provider=config.provider, model=config.model)
        mcp = await _start_mcp(config)
        _apply_tool_filter(config)
        ctx = _make_context(config, policy, provider, session)
        try:
            return await run_turn(config, session, ctx, provider, config.initial_prompt)
        finally:
            if mcp is not None:
                await mcp.aclose()
            await provider.close()

    asyncio.run(main())
    print()
    print(_paint(DIM, session.cost.summary(config.model)), file=sys.stderr)
    # Harness hook: a benchmark runner sets TINYORBIT_TRANSCRIPT to get the full
    # message history and cost ledger as JSON, since stdout only shows a
    # truncated rendering of tool calls and results.
    if path := os.environ.get("TINYORBIT_TRANSCRIPT"):
        _dump_transcript(Path(path), config, session)
    return 0


def _dump_transcript(path: Path, config: "Config", session: SessionState) -> None:
    import json
    from dataclasses import asdict

    def default(o: object) -> object:
        return o.model_dump() if hasattr(o, "model_dump") else str(o)

    payload = {
        "session_id": session.session_id,
        "model": config.model,
        "effort": config.effort,
        "max_turns": config.max_turns,
        "permission_mode": config.permission_mode.value,
        "turns": session.turns,
        "cost": asdict(session.cost),
        "estimated_usd": session.cost.estimate_usd(config.model),
        "messages": session.messages,
    }
    path.write_text(json.dumps(payload, indent=1, default=default))


# ── interactive ─────────────────────────────────────────────────────────────

HELP = """\
commands:  /help  /cost  /clear  /mode [default|acceptEdits|bypassPermissions]  /exit
Ctrl+C during a turn interrupts it; at the prompt it exits."""


def run_interactive(config: "Config") -> int:
    policy = PermissionPolicy(mode=config.permission_mode, allow_rules=config.allow_rules)
    session = _open_session(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    provider = resolve_provider(provider=config.provider, model=config.model)
    mcp = loop.run_until_complete(_start_mcp(config))
    _apply_tool_filter(config)
    ctx = _make_context(config, policy, provider, session)

    async def ask(prompt: str) -> bool:
        print(_paint(YELLOW, f"\n  ? allow {prompt}"))
        answer = (await asyncio.to_thread(input, "    [y]es / [n]o / [a]lways for this tool: ")).strip().lower()
        if answer.startswith("a"):
            policy.session_allows.add(prompt.split(":", 1)[0])
            return True
        return answer.startswith("y")

    policy.ask = ask

    print(_paint(BOLD, "tinyorbit") + _paint(DIM, f"  {config.model}  {config.cwd}  mode={config.permission_mode.value}"))
    print(_paint(DIM, "type /help for commands"))

    pending = config.initial_prompt
    try:
        while True:
            if pending:
                line, pending = pending, None
                print(_paint(BOLD, f"\n> {line}"))
            else:
                try:
                    line = input(_paint(BOLD, "\n> ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
            if not line:
                continue
            if line.startswith("/"):
                if _handle_command(line, config, session, policy):
                    break
                continue

            ctx.abort.clear()
            loop.add_signal_handler(signal.SIGINT, ctx.abort.set)
            try:
                loop.run_until_complete(run_turn(config, session, ctx, provider, line))
            finally:
                loop.remove_signal_handler(signal.SIGINT)
    finally:
        if mcp is not None:
            loop.run_until_complete(mcp.aclose())
        loop.run_until_complete(provider.close())
        loop.close()
    print(_paint(DIM, session.cost.summary(config.model)))
    return 0


def _handle_command(line: str, config: "Config", session: SessionState, policy: PermissionPolicy) -> bool:
    """Returns True when the REPL should exit."""
    cmd, _, arg = line.partition(" ")
    if cmd in ("/exit", "/quit", "/q"):
        return True
    if cmd == "/help":
        print(HELP)
    elif cmd == "/cost":
        print(session.cost.summary(config.model))
    elif cmd == "/clear":
        session.messages.clear()
        print(_paint(DIM, "conversation cleared"))
    elif cmd == "/mode":
        try:
            policy.mode = PermissionMode(arg.strip())
            print(_paint(DIM, f"permission mode: {policy.mode.value}"))
        except ValueError:
            print(f"unknown mode {arg!r}; one of: {', '.join(m.value for m in PermissionMode)}")
    else:
        print(f"unknown command {cmd}; try /help")
    return False
