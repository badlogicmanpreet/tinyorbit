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

from tinyorbit.api import StatusEvent, TextDelta, ThinkingDelta, make_client
from tinyorbit.permissions import PermissionMode, PermissionPolicy
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

    def _newline(self) -> None:
        if self._wrote_text:
            print()
            self._wrote_text = False


def _make_context(config: "Config", policy: PermissionPolicy) -> ToolUseContext:
    return ToolUseContext(
        cwd=config.cwd,
        abort=asyncio.Event(),
        permissions=policy,
        data_dir=config.data_dir,
    )


async def run_turn(config: "Config", session: SessionState, ctx: ToolUseContext, client, prompt: str) -> str:
    """Run one user turn to completion. Returns the final assistant text."""
    session.messages.append({"role": "user", "content": prompt})
    params = QueryParams(
        messages=session.messages,
        system=config.capabilities["system_prompt"],
        tools=config.capabilities["tools"],
        ctx=ctx,
        model=config.model,
        client=client,
        cost=session.cost,
        source="print" if config.print_mode else "repl",
        max_turns=config.max_turns,
        effort=config.effort,
        fallbacks=config.fallbacks,
        thinking_display=config.thinking_display,
    )
    renderer = Renderer()
    final_text = ""
    async for event in query(params):
        renderer.handle(event)
        if isinstance(event, AssistantTurn):
            final_text = "".join(b.text for b in event.message.content if getattr(b, "type", "") == "text")
        elif isinstance(event, Done):
            session.messages[:] = event.messages
            session.turns += 1
            if event.reason not in ("completed",):
                print(_paint(YELLOW, f"  [turn ended: {event.reason}]"))
    return final_text


# ── --print ─────────────────────────────────────────────────────────────────

def run_print_mode(config: "Config") -> int:
    if not config.initial_prompt:
        print("--print needs a prompt", file=sys.stderr)
        return 2
    policy = PermissionPolicy(mode=config.permission_mode, ask=None, allow_rules=config.allow_rules)
    session = SessionState()

    async def main() -> str:
        client = make_client()
        ctx = _make_context(config, policy)
        try:
            return await run_turn(config, session, ctx, client, config.initial_prompt)
        finally:
            await client.close()

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
    session = SessionState()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = make_client()
    ctx = _make_context(config, policy)

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
                loop.run_until_complete(run_turn(config, session, ctx, client, line))
            finally:
                loop.remove_signal_handler(signal.SIGINT)
    finally:
        loop.run_until_complete(client.close())
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
