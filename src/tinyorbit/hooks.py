"""PreToolUse guard hooks (Ch 12).

A hook is a policy callback that runs *before* a tool executes and can veto it.
It is the seam an embedder uses to enforce rules the agent must not talk its
way past: keep writes inside a sandbox, refuse `rm -rf`, block network calls,
redact a path. Guards live here, above the model, so no prompt can disable them.

Where it slots in: `run_tool` runs the matching PreToolUse hooks right before
the permission check. Three outcomes, fail-closed by design:

    deny   -> the tool never runs; the model gets an error tool_result.
    allow  -> skip the normal permission prompt and run (an explicit trust).
    pass   -> no opinion; fall through to the usual permission chain.

Precedence: any `deny` wins over any `allow`, and a hook that raises is treated
as a `deny` — a crashing guard must not silently wave a call through.

Discovery mirrors MCP sources: empty by default (a plain run and every
benchmark run pay nothing and behave identically), opt-in via `TINYORBIT_HOOKS`,
a comma-separated list of modules each exposing `pre_tool_use_hooks()`. An
embedder that drives tinyorbit as a library can instead set
`config.capabilities["hooks"]` directly with the HookSpec objects below.
"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Awaitable, Callable, Union


@dataclass
class HookInput:
    """What a PreToolUse hook sees: the call it may veto, plus the live context."""

    event: str                 # "PreToolUse"
    tool_name: str
    tool_input: dict
    ctx: Any                   # ToolUseContext — Any keeps this module import-light


@dataclass
class HookOutcome:
    """A hook's verdict. `pass` (or returning None) means 'no opinion'."""

    decision: str              # "allow" | "deny" | "pass"
    reason: str = ""


# A hook returns an outcome (or None = pass); it may be sync or async.
Hook = Callable[[HookInput], Union[HookOutcome, None, Awaitable[Union[HookOutcome, None]]]]


@dataclass
class HookSpec:
    """A registered hook: the callback plus which tools it applies to."""

    callback: Hook
    match: str = "*"           # fnmatch glob on the tool name; "*" = every tool
    event: str = "PreToolUse"


PASS = HookOutcome("pass")


async def run_pre_tool_use(hooks: list[HookSpec], tool_name: str, tool_input: dict,
                           ctx: Any) -> HookOutcome:
    """Run every matching PreToolUse hook and combine their verdicts.

    Fail-closed: the first `deny` (or the first hook that raises) short-circuits
    and wins; an `allow` is remembered but never overrides a later `deny`.
    """
    inp = HookInput("PreToolUse", tool_name, tool_input, ctx)
    verdict = PASS
    for spec in hooks:
        if spec.event != "PreToolUse" or not fnmatch(tool_name, spec.match):
            continue
        try:
            out = spec.callback(inp)
            if inspect.isawaitable(out):
                out = await out
        except Exception as exc:  # a guard that crashes must block, not wave through
            return HookOutcome("deny", f"hook raised {type(exc).__name__}: {exc}")
        if out is None or out.decision == "pass":
            continue
        if out.decision == "deny":
            return out             # first deny wins
        if out.decision == "allow":
            verdict = out          # keep scanning; a later deny still overrides
    return verdict


# ── discovery (opt-in, empty by default) ──────────────────────────────────────

def discover_hooks() -> list[HookSpec]:
    """Load PreToolUse hooks named in `TINYORBIT_HOOKS`, or none.

    Each entry is a module path lazily imported and asked for `pre_tool_use_hooks()
    -> list[HookSpec]`. A missing module or function is skipped rather than fatal,
    so a stale env var never bricks a run.
    """
    raw = os.environ.get("TINYORBIT_HOOKS", "").strip()
    if not raw:
        return []
    out: list[HookSpec] = []
    for name in (n.strip() for n in raw.split(",")):
        if not name:
            continue
        try:
            module = importlib.import_module(name)
            factory = getattr(module, "pre_tool_use_hooks", None)
            if callable(factory):
                out.extend(factory() or [])
        except Exception:  # a bad hook module disables that hook, not the agent
            continue
    return out
