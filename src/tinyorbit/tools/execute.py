"""The tool execution pipeline — from tool_use block to tool_result block.

The full pipeline in Ch 6 has 14 steps. Ours keeps the ones that change
behaviour: lookup, abort check, schema validation, semantic validation,
permission resolution, execution, result budgeting, error capture.
Hooks (Ch 12) slot in around the permission step later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from tinyorbit.tools.base import Tool, ToolResult, ToolUseContext, check_schema
from tinyorbit.tools.registry import find_tool

MAX_RESULT_CHARS = 30_000


@dataclass(frozen=True)
class ToolCall:
    """A tool_use block, detached from the SDK type so tests can build them."""

    id: str
    name: str
    input: dict[str, Any]


def tool_result_block(call_id: str, content: str, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": call_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def budget_result(text: str, ctx: ToolUseContext, call_id: str) -> str:
    """Layer 0 of context management: never let one result flood the window.

    Oversized output is persisted to disk and replaced with a preview plus
    a pointer, so the model can Read the rest if it actually needs it.
    """
    if len(text) <= MAX_RESULT_CHARS:
        return text
    store = ctx.data_dir / "tool-results"
    store.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{call_id}{text[:64]}".encode()).hexdigest()[:12]
    path = store / f"{digest}.txt"
    path.write_text(text)
    return (
        f"{text[:MAX_RESULT_CHARS]}\n\n"
        f"[output truncated: {len(text):,} characters total. Full output saved to {path}; "
        f"Read it with offset/limit if you need more.]"
    )


async def run_tool(call: ToolCall, tools: list[Tool], ctx: ToolUseContext) -> dict:
    """Run one tool call through the pipeline. Always returns a tool_result block."""
    tool = find_tool(tools, call.name)
    if tool is None:
        return tool_result_block(call.id, f"Unknown tool: {call.name}", is_error=True)

    if ctx.abort.is_set():
        return tool_result_block(call.id, "Interrupted by user before the tool ran.", is_error=True)

    error = check_schema(tool.input_schema, call.input) or tool.validate(call.input)
    if error:
        return tool_result_block(call.id, f"Invalid input for {tool.name}: {error}", is_error=True)

    # PreToolUse guard hooks (Ch 12) run before permissions: a deny blocks the
    # call outright, an allow bypasses the permission prompt, a pass falls
    # through. Empty in a plain run, so this is a no-op on the hot path.
    hook_allow = False
    if ctx.hooks:
        from tinyorbit.hooks import run_pre_tool_use

        outcome = await run_pre_tool_use(ctx.hooks, tool.name, call.input, ctx)
        if outcome.decision == "deny":
            return tool_result_block(
                call.id, f"Blocked by policy for {tool.name}: {outcome.reason}", is_error=True
            )
        hook_allow = outcome.decision == "allow"

    if not hook_allow:
        decision = await ctx.permissions.can_use_tool(tool, call.input, ctx)
        if decision.behavior != "allow":
            return tool_result_block(
                call.id, f"Permission denied for {tool.name}: {decision.reason}", is_error=True
            )

    try:
        result = await tool.call(call.input, ctx)
    except Exception as exc:  # a tool bug must not take the loop down
        result = ToolResult(f"{tool.name} failed: {type(exc).__name__}: {exc}", is_error=True)

    content = budget_result(result.data, ctx, call.id) or "(no output)"
    return tool_result_block(call.id, content, result.is_error)


def partition_for_concurrency(calls: list[ToolCall], tools: list[Tool]) -> list[list[ToolCall]]:
    """Group consecutive concurrency-safe calls so they can run in parallel (Ch 7).

    Unsafe calls become singleton groups. Order is preserved, which matters:
    the model may Read a file and then Edit it in the same turn.
    """
    groups: list[list[ToolCall]] = []
    current: list[ToolCall] = []
    for call in calls:
        tool = find_tool(tools, call.name)
        try:
            safe = bool(tool and tool.is_concurrency_safe(call.input))
        except Exception:
            safe = False  # fail closed on malformed input
        if safe:
            current.append(call)
            continue
        if current:
            groups.append(current)
            current = []
        groups.append([call])
    if current:
        groups.append(current)
    return groups


def missing_tool_results(calls: list[ToolCall], have: set[str], reason: str) -> list[dict]:
    """The protocol safety net (Ch 5): every tool_use must get a tool_result."""
    return [tool_result_block(c.id, reason, is_error=True) for c in calls if c.id not in have]
