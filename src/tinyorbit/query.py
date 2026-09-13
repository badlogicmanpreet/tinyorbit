"""The agent loop (Ch 5).

An async generator: the caller iterates events (text, tool activity, status)
and the last event is always a Done carrying the reason and the final
message history. Every `continue` site rebuilds the whole LoopState so the
transition is explicit and readable.

The skeleton, as the book puts it:

    while True:
        messages = compress_if_needed(state.messages)
        response  = call_model(messages)          # may raise, may retry
        if no tool calls: return completed
        results   = execute_tools(response)
        state     = next_state(...)

Everything else here is one of the book's elaborations: reactive compaction
on 413, max-output escalation, the orphaned tool_result safety net, abort
handling on both sides of the model call, and a max_turns circuit breaker.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Awaitable, Callable

from tinyorbit.api import (
    DEFAULT_MAX_TOKENS,
    ESCALATED_MAX_TOKENS,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    StatusEvent,
    TextDelta,
    ThinkingDelta,
    query_model,
)
from tinyorbit.state import CostTracker
from tinyorbit.tools import (
    Tool,
    ToolCall,
    ToolUseContext,
    missing_tool_results,
    partition_for_concurrency,
    run_tool,
)

AUTO_COMPACT_THRESHOLD = 150_000      # context tokens; well under the 1M window, cheap to hit
MAX_COMPACT_FAILURES = 3
MAX_OUTPUT_RECOVERIES = 3


# ── events the loop yields ──────────────────────────────────────────────────

@dataclass(frozen=True)
class AssistantTurn:
    message: Any


@dataclass(frozen=True)
class ToolStarted:
    call: ToolCall
    summary: str


@dataclass(frozen=True)
class ToolFinished:
    call: ToolCall
    result: dict          # the tool_result block


@dataclass(frozen=True)
class Done:
    reason: str           # see TERMINAL_REASONS
    messages: list[Any]


TERMINAL_REASONS = (
    "completed", "max_turns", "model_error", "prompt_too_long",
    "aborted_streaming", "aborted_tools", "refusal",
)

QueryEvent = TextDelta | ThinkingDelta | StatusEvent | AssistantTurn | ToolStarted | ToolFinished | Done


# ── inputs ──────────────────────────────────────────────────────────────────

@dataclass
class QueryParams:
    messages: list[Any]
    system: list[dict]
    tools: list[Tool]
    ctx: ToolUseContext
    model: str
    client: Any = None                    # anthropic.AsyncAnthropic; None under test
    cost: CostTracker = field(default_factory=CostTracker)
    source: str = "repl"                  # 'repl' | 'print' | 'compact' | 'agent:<id>'
    max_turns: int | None = None
    effort: str | None = None
    fallbacks: bool = True
    thinking_display: str = "omitted"


CompactFn = Callable[["QueryParams", list[Any]], Awaitable[list[Any]]]


@dataclass
class QueryDeps:
    """The injection seam. Tests swap in a scripted model and a fake compactor."""

    call_model: Callable[..., AsyncIterator[Any]] = query_model
    compact: CompactFn | None = None


@dataclass(frozen=True)
class Continue:
    reason: str


@dataclass(frozen=True)
class LoopState:
    messages: list[Any]
    turn_count: int = 0
    max_tokens_override: int | None = None
    output_recovery_count: int = 0
    attempted_reactive_compact: bool = False
    compact_failures: int = 0
    transition: Continue | None = None


# ── compaction ──────────────────────────────────────────────────────────────

COMPACT_INSTRUCTION = (
    "Context is running long. Summarize this conversation for someone who will continue "
    "the work: the user's goals, what has been done (files touched, commands run, results), "
    "what is still open, and any facts about the codebase that took effort to discover. "
    "Be concrete; include file paths. Output only the summary."
)


async def compact_messages(params: QueryParams, messages: list[Any]) -> list[Any]:
    """Layer 4: replace the history with a model-written summary."""
    request_messages = [*messages, {"role": "user", "content": COMPACT_INSTRUCTION}]
    req = ModelRequest(
        model=params.model,
        system=[{"type": "text", "text": "You write precise handoff summaries of coding sessions."}],
        messages=request_messages,
        tools=[],
        max_tokens=8_000,
        fallbacks=params.fallbacks,
    )
    summary = ""
    async for event in query_model(params.client, req, params.ctx.abort):
        if isinstance(event, ModelResponse):
            params.cost.add(event.message.usage)
            summary = "".join(b.text for b in event.message.content if b.type == "text")
    if not summary.strip():
        raise ModelCallError("model_error", "compaction returned no text")
    return [{"role": "user", "content": f"[Context was compacted. Summary so far]\n\n{summary}"}]


# ── the loop ────────────────────────────────────────────────────────────────

def _tool_calls_of(message: Any) -> list[ToolCall]:
    return [
        ToolCall(id=b.id, name=b.name, input=dict(b.input))
        for b in message.content
        if getattr(b, "type", None) == "tool_use"
    ]


def _find_tool_summary(tools: list[Tool], call: ToolCall) -> str:
    tool = next((t for t in tools if t.name == call.name), None)
    if tool is None:
        return call.name
    try:
        return tool.describe_call(call.input)
    except Exception:
        return call.name


async def query(params: QueryParams, deps: QueryDeps | None = None) -> AsyncIterator[QueryEvent]:
    deps = deps or QueryDeps()
    compact = deps.compact or compact_messages
    ctx = params.ctx
    tool_schemas = [t.to_api() for t in params.tools]
    state = LoopState(messages=list(params.messages))

    while True:
        messages = state.messages

        # ── context pipeline (only the heaviest layer exists here) ──
        if (
            params.source != "compact"
            and params.cost.last_context_tokens > AUTO_COMPACT_THRESHOLD
            and state.compact_failures < MAX_COMPACT_FAILURES
        ):
            try:
                yield StatusEvent(f"context at {params.cost.last_context_tokens:,} tokens; compacting")
                messages = await compact(params, messages)
                params.cost.last_context_tokens = 0
                state = replace(state, messages=messages, compact_failures=0)
            except ModelCallError as exc:
                yield StatusEvent(f"compaction failed: {exc}")
                state = replace(state, compact_failures=state.compact_failures + 1)

        # ── model call ──
        req = ModelRequest(
            model=params.model,
            system=params.system,
            messages=messages,
            tools=tool_schemas,
            max_tokens=state.max_tokens_override or DEFAULT_MAX_TOKENS,
            effort=params.effort,
            fallbacks=params.fallbacks,
            thinking_display=params.thinking_display,
        )
        response = None
        try:
            async for event in deps.call_model(params.client, req, ctx.abort):
                if isinstance(event, ModelResponse):
                    response = event.message
                else:
                    yield event
        except ModelCallError as exc:
            if exc.reason == "aborted":
                yield Done("aborted_streaming", messages)
                return
            if exc.reason == "prompt_too_long" and not state.attempted_reactive_compact:
                try:
                    yield StatusEvent("prompt too long; compacting and retrying")
                    compacted = await compact(params, messages)
                except ModelCallError as inner:
                    yield StatusEvent(f"compaction failed: {inner}")
                    yield Done("prompt_too_long", messages)
                    return
                state = LoopState(
                    messages=compacted,
                    turn_count=state.turn_count,
                    attempted_reactive_compact=True,
                    transition=Continue("reactive_compact_retry"),
                )
                continue
            yield StatusEvent(f"model error: {exc}")
            yield Done("model_error" if exc.reason != "prompt_too_long" else "prompt_too_long", messages)
            return

        assert response is not None
        params.cost.add(response.usage)
        yield AssistantTurn(response)
        tool_calls = _tool_calls_of(response)

        # ── output-cap recovery: escalate once, then nudge up to N times ──
        if response.stop_reason == "max_tokens" and not tool_calls:
            if state.max_tokens_override is None:
                yield StatusEvent("output hit the token cap; retrying with a larger cap")
                state = LoopState(
                    messages=messages,
                    turn_count=state.turn_count,
                    max_tokens_override=ESCALATED_MAX_TOKENS,
                    output_recovery_count=state.output_recovery_count,
                    attempted_reactive_compact=state.attempted_reactive_compact,
                    transition=Continue("max_output_tokens_escalate"),
                )
                continue
            if state.output_recovery_count < MAX_OUTPUT_RECOVERIES:
                yield StatusEvent("output still truncated; asking the model to continue")
                state = LoopState(
                    messages=[
                        *messages,
                        {"role": "assistant", "content": response.content},
                        {"role": "user", "content": "Your previous message was cut off. Continue from where you stopped."},
                    ],
                    turn_count=state.turn_count + 1,
                    max_tokens_override=ESCALATED_MAX_TOKENS,
                    output_recovery_count=state.output_recovery_count + 1,
                    attempted_reactive_compact=state.attempted_reactive_compact,
                    transition=Continue("max_output_tokens_recovery"),
                )
                continue

        messages = [*messages, {"role": "assistant", "content": response.content}]

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            yield StatusEvent(f"model declined the request (category: {category or 'unspecified'})")
            yield Done("refusal", messages)
            return

        if not tool_calls:
            if response.stop_reason == "pause_turn":
                state = replace(state, messages=messages, transition=Continue("pause_turn"))
                continue
            yield Done("completed", messages)
            return

        # ── tool execution: parallel where safe, serial otherwise ──
        results: list[dict] = []
        aborted = False
        for group in partition_for_concurrency(tool_calls, params.tools):
            if ctx.abort.is_set():
                aborted = True
                break
            for call in group:
                yield ToolStarted(call, _find_tool_summary(params.tools, call))
            group_results = await asyncio.gather(*(run_tool(c, params.tools, ctx) for c in group))
            # strict=True asserts the invariant the API protocol needs: gather returns
            # exactly one result per call, so no tool_use can go unanswered.
            for call, result in zip(group, group_results, strict=True):
                yield ToolFinished(call, result)
            results.extend(group_results)

        if aborted or ctx.abort.is_set():
            have = {r["tool_use_id"] for r in results}
            results.extend(missing_tool_results(tool_calls, have, "Interrupted by user."))
            messages = [*messages, {"role": "user", "content": results}]
            yield Done("aborted_tools", messages)
            return

        messages = [*messages, {"role": "user", "content": results}]
        next_turn = state.turn_count + 1
        if params.max_turns is not None and next_turn >= params.max_turns:
            yield StatusEvent(f"stopping: reached max_turns={params.max_turns}")
            yield Done("max_turns", messages)
            return

        # Full reconstruction: every field is named, so the transition is legible.
        state = LoopState(
            messages=messages,
            turn_count=next_turn,
            max_tokens_override=None,
            output_recovery_count=0,
            attempted_reactive_compact=False,
            compact_failures=state.compact_failures,
            transition=Continue("next_turn"),
        )
