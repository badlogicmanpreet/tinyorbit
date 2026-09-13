"""The API layer — one function that talks to the model (Ch 4).

query_model() is an async generator, for the reason the book gives: retry
progress ("overloaded, retrying in 4s") becomes part of the event stream
instead of a side channel. It yields text deltas while streaming, status
events while retrying, and finally the complete Message.

Errors that the loop can recover from (prompt too long, abort) are raised
as ModelCallError with a `reason` the loop switches on. Everything else is
`model_error`, terminal.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import anthropic

DEFAULT_MAX_TOKENS = 16_000     # p99 output is far below this; escalate on demand
ESCALATED_MAX_TOKENS = 64_000
MAX_RETRIES = 4
RETRYABLE_STATUS = {408, 409, 429, 529}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class ModelRequest:
    model: str
    system: list[dict]
    messages: list[Any]
    tools: list[dict] = field(default_factory=list)
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str | None = None
    fallbacks: bool = True          # server-side refusal fallback (Opus 5 / Fable 5.1)
    thinking_display: str = "omitted"   # "summarized" returns readable reasoning; see below


# ── events ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class StatusEvent:
    message: str


@dataclass(frozen=True)
class ModelResponse:
    message: Any     # anthropic.types.Message (or a stand-in under test)


ModelEvent = TextDelta | ThinkingDelta | StatusEvent | ModelResponse


class ModelCallError(Exception):
    reason: str   # "prompt_too_long" | "aborted" | "model_error"

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# ── retry policy ────────────────────────────────────────────────────────────

def _is_retryable(exc: anthropic.APIError) -> bool:
    if isinstance(exc, anthropic.APIConnectionError):   # includes timeouts
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS or exc.status_code >= 500
    return False


def _backoff_seconds(exc: anthropic.APIError, attempt: int) -> float:
    if isinstance(exc, anthropic.RateLimitError):
        header = exc.response.headers.get("retry-after")
        if header and header.isdigit():
            return float(header)
    return min(2 ** attempt, 30) + random.uniform(0, 1)


async def _sleep_unless_aborted(seconds: float, abort: asyncio.Event) -> bool:
    """Returns True if the abort fired during the wait."""
    try:
        await asyncio.wait_for(abort.wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False


# ── stream tracing ──────────────────────────────────────────────────────────
# Observable transport: set TINYORBIT_TRACE=<path> to append one JSON line per
# SSE event, with a millisecond offset from the request. Off by default, so the
# streaming hot path allocates nothing extra in normal use.

_TRACE_FH = None
_CALL_SEQ = itertools.count(1)


def _trace(kind: str, **fields: Any) -> None:
    global _TRACE_FH
    path = os.environ.get("TINYORBIT_TRACE")
    if not path:
        return
    if _TRACE_FH is None:
        _TRACE_FH = open(path, "a", buffering=1)
    _TRACE_FH.write(json.dumps({"kind": kind, "wall": round(time.time(), 4), **fields}, default=str) + "\n")


def _shape(messages: list[Any]) -> list[dict]:
    """What the harness is about to resend: every message as role + block types + sizes."""
    out = []
    for m in messages:
        role = m["role"] if isinstance(m, dict) else getattr(m, "role", "?")
        content = m["content"] if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(content, str):
            out.append({"role": role, "blocks": [{"type": "text", "chars": len(content)}]})
            continue
        blocks = []
        for b in content:
            bt = b.get("type") if isinstance(b, dict) else getattr(b, "type", "?")
            entry: dict[str, Any] = {"type": bt}
            if bt in ("text", "thinking"):
                val = (b.get(bt) if isinstance(b, dict) else getattr(b, bt, "")) or ""
                entry["chars"] = len(val)
            elif bt == "tool_use":
                entry["name"] = b.get("name") if isinstance(b, dict) else getattr(b, "name", "")
            elif bt == "tool_result":
                val = b.get("content") if isinstance(b, dict) else getattr(b, "content", "")
                entry["chars"] = len(str(val))
            blocks.append(entry)
        out.append({"role": role, "blocks": blocks})
    return out


def _event_detail(event: Any) -> dict:
    """Pull the few fields that make an SSE event legible, defensively."""
    d: dict[str, Any] = {}
    block = getattr(event, "content_block", None)
    if block is not None:
        d["block"] = getattr(block, "type", None)
        if getattr(block, "name", None):
            d["name"] = block.name
    delta = getattr(event, "delta", None)
    if delta is not None:
        dt = getattr(delta, "type", None)
        if dt:
            d["delta"] = dt
        for attr in ("text", "thinking", "partial_json"):
            val = getattr(delta, attr, None)
            if val:
                d["chars"] = len(val)
                d["text"] = val          # replayable: the trace can reconstruct the stream
        if getattr(delta, "stop_reason", None):
            d["stop_reason"] = delta.stop_reason
    usage = getattr(event, "usage", None) or getattr(getattr(event, "message", None), "usage", None)
    if usage is not None:
        d["usage"] = {
            "input": getattr(usage, "input_tokens", None),
            "cache_read": getattr(usage, "cache_read_input_tokens", None),
            "cache_write": getattr(usage, "cache_creation_input_tokens", None),
            "output": getattr(usage, "output_tokens", None),
        }
    if getattr(event, "index", None) is not None:
        d["index"] = event.index
    return d


# ── the call ────────────────────────────────────────────────────────────────

async def _stream_once(
    client: anthropic.AsyncAnthropic, req: ModelRequest, abort: asyncio.Event
) -> AsyncIterator[ModelEvent]:
    kwargs: dict[str, Any] = dict(
        model=req.model,
        max_tokens=req.max_tokens,
        system=req.system,
        messages=req.messages,
        # Adaptive thinking: the model decides when and how deeply to think. The
        # raw chain of thought is never returned; display picks between an empty
        # placeholder block ("omitted", the API default) and a readable summary.
        thinking={"type": "adaptive", "display": req.thinking_display},
        cache_control={"type": "ephemeral"},   # auto-cache the last cacheable block
    )
    if req.tools:
        kwargs["tools"] = req.tools
    if req.effort:
        kwargs["output_config"] = {"effort": req.effort}

    if req.fallbacks:
        stream_ctx = client.beta.messages.stream(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
    else:
        stream_ctx = client.messages.stream(**kwargs)

    call = next(_CALL_SEQ)
    started = time.perf_counter()
    _trace(
        "request", call=call, model=req.model, max_tokens=req.max_tokens,
        effort=req.effort, thinking_display=req.thinking_display, fallbacks=req.fallbacks,
        n_messages=len(req.messages), n_tools=len(req.tools),
        tools=[t.get("name") for t in req.tools],
        system=[{"chars": len(b.get("text", "")), "cached": "cache_control" in b} for b in req.system],
        history=_shape(req.messages),
    )

    async with stream_ctx as stream:
        async for event in stream:
            _trace("event", call=call, t=round((time.perf_counter() - started) * 1000, 1),
                   type=getattr(event, "type", "?"), **_event_detail(event))
            if abort.is_set():
                # Leaving the context manager closes the HTTP stream.
                raise ModelCallError("aborted", "interrupted while streaming")
            if event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    yield TextDelta(delta.text)
                elif delta.type == "thinking_delta" and delta.thinking:
                    yield ThinkingDelta(delta.thinking)
        message = await stream.get_final_message()
    _trace(
        "response", call=call, t=round((time.perf_counter() - started) * 1000, 1),
        stop_reason=message.stop_reason, model=getattr(message, "model", None),
        blocks=[
            {"type": b.type, "chars": len(getattr(b, b.type, "") or "") if b.type in ("text", "thinking") else None,
             "name": getattr(b, "name", None), "signature": bool(getattr(b, "signature", None))}
            for b in message.content
        ],
        usage={
            "input": message.usage.input_tokens,
            "cache_read": getattr(message.usage, "cache_read_input_tokens", 0),
            "cache_write": getattr(message.usage, "cache_creation_input_tokens", 0),
            "output": message.usage.output_tokens,
        },
    )
    yield ModelResponse(message)


async def query_model(
    client: anthropic.AsyncAnthropic, req: ModelRequest, abort: asyncio.Event
) -> AsyncIterator[ModelEvent]:
    """Stream one model turn with yield-based retry."""
    attempt = 0
    while True:
        try:
            async for event in _stream_once(client, req, abort):
                yield event
            return
        except TypeError as exc:
            # The SDK raises a bare TypeError when it cannot find any credential.
            if "authentication" not in str(exc).lower():
                raise
            raise ModelCallError(
                "model_error",
                "no credentials found. Set ANTHROPIC_API_KEY, or run `ant auth login`.",
            ) from exc
        except anthropic.APIError as exc:
            if isinstance(exc, anthropic.BadRequestError) and "prompt is too long" in str(exc).lower():
                raise ModelCallError("prompt_too_long", str(exc)) from exc
            if not _is_retryable(exc) or attempt >= MAX_RETRIES:
                raise ModelCallError("model_error", str(exc)) from exc
            attempt += 1
            delay = _backoff_seconds(exc, attempt)
            label = getattr(exc, "status_code", None) or type(exc).__name__
            yield StatusEvent(f"API error ({label}); retry {attempt}/{MAX_RETRIES} in {delay:.0f}s")
            if await _sleep_unless_aborted(delay, abort):
                raise ModelCallError("aborted", "interrupted during retry wait") from exc


def make_client() -> anthropic.AsyncAnthropic:
    """Credentials resolve from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`.

    max_retries=0 because query_model owns retries; two retry layers would
    hide the status events the UI is supposed to show.
    """
    return anthropic.AsyncAnthropic(max_retries=0)
