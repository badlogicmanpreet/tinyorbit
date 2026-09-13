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
import random
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


# ── the call ────────────────────────────────────────────────────────────────

async def _stream_once(
    client: anthropic.AsyncAnthropic, req: ModelRequest, abort: asyncio.Event
) -> AsyncIterator[ModelEvent]:
    kwargs: dict[str, Any] = dict(
        model=req.model,
        max_tokens=req.max_tokens,
        system=req.system,
        messages=req.messages,
        thinking={"type": "adaptive"},
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

    async with stream_ctx as stream:
        async for event in stream:
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
