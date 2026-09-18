"""Anthropic dialect provider (SDK core).

The first-class, reference provider. `stream()` is the original api.py transport
verbatim in behavior — yield-based retry, adaptive thinking, prompt caching,
server-side fallbacks, SSE tracing — now wrapped in a Provider so the loop can
swap in other dialects. Canonical message shape is Anthropic's, so this dialect
is effectively passthrough: no translation on the way in or out.

Transport is parameterized: the default client hits the first-party API, but a
caller (e.g. an internal gateway provider) may inject a pre-built client with a
different base_url / auth. That is the single chokepoint for Bedrock/gateway.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import time
from typing import Any, AsyncIterator

from tinyorbit.providers.base import (
    ModelCallError,
    ModelRequest,
    ModelResponse,
    Provider,
    ProviderCapabilities,
    StatusEvent,
    TextDelta,
    ThinkingDelta,
    register_provider,
)

MAX_RETRIES = 4
RETRYABLE_STATUS = {408, 409, 429, 529}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# (input, output) USD per Mtok. Cache reads bill at 0.1x input, writes at 1.25x.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


# ── lazy SDK import ─────────────────────────────────────────────────────────
# The anthropic SDK is an optional extra: it is the transport for THIS dialect
# only, imported on first use so a build that never talks to Anthropic (an
# OpenAI-dialect model, an internal gateway) needs neither the dependency nor
# the import. Registration below runs without it — only reaching a real client
# does. Mirrors how the OpenAI provider treats its own SDK.

def _anthropic() -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover — install tinyorbit[anthropic]
        raise ModelCallError(
            "model_error",
            "the Anthropic dialect needs the anthropic SDK: pip install 'tinyorbit[anthropic]'",
        ) from exc
    return anthropic


# ── retry policy ──────────────────────────────────────────────────────────────
# These run only while handling an anthropic exception, so the SDK is already
# imported; a plain local import just rebinds the cached module.

def _is_retryable(exc: Any) -> bool:
    import anthropic
    if isinstance(exc, anthropic.APIConnectionError):   # includes timeouts
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS or exc.status_code >= 500
    return False


def _backoff_seconds(exc: Any, attempt: int) -> float:
    import anthropic
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


# ── stream tracing ────────────────────────────────────────────────────────────
# Set TINYORBIT_TRACE=<path> to append one JSON line per SSE event. Off by
# default, so the streaming hot path allocates nothing extra in normal use.

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


# ── the provider ──────────────────────────────────────────────────────────────

class AnthropicProvider(Provider):
    name = "anthropic"
    capabilities = ProviderCapabilities(
        prompt_caching=True, adaptive_thinking=True, effort=True,
        server_fallbacks=True, context_1m=True,
    )

    def __init__(self, *, client: Any = None, model: str | None = None, **_: Any):
        # Credentials resolve from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN /
        # `ant auth login`. max_retries=0 because stream() owns retries; two
        # retry layers would hide the status events the UI is supposed to show.
        self._client = client or _anthropic().AsyncAnthropic(max_retries=0)

    def price_for(self, model: str) -> tuple[float, float] | None:
        return PRICES_PER_MTOK.get(model)

    async def close(self) -> None:
        await self._client.close()

    async def _stream_once(self, req: ModelRequest, abort: asyncio.Event) -> AsyncIterator[Any]:
        caps = self.capabilities
        kwargs: dict[str, Any] = dict(
            model=req.model,
            max_tokens=req.max_tokens,
            system=req.system,
            messages=req.messages,
        )
        if caps.adaptive_thinking:
            # Adaptive thinking: the model decides when and how deeply to think.
            # display picks between an empty placeholder ("omitted") and a summary.
            kwargs["thinking"] = {"type": "adaptive", "display": req.thinking_display}
        if caps.prompt_caching:
            kwargs["cache_control"] = {"type": "ephemeral"}   # auto-cache last cacheable block
        if req.tools:
            kwargs["tools"] = req.tools
        if req.effort and caps.effort:
            kwargs["output_config"] = {"effort": req.effort}

        if req.fallbacks and caps.server_fallbacks:
            stream_ctx = self._client.beta.messages.stream(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            stream_ctx = self._client.messages.stream(**kwargs)

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
            usage={
                "input": message.usage.input_tokens,
                "cache_read": getattr(message.usage, "cache_read_input_tokens", 0),
                "cache_write": getattr(message.usage, "cache_creation_input_tokens", 0),
                "output": message.usage.output_tokens,
            },
        )
        yield ModelResponse(message)

    async def stream(self, req: ModelRequest, abort: asyncio.Event) -> AsyncIterator[Any]:
        """Stream one model turn with yield-based retry."""
        anthropic = _anthropic()   # bind once so the except clause has the classes
        attempt = 0
        while True:
            try:
                async for event in self._stream_once(req, abort):
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


register_provider("anthropic", AnthropicProvider)


# ── back-compat free functions ────────────────────────────────────────────────
# Kept so tinyorbit.api can re-export the pre-provider API surface unchanged.

def make_client() -> Any:
    return _anthropic().AsyncAnthropic(max_retries=0)


async def query_model(client: Any, req: ModelRequest, abort: asyncio.Event) -> AsyncIterator[Any]:
    async for event in AnthropicProvider(client=client).stream(req, abort):
        yield event
