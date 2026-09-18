"""OpenAI dialect provider (SDK core).

Reaches any model behind an OpenAI-compatible API — first-party OpenAI or an
LLM gateway that fronts many models with `/v1/chat/completions`. This is the
"talk to any model" path: one dialect, lowest-common-denominator features (no
Anthropic prompt caching or adaptive-thinking blocks).

All the vendor weirdness is quarantined here as pure translation functions:

  * canonical (Anthropic-shaped) messages/tools/system  ->  OpenAI wire format
  * OpenAI streaming deltas + finish_reason              ->  neutral events +
                                                             a NormalizedMessage

The `openai` package is imported lazily so it stays an optional dependency; a
core install without it works fine until this provider is actually resolved.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from tinyorbit.providers.base import (
    ModelCallError,
    ModelRequest,
    ModelResponse,
    NormalizedMessage,
    Provider,
    ProviderCapabilities,
    StatusEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
    register_provider,
)

MAX_RETRIES = 4

# OpenAI finish_reason -> canonical (Anthropic) stop_reason.
_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


# ── outbound translation (canonical -> OpenAI wire) ───────────────────────────

def _block_type(b: Any) -> str:
    return b.get("type") if isinstance(b, dict) else getattr(b, "type", "")


def _block_attr(b: Any, name: str, default: Any = None) -> Any:
    return b.get(name, default) if isinstance(b, dict) else getattr(b, name, default)


def encode_system(system: list[dict]) -> str:
    """Flatten cached Anthropic system blocks into a single system string."""
    parts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
    return "\n\n".join(p for p in parts if p)


def encode_tools(tools: list[dict]) -> list[dict]:
    """Anthropic `{name, description, input_schema}` -> OpenAI function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def encode_messages(messages: list[Any]) -> list[dict]:
    """Canonical messages -> OpenAI chat messages.

    Handles str content and content-block lists whether the blocks are dicts
    (as the loop builds tool_result/assistant turns) or our block dataclasses
    (as this provider emits assistant turns for re-send). Anthropic packs tool
    results into one user message; OpenAI wants one `role:"tool"` message each.
    """
    out: list[dict] = []
    for m in messages:
        role = m["role"] if isinstance(m, dict) else getattr(m, "role", "user")
        content = m["content"] if isinstance(m, dict) else getattr(m, "content", "")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for b in content:
            bt = _block_type(b)
            if bt == "text":
                text_parts.append(_block_attr(b, "text", "") or "")
            elif bt == "tool_use":
                tool_calls.append({
                    "id": _block_attr(b, "id"),
                    "type": "function",
                    "function": {
                        "name": _block_attr(b, "name"),
                        "arguments": json.dumps(_block_attr(b, "input", {}) or {}),
                    },
                })
            elif bt == "tool_result":
                result = _block_attr(b, "content", "")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": _block_attr(b, "tool_use_id"),
                    "content": result if isinstance(result, str) else json.dumps(result),
                })
            # thinking blocks are dropped: OpenAI has no re-sendable equivalent.

        if role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        else:
            if text_parts:
                out.append({"role": role, "content": "\n".join(text_parts)})
            out.extend(tool_results)   # tool results are their own messages
    return out


def build_request(req: ModelRequest) -> dict:
    """Assemble the kwargs for chat.completions.create()."""
    messages = encode_messages(req.messages)
    system = encode_system(req.system)
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    kwargs: dict[str, Any] = {
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "stream": True,
        # ask the final chunk to carry usage so cost accounting works
        "stream_options": {"include_usage": True},
    }
    if req.tools:
        kwargs["tools"] = encode_tools(req.tools)
        kwargs["tool_choice"] = "auto"
    if req.effort:
        # reasoning models accept reasoning_effort; non-reasoning models ignore
        # or reject it — the gateway/model decides. Passed through, not assumed.
        kwargs["reasoning_effort"] = req.effort
    return kwargs


# ── inbound translation (OpenAI stream -> neutral response) ───────────────────

def _finalize(text: str, tool_calls: dict[int, dict], finish_reason: str | None, usage: Any) -> NormalizedMessage:
    """Build a loop-compatible message from accumulated stream state."""
    content: list[Any] = []
    if text:
        content.append(TextBlock(text=text))
    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        args = tc.get("arguments", "") or "{}"
        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        content.append(ToolUseBlock(id=tc.get("id", f"call_{idx}"), name=tc.get("name", ""), input=parsed))

    stop_reason = _STOP_REASON.get(finish_reason or "", "end_turn")
    if tool_calls and stop_reason == "end_turn":
        stop_reason = "tool_use"   # some gateways report "stop" alongside tool calls

    u = Usage(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
    return NormalizedMessage(content=content, stop_reason=stop_reason, usage=u)


# ── the provider ──────────────────────────────────────────────────────────────

class OpenAIProvider(Provider):
    name = "openai"
    capabilities = ProviderCapabilities(effort=True)   # no caching/thinking/fallbacks

    def __init__(self, *, client: Any = None, model: str | None = None,
                 base_url: str | None = None, api_key: str | None = None, **_: Any):
        # Build the client lazily on first stream() so resolving this provider
        # (dialect selection) does not require the optional `openai` package to
        # be installed — only actually calling the model does.
        self._client = client
        self._base_url = base_url
        self._api_key = api_key

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover — depends on optional extra
                raise ModelCallError(
                    "model_error",
                    "the openai provider needs the 'openai' package: pip install tinyorbit[openai]",
                ) from exc
            kwargs: dict[str, Any] = {"max_retries": 0}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            await close()

    async def _stream_once(self, req: ModelRequest, abort: asyncio.Event) -> AsyncIterator[Any]:
        text_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        finish_reason: str | None = None
        usage: Any = None

        stream = await self._ensure_client().chat.completions.create(**build_request(req))
        async for chunk in stream:
            if abort.is_set():
                raise ModelCallError("aborted", "interrupted while streaming")
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
                yield TextDelta(delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = tool_calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments

        yield ModelResponse(_finalize("".join(text_parts), tool_calls, finish_reason, usage))

    async def stream(self, req: ModelRequest, abort: asyncio.Event) -> AsyncIterator[Any]:
        """Stream one turn with retry. Maps OpenAI errors onto neutral reasons."""
        try:
            import openai
        except ImportError:   # pragma: no cover
            openai = None

        attempt = 0
        while True:
            try:
                async for event in self._stream_once(req, abort):
                    yield event
                return
            except ModelCallError:
                raise
            except Exception as exc:   # noqa: BLE001 — normalize any SDK error
                if openai is not None and isinstance(exc, openai.BadRequestError) \
                        and "context length" in str(exc).lower():
                    raise ModelCallError("prompt_too_long", str(exc)) from exc
                retryable = openai is not None and isinstance(
                    exc, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
                )
                if not retryable or attempt >= MAX_RETRIES:
                    raise ModelCallError("model_error", str(exc)) from exc
                attempt += 1
                delay = min(2 ** attempt, 30)
                yield StatusEvent(f"API error ({type(exc).__name__}); retry {attempt}/{MAX_RETRIES} in {delay:.0f}s")
                try:
                    await asyncio.wait_for(abort.wait(), timeout=delay)
                    raise ModelCallError("aborted", "interrupted during retry wait") from exc
                except asyncio.TimeoutError:
                    pass


register_provider("openai", OpenAIProvider)
