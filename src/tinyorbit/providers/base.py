"""Provider abstraction — the model-agnostic transport contract (SDK core).

tinyorbit owns its agent loop; a *Provider* owns one dialect of talking to a
model. The loop never imports a vendor SDK: it builds a neutral `ModelRequest`,
hands it to `provider.stream(...)`, and consumes a neutral event stream ending
in `ModelResponse`. Everything vendor-specific — wire format, auth, retry
taxonomy, feature flags — lives behind this seam.

Two orthogonal axes the design keeps separate:

  * dialect  — the wire format (Anthropic Messages vs OpenAI Chat). A Provider
               subclass owns one dialect.
  * transport — endpoint + auth (first-party API, Bedrock, an LLM gateway).
               Parameterized per Provider instance (base_url, credentials).

Canonical message shape is Anthropic-style content blocks, because that is what
the loop stores and re-sends verbatim. Non-Anthropic providers translate at
their own boundary and hand back a duck-typed message (see `NormalizedMessage`).

This module is provider-neutral and carries no vendor import; it is the piece
meant to live in the open-source core.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

# Output-token budget. p99 output is far below the default; the loop escalates
# to the larger cap on demand (max_tokens recovery).
DEFAULT_MAX_TOKENS = 16_000
ESCALATED_MAX_TOKENS = 64_000


# ── the neutral request ───────────────────────────────────────────────────────

@dataclass
class ModelRequest:
    """One model turn, expressed in vendor-neutral terms.

    `system`, `messages` and `tools` are in the canonical (Anthropic-shaped)
    representation; each Provider translates them to its own wire format.
    """

    model: str
    system: list[dict]
    messages: list[Any]
    tools: list[dict] = field(default_factory=list)
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str | None = None
    fallbacks: bool = True              # server-side refusal fallback, where supported
    thinking_display: str = "omitted"   # "summarized" returns readable reasoning


# ── the neutral event stream ──────────────────────────────────────────────────
# Retry progress is part of the stream (StatusEvent), not a side channel, so the
# UI can show "overloaded, retrying in 4s" the same way for every provider.

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
    message: Any     # a vendor Message, or a NormalizedMessage stand-in


ModelEvent = TextDelta | ThinkingDelta | StatusEvent | ModelResponse


class ModelCallError(Exception):
    """Errors the loop switches on. `reason` is provider-neutral so the loop
    stays dialect-agnostic; each Provider maps its SDK exceptions onto these."""

    reason: str   # "prompt_too_long" | "aborted" | "model_error"

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# ── canonical message blocks for non-Anthropic providers ──────────────────────
# The loop consumes the model's reply as an object with `.content` (a list of
# blocks), `.stop_reason`, `.usage`, and re-sends `.content` verbatim. The
# Anthropic SDK Message already fits this shape; other dialects build these
# lightweight stand-ins so the loop needs no special-casing.

@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ThinkingBlock:
    thinking: str
    # The opaque signature the API returns for a thinking block. Preserved so it
    # can be replayed verbatim on re-send: without it the API rejects a prior
    # thinking block (and requires one when the turn also called a tool).
    signature: str = ""
    type: str = "thinking"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class Usage:
    """Vendor-neutral usage, named to match what CostTracker.add() reads."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class NormalizedMessage:
    content: list[Any]
    stop_reason: str
    usage: Usage
    stop_details: Any = None
    role: str = "assistant"
    model: str | None = None


# ── capability model ──────────────────────────────────────────────────────────
# Rather than `if provider == "openai"` scattered through the request builder,
# a Provider declares what it supports and the encoder emits a feature only when
# the capability is present. Missing features degrade, they never error.

@dataclass(frozen=True)
class ProviderCapabilities:
    prompt_caching: bool = False      # Anthropic cache_control blocks
    adaptive_thinking: bool = False   # thinking={"type":"adaptive"}
    effort: bool = False              # output_config / reasoning_effort
    server_fallbacks: bool = False    # server-side refusal fallback beta
    context_1m: bool = False          # 1M-token context window


# ── the Provider contract ─────────────────────────────────────────────────────

class Provider:
    """One dialect of talking to a model. Holds its own SDK client (transport).

    Subclasses implement `stream()`; the loop only ever sees this interface.
    """

    name: str = "base"
    capabilities: ProviderCapabilities = ProviderCapabilities()

    async def stream(self, req: ModelRequest, abort: Any) -> AsyncIterator[ModelEvent]:
        raise NotImplementedError
        yield  # pragma: no cover — marks this an async generator for type checkers

    async def close(self) -> None:
        """Release transport resources (HTTP pool). Default: nothing to do."""

    def price_for(self, model: str) -> tuple[float, float] | None:
        """(input, output) USD per Mtok, or None if unknown. Optional."""
        return None


# ── registry + resolution ─────────────────────────────────────────────────────
# Providers self-register by name. Resolution picks one from an explicit name,
# the TINYORBIT_PROVIDER env var, or a model-id heuristic. Unknown names trigger
# a lazy import of `tinyorbit.providers.<name>` — that is how an opt-in provider
# (e.g. an internal gateway) plugs in without any generic file referencing it.

ProviderFactory = Callable[..., Provider]
_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    _REGISTRY[name] = factory


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def _infer_provider(model: str | None) -> str:
    """Best-effort dialect guess from a model id. Anthropic is the default."""
    m = (model or "").lower()
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "text-", "openai")):
        return "openai"
    return "anthropic"


def resolve_provider(*, provider: str | None = None, model: str | None = None, **opts: Any) -> Provider:
    """Instantiate the Provider for this run.

    Precedence: explicit `provider` arg → $TINYORBIT_PROVIDER → model-id
    heuristic. An unregistered name is lazily imported by convention so opt-in
    providers register themselves the first time they are asked for.
    """
    name = provider or os.environ.get("TINYORBIT_PROVIDER") or _infer_provider(model)

    if name not in _REGISTRY:
        try:
            importlib.import_module(f"tinyorbit.providers.{name}")
        except ImportError:
            pass

    factory = _REGISTRY.get(name)
    if factory is None:
        known = ", ".join(available_providers()) or "(none registered)"
        raise ModelCallError("model_error", f"unknown provider {name!r}; available: {known}")
    return factory(model=model, **opts)


# ── the default call_model seam ───────────────────────────────────────────────
# QueryDeps.call_model defaults to this. It keeps the (provider, req, abort)
# signature the loop and the scripted test model share, and simply delegates to
# the resolved Provider. Tests swap in their own call_model and never touch a
# real provider.

async def call_model(provider: Provider, req: ModelRequest, abort: Any) -> AsyncIterator[ModelEvent]:
    async for event in provider.stream(req, abort):
        yield event
