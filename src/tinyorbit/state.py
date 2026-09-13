"""Session state — what survives across turns (Ch 3).

Ch 3 describes two tiers: a mutable singleton (session-wide facts) and a
reactive store (UI). We only need the first tier: the message history, the
cost ledger, and the identifiers a session carries.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

# (input, output) USD per million tokens. Cache reads bill at 0.1x input,
# cache writes at 1.25x input.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


@dataclass
class CostTracker:
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    last_context_tokens: int = 0     # how big the last request was; drives auto-compact

    def add(self, usage: Any) -> None:
        self.api_calls += 1
        inp = getattr(usage, "input_tokens", 0) or 0
        read = getattr(usage, "cache_read_input_tokens", 0) or 0
        write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.input_tokens += inp
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += read
        self.cache_write_tokens += write
        self.last_context_tokens = inp + read + write

    def estimate_usd(self, model: str) -> float | None:
        prices = PRICES_PER_MTOK.get(model)
        if prices is None:
            return None
        inp, out = prices
        return (
            self.input_tokens * inp
            + self.cache_read_tokens * inp * 0.1
            + self.cache_write_tokens * inp * 1.25
            + self.output_tokens * out
        ) / 1_000_000

    def summary(self, model: str) -> str:
        usd = self.estimate_usd(model)
        cost = f"~${usd:.4f}" if usd is not None else "n/a (unknown model price)"
        return (
            f"{self.api_calls} API calls | in {self.input_tokens:,} "
            f"(cache read {self.cache_read_tokens:,}, write {self.cache_write_tokens:,}) "
            f"| out {self.output_tokens:,} | {cost}"
        )


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[Any] = field(default_factory=list)
    cost: CostTracker = field(default_factory=CostTracker)
    turns: int = 0
