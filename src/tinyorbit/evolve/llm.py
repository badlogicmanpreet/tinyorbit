"""The one model call the evolve layer needs: system + user text in, text out.

Reflection and mutation are single-shot completions, not agent loops, so they
get a narrower seam than QueryDeps: `Complete = async (system, user) -> str`.
Tests pass a scripted function; a live run wraps a Provider with
`provider_complete`, which streams through the same neutral ModelRequest the
loop uses and books tokens on a CostTracker so evolution cost is visible.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from tinyorbit.providers.base import ModelRequest, ModelResponse
from tinyorbit.state import CostTracker

Complete = Callable[[str, str], Awaitable[str]]


def provider_complete(provider: Any, model: str, *, cost: CostTracker | None = None,
                      max_tokens: int = 16_000, effort: str | None = None) -> Complete:
    async def complete(system: str, user: str) -> str:
        req = ModelRequest(
            model=model,
            system=[{"type": "text", "text": system}],
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            effort=effort,
        )
        text = ""
        async for event in provider.stream(req, asyncio.Event()):
            if isinstance(event, ModelResponse):
                if cost is not None:
                    cost.add(event.message.usage)
                text = "".join(b.text for b in event.message.content if getattr(b, "type", "") == "text")
        return text

    return complete
