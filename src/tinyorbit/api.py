"""Back-compat facade for the API layer.

The transport moved to `tinyorbit.providers` when tinyorbit grew from a
single-vendor agent into a model-agnostic SDK. This module re-exports the
pre-provider surface so older imports (and the test suite, which drives the
loop through the QueryDeps seam) keep working unchanged.

New code should import from `tinyorbit.providers`.
"""

from __future__ import annotations

from tinyorbit.providers.anthropic import make_client, query_model
from tinyorbit.providers.base import (
    DEFAULT_MAX_TOKENS,
    ESCALATED_MAX_TOKENS,
    ModelCallError,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    StatusEvent,
    TextDelta,
    ThinkingDelta,
    call_model,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "ESCALATED_MAX_TOKENS",
    "ModelCallError",
    "ModelEvent",
    "ModelRequest",
    "ModelResponse",
    "StatusEvent",
    "TextDelta",
    "ThinkingDelta",
    "call_model",
    "make_client",
    "query_model",
]
