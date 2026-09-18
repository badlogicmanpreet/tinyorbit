"""Provider layer — model-agnostic transport (SDK core).

Importing this package registers the open dialects (anthropic, openai). Other
providers (e.g. an internal gateway) self-register on first use via the lazy
import in `resolve_provider`, so nothing here needs to name them.
"""

# Register the built-in dialects by importing their modules.
from tinyorbit.providers import anthropic as _anthropic  # noqa: E402,F401
from tinyorbit.providers import openai as _openai  # noqa: E402,F401
from tinyorbit.providers.base import (
    DEFAULT_MAX_TOKENS,
    ESCALATED_MAX_TOKENS,
    ModelCallError,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    NormalizedMessage,
    Provider,
    ProviderCapabilities,
    StatusEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    Usage,
    available_providers,
    call_model,
    register_provider,
    resolve_provider,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "ESCALATED_MAX_TOKENS",
    "ModelCallError",
    "ModelEvent",
    "ModelRequest",
    "ModelResponse",
    "NormalizedMessage",
    "Provider",
    "ProviderCapabilities",
    "StatusEvent",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "ToolUseBlock",
    "Usage",
    "available_providers",
    "call_model",
    "register_provider",
    "resolve_provider",
]
