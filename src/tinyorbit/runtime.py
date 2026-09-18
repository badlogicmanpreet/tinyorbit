"""Embedding seam — drive the agent loop as a library, not a terminal (Ch 13).

The REPL is one consumer of the query loop; a service is another — an event
bus, an HTTP handler, a test harness. They all need the same three things a
turn depends on: a configured provider, a session that persists across turns
(so the connection pool and prompt cache survive), and a way to observe events
as they stream. Runtime packages exactly that and nothing terminal-specific.

Instead of the REPL's ANSI Renderer, the embedder passes a plain callback to
`submit(prompt, on_event=...)` and receives the identical event objects the
terminal would render (TextDelta, ToolStarted, Done, …). The callback is
synchronous — to hand events to an async consumer, push them onto a queue with
`put_nowait` and drain them elsewhere; that keeps the loop from awaiting an
external sink mid-stream.

Lifecycle mirrors bootstrap.launch(): the caller runs init() + setup() to
populate `config.capabilities`, then drives Runtime as an async context
manager. All startup machinery (provider resolution, MCP connect, tool-pool
filtering, context assembly) is shared with the REPL rather than re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from tinyorbit.permissions import PermissionPolicy
from tinyorbit.providers import resolve_provider
from tinyorbit.repl import _apply_tool_filter, _make_context, _open_session, _start_mcp, run_turn
from tinyorbit.state import SessionState

if TYPE_CHECKING:
    from tinyorbit.bootstrap import Config

# The event stream is untyped on purpose: the sink sees whatever query() yields,
# the same objects the terminal Renderer switches on.
EventSink = Callable[[object], None]


@dataclass
class TurnResult:
    """What a submitted turn produced. Events already reached the sink live."""

    text: str
    session_id: str
    turns: int


class Runtime:
    """A persistent agent session an embedder drives programmatically.

    Non-interactive by construction: with no stdin there is nobody to answer a
    permission prompt, so the policy's `ask` is left None and decisions fall
    through to the mode and allow-rules — exactly like --print. Pass a custom
    policy to override.
    """

    def __init__(self, config: "Config", *, policy: PermissionPolicy | None = None,
                 session: SessionState | None = None) -> None:
        self._config = config
        self._policy = policy or PermissionPolicy(
            mode=config.permission_mode, ask=None, allow_rules=config.allow_rules,
        )
        self._session = session or _open_session(config)
        self._provider = None
        self._mcp = None
        self._ctx = None

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def session(self) -> SessionState:
        return self._session

    async def start(self) -> "Runtime":
        """Resolve the provider, connect MCP, filter the pool, build the context."""
        self._provider = resolve_provider(provider=self._config.provider, model=self._config.model)
        self._mcp = await _start_mcp(self._config)
        _apply_tool_filter(self._config)
        self._ctx = _make_context(self._config, self._policy, self._provider, self._session)
        return self

    async def submit(self, prompt: str, on_event: EventSink | None = None) -> TurnResult:
        """Run one turn to completion, streaming events to on_event if given."""
        if self._ctx is None:
            raise RuntimeError("Runtime.start() must be called before submit()")
        # A fresh abort gate per turn, matching the REPL: a caller that wires a
        # cancel signal to ctx.abort.set can interrupt the turn in flight.
        self._ctx.abort.clear()
        text = await run_turn(
            self._config, self._session, self._ctx, self._provider, prompt, sink=on_event,
        )
        return TurnResult(text=text, session_id=self._session.session_id, turns=self._session.turns)

    async def aclose(self) -> None:
        if self._mcp is not None:
            await self._mcp.aclose()
            self._mcp = None
        if self._provider is not None:
            await self._provider.close()
            self._provider = None

    async def __aenter__(self) -> "Runtime":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
