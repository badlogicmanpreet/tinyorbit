"""MCP server configuration — the neutral description of one upstream server.

An `McpServerConfig` names a server, its endpoint, and how to build the HTTP
client that talks to it. `http_factory` is the transport-injection seam: the
generic client never hard-codes TLS/auth, so an opt-in gateway can supply a
mutually-authenticated (mTLS) client here without the core knowing anything
about it (mirrors how a Provider parameterizes its own transport).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class McpServerConfig:
    """One upstream MCP server, described in transport-neutral terms.

    * `headers` are sent on every request (a plain bearer token lives here).
    * `http_factory`, when set, returns a ready-to-use `httpx.AsyncClient`; it
      overrides `headers` construction and is how mTLS is injected. When None,
      the client builds a default client carrying `headers`.
    """

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    transport: str = "http"                       # streamable-HTTP is the only wire we speak
    http_factory: Callable[[], Any] | None = None  # () -> httpx.AsyncClient
    read_timeout: float = 120.0
