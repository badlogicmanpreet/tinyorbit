"""Connect a set of MCP servers and expose their tools as one pool.

`connect_mcp_servers` is the async startup seam the REPL calls after it has an
event loop (bootstrap.setup() is sync; MCP discovery is not). A server that
fails to connect is skipped, not fatal — a broken gateway must not sink the
whole session — and its error is recorded for the caller to surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from tinyorbit.mcp.client import McpClient
from tinyorbit.mcp.config import McpServerConfig
from tinyorbit.mcp.tools import McpTool, build_mcp_tools


def _default_http_factory(cfg: McpServerConfig) -> Callable[[], Any]:
    """Build a plain httpx client carrying the server's static headers.

    httpx is imported lazily so the core never needs it until a server is
    actually configured (it is an optional extra, not a core dependency)."""
    def factory() -> Any:
        import httpx
        return httpx.AsyncClient(headers=cfg.headers or {})
    return factory


@dataclass
class McpSession:
    """The live result of connecting one or more servers."""

    clients: list[McpClient] = field(default_factory=list)
    tools: list[McpTool] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    async def aclose(self) -> None:
        for client in self.clients:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                pass


async def connect_mcp_servers(configs: list[McpServerConfig]) -> McpSession:
    """Connect each configured server, discover its tools, aggregate the pool."""
    session = McpSession()
    for cfg in configs:
        if cfg.transport != "http":
            session.errors.append(f"{cfg.name}: unsupported transport {cfg.transport!r}")
            continue
        factory = cfg.http_factory or _default_http_factory(cfg)
        client = McpClient(cfg.name, cfg.url, http_factory=factory, read_timeout=cfg.read_timeout)
        try:
            await client.connect()
            specs = await client.list_tools()
        except Exception as exc:  # noqa: BLE001 — one bad server is not fatal
            session.errors.append(f"{cfg.name}: {type(exc).__name__}: {exc}")
            await client.close()
            continue
        session.clients.append(client)
        session.tools.extend(build_mcp_tools(cfg.name, specs, client))
    return session


def filter_tools(tools: list[McpTool], allow: list[str] | None = None,
                 deny: list[str] | None = None) -> list[McpTool]:
    """Apply allow/deny globs over namespaced tool names (deny wins).

    Globs match the full `mcp__<server>__<tool>` name, e.g. `mcp__gw__*` or a
    single `mcp__gw__dangerous_tool`. An empty/absent allow list allows all.
    Thin wrapper over the generic pool filter so MCP and `--allowed-tools`
    behave identically.
    """
    from tinyorbit.tools.registry import select_tools

    return select_tools(tools, allow, deny)
