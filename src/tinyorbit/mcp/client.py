"""A minimal MCP client over streamable-HTTP (JSON-RPC 2.0).

tinyorbit speaks the MCP wire itself rather than pulling in the `mcp` SDK, for
the same reason it owns its agent loop: one small, auditable transport with no
extra dependency (only httpx, and only when a server is actually configured).

Streamable-HTTP is one POST endpoint. Each JSON-RPC message is POSTed with
`Accept: application/json, text/event-stream`; the server answers with either a
single JSON body or an SSE stream of `data:` frames. We read the whole response
and pick the frame whose id matches our request. The `initialize` handshake
returns an optional `Mcp-Session-Id` that must ride along on every later call.

The HTTP client is injected via a factory (see `McpServerConfig.http_factory`),
so mTLS/auth is a transport concern the client stays ignorant of — the same
dialect/transport split the Provider layer uses.
"""

from __future__ import annotations

import json
from typing import Any, Callable

_PROTOCOL_VERSION = "2025-03-26"
_CLIENT_INFO = {"name": "tinyorbit", "version": "0.1"}


class McpError(Exception):
    """A JSON-RPC error, a transport failure, or a malformed response."""


def parse_sse(text: str) -> list[dict]:
    """Collect JSON payloads from `data:` lines of an SSE body.

    MCP frames one JSON-RPC message per event; a data field may in theory span
    multiple `data:` lines (concatenated), so we buffer until a blank line.
    """
    messages: list[dict] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        raw = "".join(buf)
        buf.clear()
        raw = raw.strip()
        if raw:
            messages.append(json.loads(raw))

    for line in text.splitlines():
        if line.startswith("data:"):
            buf.append(line[5:].lstrip())
        elif line == "":
            flush()
    flush()
    return messages


def extract_result(messages: list[dict], req_id: int) -> Any:
    """Return the `result` of the response matching `req_id`, or raise on error.

    Server→client requests/notifications interleaved in the stream (which carry
    a different id or none) are ignored — we only resolve our own call.
    """
    for msg in messages:
        if msg.get("id") != req_id:
            continue
        if "error" in msg:
            err = msg["error"]
            raise McpError(f"{err.get('code')}: {err.get('message', err)}")
        return msg.get("result")
    raise McpError(f"no JSON-RPC response for id {req_id} in {len(messages)} message(s)")


class McpClient:
    """One session against a single streamable-HTTP MCP server."""

    def __init__(self, name: str, url: str, *, http_factory: Callable[[], Any],
                 read_timeout: float = 120.0) -> None:
        self.name = name
        self.url = url
        self._http_factory = http_factory
        self._read_timeout = read_timeout
        self._http: Any = None
        self._session_id: str | None = None
        self._next_id = 0

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def connect(self) -> None:
        """Open the transport and run the MCP initialize handshake."""
        self._http = self._http_factory()
        await self._request("initialize", {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": _CLIENT_INFO,
        })
        # notifications/initialized has no id and expects no response body.
        await self._notify("notifications/initialized")

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ── MCP methods ──────────────────────────────────────────────────────────
    async def list_tools(self) -> list[dict]:
        result = await self._request("tools/list", {})
        return list((result or {}).get("tools", []))

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Return the raw MCP tool result: {content: [...], isError?: bool}."""
        return await self._request("tools/call", {"name": name, "arguments": arguments}) or {}

    # ── transport ──────────────────────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            # After initialize, echo the negotiated protocol version (spec advice).
            "MCP-Protocol-Version": _PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _post(self, payload: dict) -> Any:
        if self._http is None:
            raise McpError(f"{self.name}: client is not connected")
        resp = await self._http.post(
            self.url, headers=self._headers(), json=payload, timeout=self._read_timeout,
        )
        # The server assigns the session on the initialize response.
        sid = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
        if sid:
            self._session_id = sid
        status = resp.status_code
        if status == 202:            # accepted notification: no body to parse
            return resp
        if status >= 400:
            raise McpError(f"{self.name}: HTTP {status}: {(resp.text or '')[:300]}")
        return resp

    async def _request(self, method: str, params: dict) -> Any:
        self._next_id += 1
        req_id = self._next_id
        resp = await self._post({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        ctype = (resp.headers.get("content-type") or "").lower()
        if "text/event-stream" in ctype:
            messages = parse_sse(resp.text)
        else:
            body = resp.json()
            messages = body if isinstance(body, list) else [body]
        return extract_result(messages, req_id)

    async def _notify(self, method: str, params: dict | None = None) -> None:
        await self._post({"jsonrpc": "2.0", "method": method, "params": params or {}})
