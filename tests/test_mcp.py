"""MCP client layer tests — generic SDK core. No network.

The streamable-HTTP transport is driven with a fake httpx-shaped client that
records POSTs and returns canned JSON/SSE responses, so the JSON-RPC framing,
session handling, tool adaptation, discovery, and glob filtering are all
exercised without a socket.
"""

from __future__ import annotations

import pytest

from tinyorbit.mcp import (
    McpClient,
    McpError,
    McpServerConfig,
    McpTool,
    connect_mcp_servers,
    filter_tools,
    mcp_tool_name,
    parse_mcp_json,
    register_mcp_source,
    resolve_mcp_source,
)
from tinyorbit.mcp.client import extract_result, parse_sse
from tinyorbit.mcp.discovery import discover_mcp_servers, expand_env
from tinyorbit.mcp.tools import flatten_content

# ── fake transport ─────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status_code=200, headers=None, json_body=None, text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self._json = json_body
        self.text = text

    def json(self):
        return self._json


class FakeHttp:
    """An httpx.AsyncClient stand-in: `post` delegates to a responder callable."""

    def __init__(self, responder):
        self._responder = responder
        self.posts = []
        self.closed = False

    async def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers, "json": json})
        return self._responder(json)

    async def aclose(self):
        self.closed = True


def _json_ok(payload, result, headers=None):
    h = {"content-type": "application/json"}
    h.update(headers or {})
    return FakeResponse(200, h, json_body={"jsonrpc": "2.0", "id": payload["id"], "result": result})


def _server_responder(tools=None, tool_result=None):
    tools = tools if tools is not None else [
        {"name": "search", "description": "find things", "inputSchema": {"type": "object"}},
    ]
    tool_result = tool_result or {"content": [{"type": "text", "text": "pong"}]}

    def responder(payload):
        method = payload.get("method")
        if method == "initialize":
            return _json_ok(payload, {"protocolVersion": "2025-03-26"}, {"mcp-session-id": "S-1"})
        if method == "notifications/initialized":
            return FakeResponse(202, {}, text="")
        if method == "tools/list":
            return _json_ok(payload, {"tools": tools})
        if method == "tools/call":
            return _json_ok(payload, tool_result)
        raise AssertionError(f"unexpected method {method}")

    return responder


# ── SSE + result extraction ─────────────────────────────────────────────────────

def test_parse_sse_collects_data_frames():
    body = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
    assert parse_sse(body) == [{"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}]


def test_extract_result_matches_id_and_raises_on_error():
    msgs = [
        {"jsonrpc": "2.0", "method": "notifications/progress"},   # no id, ignored
        {"jsonrpc": "2.0", "id": 5, "result": {"value": 42}},
    ]
    assert extract_result(msgs, 5) == {"value": 42}
    with pytest.raises(McpError):
        extract_result([{"id": 5, "error": {"code": -32000, "message": "boom"}}], 5)
    with pytest.raises(McpError):
        extract_result([], 5)


# ── client lifecycle ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_client_connect_captures_session_and_lists_and_calls():
    http = FakeHttp(_server_responder())
    client = McpClient("gw", "https://gw/mcp", http_factory=lambda: http)
    await client.connect()

    # initialize + notifications/initialized both posted
    methods = [p["json"]["method"] for p in http.posts]
    assert methods == ["initialize", "notifications/initialized"]

    tools = await client.list_tools()
    assert tools[0]["name"] == "search"
    # session id from the initialize response now rides on later requests
    assert http.posts[-1]["headers"]["Mcp-Session-Id"] == "S-1"
    assert http.posts[-1]["headers"]["Accept"] == "application/json, text/event-stream"

    result = await client.call_tool("search", {"q": "hi"})
    assert result["content"][0]["text"] == "pong"

    await client.close()
    assert http.closed is True


@pytest.mark.asyncio
async def test_client_reads_sse_response():
    def responder(payload):
        if payload.get("method") == "initialize":
            return _json_ok(payload, {"protocolVersion": "2025-03-26"})
        if payload.get("method") == "notifications/initialized":
            return FakeResponse(202, {}, text="")
        # tools/list served as an SSE stream
        sse = ('data: {"jsonrpc":"2.0","id":%d,"result":{"tools":[{"name":"t"}]}}\n\n'
               % payload["id"])
        return FakeResponse(200, {"content-type": "text/event-stream"}, text=sse)

    client = McpClient("gw", "https://gw/mcp", http_factory=lambda: FakeHttp(responder))
    await client.connect()
    tools = await client.list_tools()
    assert tools == [{"name": "t"}]


@pytest.mark.asyncio
async def test_client_raises_on_http_error():
    def responder(payload):
        return FakeResponse(500, {"content-type": "application/json"}, text="upstream boom")

    client = McpClient("gw", "https://gw/mcp", http_factory=lambda: FakeHttp(responder))
    with pytest.raises(McpError) as ei:
        await client.connect()
    assert "500" in str(ei.value)


# ── tool adaptation ─────────────────────────────────────────────────────────────

def test_flatten_content_joins_text_and_flags_error():
    text, is_error = flatten_content({"content": [{"type": "text", "text": "a"},
                                                  {"type": "text", "text": "b"}]})
    assert text == "a\nb" and is_error is False
    text, is_error = flatten_content({"content": [{"type": "text", "text": "nope"}], "isError": True})
    assert is_error is True
    # non-text block is rendered, not dropped
    text, _ = flatten_content({"content": [{"type": "image", "data": "x"}]})
    assert "image" in text


def test_mcp_tool_shape_and_annotations():
    spec = {"name": "read_db", "description": "reads", "inputSchema": {"type": "object"},
            "annotations": {"readOnlyHint": True}}
    tool = McpTool("gw", spec, client=None)  # type: ignore[arg-type]
    assert tool.name == mcp_tool_name("gw", "read_db") == "mcp__gw__read_db"
    assert tool.input_schema == {"type": "object"}
    assert tool.is_read_only({}) is True and tool.is_concurrency_safe({}) is True


@pytest.mark.asyncio
async def test_mcp_tool_call_proxies_to_client():
    class StubClient:
        async def call_tool(self, name, args):
            assert name == "search" and args == {"q": "x"}
            return {"content": [{"type": "text", "text": "hit"}]}

    tool = McpTool("gw", {"name": "search"}, client=StubClient())  # type: ignore[arg-type]
    result = await tool.call({"q": "x"}, ctx=None)  # type: ignore[arg-type]
    assert result.data == "hit" and result.is_error is False


@pytest.mark.asyncio
async def test_mcp_tool_call_surfaces_error():
    class BoomClient:
        async def call_tool(self, name, args):
            raise McpError("nope")

    tool = McpTool("gw", {"name": "search"}, client=BoomClient())  # type: ignore[arg-type]
    result = await tool.call({}, ctx=None)  # type: ignore[arg-type]
    assert result.is_error is True and "nope" in result.data


# ── session: connect many, skip failures, filter ────────────────────────────────

@pytest.mark.asyncio
async def test_connect_mcp_servers_aggregates_and_skips_failures():
    good = McpServerConfig(name="ok", url="https://ok/mcp",
                           http_factory=lambda: FakeHttp(_server_responder()))

    def broken_factory():
        def responder(payload):
            raise RuntimeError("connection refused")
        return FakeHttp(responder)

    bad = McpServerConfig(name="bad", url="https://bad/mcp", http_factory=broken_factory)

    session = await connect_mcp_servers([good, bad])
    assert [t.name for t in session.tools] == ["mcp__ok__search"]
    assert len(session.clients) == 1
    assert any("bad" in e for e in session.errors)
    await session.aclose()


@pytest.mark.asyncio
async def test_connect_skips_unsupported_transport():
    cfg = McpServerConfig(name="stdio", url="x", transport="stdio")
    session = await connect_mcp_servers([cfg])
    assert session.tools == [] and any("stdio" in e for e in session.errors)


def test_filter_tools_allow_and_deny_globs():
    tools = [McpTool("gw", {"name": n}, client=None) for n in ("a", "b", "danger")]  # type: ignore[arg-type]

    def names(ts):
        return [t.name for t in ts]

    assert names(filter_tools(tools, allow=["mcp__gw__*"])) == ["mcp__gw__a", "mcp__gw__b", "mcp__gw__danger"]
    assert names(filter_tools(tools, deny=["mcp__gw__danger"])) == ["mcp__gw__a", "mcp__gw__b"]
    assert names(filter_tools(tools, allow=["mcp__gw__a"])) == ["mcp__gw__a"]


# ── discovery ────────────────────────────────────────────────────────────────────

def test_expand_env(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    monkeypatch.delenv("MISSING", raising=False)
    assert expand_env("${FOO}/x") == "bar/x"
    assert expand_env("${MISSING:-http://default/mcp}") == "http://default/mcp"
    assert expand_env("${MISSING}") == ""


def test_parse_mcp_json_variants_and_skips_non_http(monkeypatch):
    monkeypatch.setenv("GW", "https://gw.internal/mcp")
    data = {"mcpServers": {
        "a": {"type": "http", "url": "${GW}", "headers": {"Authorization": "Bearer ${TOK:-t0}"}},
        "b": {"type": "stdio", "command": "x"},          # skipped: not http
        "c": {"type": "http"},                            # skipped: no url
    }}
    configs = parse_mcp_json(data)
    assert len(configs) == 1
    assert configs[0].name == "a"
    assert configs[0].url == "https://gw.internal/mcp"
    assert configs[0].headers == {"Authorization": "Bearer t0"}


def test_discover_is_empty_without_env(monkeypatch):
    monkeypatch.delenv("TINYORBIT_MCP_CONFIG", raising=False)
    monkeypatch.delenv("TINYORBIT_MCP_SOURCES", raising=False)
    assert discover_mcp_servers() == []


def test_named_source_registry_roundtrip_and_unknown():
    marker = McpServerConfig(name="probe", url="https://probe/mcp")
    register_mcp_source("unit-probe", lambda: [marker])
    assert resolve_mcp_source("unit-probe") == [marker]
    # unknown source with no importable module resolves to nothing, not an error
    assert resolve_mcp_source("nonexistent-source-xyz") == []
