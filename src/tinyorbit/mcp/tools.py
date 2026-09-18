"""Adapt a discovered MCP tool to tinyorbit's `Tool` contract.

An MCP tool is just another self-describing action: its `inputSchema` goes to
the model verbatim and validates the reply, exactly like a built-in. The only
differences are the namespaced name (`mcp__<server>__<tool>`, so two servers can
both expose `search` without collision) and that `call()` proxies over the wire
instead of running locally.
"""

from __future__ import annotations

import json

from tinyorbit.mcp.client import McpClient, McpError
from tinyorbit.tools.base import Tool, ToolResult, ToolUseContext

_EMPTY_SCHEMA = {"type": "object", "properties": {}}


def mcp_tool_name(server: str, tool: str) -> str:
    return f"mcp__{server}__{tool}"


def flatten_content(result: dict) -> tuple[str, bool]:
    """Turn an MCP tool result into (text, is_error) for a tool_result block.

    MCP returns content as a list of typed blocks; we join text blocks and
    render anything else as compact JSON so nothing is silently dropped.
    """
    is_error = bool(result.get("isError"))
    parts: list[str] = []
    for block in result.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    # Some servers return structured output only; surface it rather than "".
    if not parts and "structuredContent" in result:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(parts), is_error


class McpTool(Tool):
    """A remote tool exposed by an MCP server, driven through an `McpClient`."""

    def __init__(self, server: str, spec: dict, client: McpClient) -> None:
        self._server = server
        self._remote_name = spec["name"]
        self._client = client
        self.name = mcp_tool_name(server, self._remote_name)
        self.description = spec.get("description", "") or ""
        self.input_schema = spec.get("inputSchema") or _EMPTY_SCHEMA
        # MCP annotations are optional hints; absence means fail-closed (a write).
        self._annotations = spec.get("annotations") or {}

    def is_read_only(self, inp: dict) -> bool:
        return bool(self._annotations.get("readOnlyHint", False))

    def is_concurrency_safe(self, inp: dict) -> bool:
        # A read-only tool has no side effects to serialize.
        return bool(self._annotations.get("readOnlyHint", False))

    def describe_call(self, inp: dict) -> str:
        return f"{self._server}:{self._remote_name} {json.dumps(inp, sort_keys=True)}"[:120]

    async def call(self, inp: dict, ctx: ToolUseContext) -> ToolResult:
        try:
            result = await self._client.call_tool(self._remote_name, inp)
        except McpError as exc:
            return ToolResult(f"MCP call failed ({self.name}): {exc}", is_error=True)
        text, is_error = flatten_content(result)
        return ToolResult(text or "(no output)", is_error=is_error)


def build_mcp_tools(server: str, specs: list[dict], client: McpClient) -> list[McpTool]:
    return [McpTool(server, spec, client) for spec in specs if spec.get("name")]
