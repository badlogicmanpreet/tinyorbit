"""MCP client layer — remote tools over streamable-HTTP (SDK core).

The agent gains tools it did not ship with by connecting to Model Context
Protocol servers. This package speaks the wire directly (no `mcp` SDK), adapts
each discovered tool to tinyorbit's `Tool` contract, and stays transport-neutral
so an opt-in gateway can inject its own mTLS/auth. Everything here is
provider-neutral and carries no vendor import.
"""

from tinyorbit.mcp.client import McpClient, McpError
from tinyorbit.mcp.config import McpServerConfig
from tinyorbit.mcp.discovery import (
    discover_mcp_servers,
    load_mcp_json,
    parse_mcp_json,
    register_mcp_source,
    resolve_mcp_source,
)
from tinyorbit.mcp.session import McpSession, connect_mcp_servers, filter_tools
from tinyorbit.mcp.tools import McpTool, build_mcp_tools, mcp_tool_name

__all__ = [
    "McpClient",
    "McpError",
    "McpServerConfig",
    "McpSession",
    "McpTool",
    "build_mcp_tools",
    "connect_mcp_servers",
    "discover_mcp_servers",
    "filter_tools",
    "load_mcp_json",
    "mcp_tool_name",
    "parse_mcp_json",
    "register_mcp_source",
    "resolve_mcp_source",
]
