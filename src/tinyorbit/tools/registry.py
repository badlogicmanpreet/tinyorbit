"""The tool registry — one source of truth for what the agent can do (Ch 6)."""

from __future__ import annotations

from tinyorbit.tools.base import Tool
from tinyorbit.tools.bash_tool import BashTool
from tinyorbit.tools.file_tools import EditTool, ReadTool, WriteTool
from tinyorbit.tools.search_tools import GlobTool, GrepTool


def get_all_base_tools() -> list[Tool]:
    """Every built-in tool, in a stable order (stable order matters for prompt caching)."""
    return [ReadTool(), WriteTool(), EditTool(), BashTool(), GlobTool(), GrepTool()]


def assemble_tool_pool(disabled: set[str] | None = None) -> list[Tool]:
    """Merge built-ins with anything registered later (MCP tools arrive in Ch 15)."""
    disabled = disabled or set()
    return [tool for tool in get_all_base_tools() if tool.name not in disabled]


def find_tool(tools: list[Tool], name: str) -> Tool | None:
    return next((tool for tool in tools if tool.name == name), None)
