"""The tool registry — one source of truth for what the agent can do (Ch 6)."""

from __future__ import annotations

from fnmatch import fnmatch

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


def select_tools(tools: list[Tool], allow: list[str] | None = None,
                 deny: list[str] | None = None) -> list[Tool]:
    """Filter a tool pool by name with fnmatch globs; deny wins over allow.

    An empty/absent allow list allows everything (the common case). This is what
    backs `--allowed-tools`/`--disallowed-tools`, and it works on any Tool —
    built-ins, Task/Skill, or MCP tools by their namespaced names (`mcp__srv__*`).
    """
    def any_match(name: str, patterns: list[str]) -> bool:
        return any(fnmatch(name, p) for p in patterns)

    kept = []
    for tool in tools:
        if deny and any_match(tool.name, deny):
            continue
        if allow and not any_match(tool.name, allow):
            continue
        kept.append(tool)
    return kept
