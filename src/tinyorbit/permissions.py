"""Permissions — who decides whether a tool call runs (Ch 6).

Three modes and a short resolution chain. The *order* of the chain is the
lesson:

    bypass  →  explicit allow rules  →  read-only tools  →  mode defaults  →  ask the human

Each layer answers only the cases it is sure about and falls through
otherwise. The interactive prompt is the last resort, not the first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Literal

from tinyorbit.tools.base import Tool, ToolUseContext


class PermissionMode(str, Enum):
    DEFAULT = "default"                  # ask for anything that is not read-only
    ACCEPT_EDITS = "acceptEdits"         # auto-allow file writes inside cwd; still ask for Bash
    BYPASS = "bypassPermissions"         # allow everything (headless / trusted sandboxes)


@dataclass(frozen=True)
class Decision:
    behavior: Literal["allow", "deny"]
    reason: str = ""


AskFn = Callable[[str], Awaitable[bool]]

_RULE = re.compile(r"^\s*(\w+)\s*(?:\((.*)\))?\s*$")


def rule_matches(rule: str, tool: Tool, inp: dict) -> bool:
    """Rules look like `Read`, `Bash(git status*)`, `Edit(src/*)`.

    A bare name allows every call of that tool. A parenthesised spec matches
    the tool's describe_call() output: exact, or prefix if it ends in `*`.
    """
    m = _RULE.match(rule)
    if not m or m.group(1) != tool.name:
        return False
    spec = m.group(2)
    if spec is None:
        return True
    target = tool.describe_call(inp)
    if spec.endswith("*"):
        return target.startswith(spec[:-1])
    return target == spec


@dataclass
class PermissionPolicy:
    mode: PermissionMode = PermissionMode.DEFAULT
    ask: AskFn | None = None
    allow_rules: list[str] = field(default_factory=list)
    session_allows: set[str] = field(default_factory=set)   # "always allow" answers this session

    async def can_use_tool(self, tool: Tool, inp: dict, ctx: ToolUseContext) -> Decision:
        if self.mode is PermissionMode.BYPASS:
            return Decision("allow", "bypassPermissions")

        for rule in [*self.allow_rules, *self.session_allows]:
            if rule_matches(rule, tool, inp):
                return Decision("allow", f"rule {rule}")

        if tool.is_read_only(inp):
            return Decision("allow", "read-only")

        if self.mode is PermissionMode.ACCEPT_EDITS and tool.name in ("Write", "Edit"):
            target = ctx.resolve(inp.get("file_path", ""))
            if target.is_relative_to(ctx.cwd):
                return Decision("allow", "acceptEdits")

        if self.ask is None:
            return Decision(
                "deny",
                "non-interactive session. Re-run with --permission-mode bypassPermissions "
                f"or --allow '{tool.name}(...)'.",
            )
        approved = await self.ask(f"{tool.name}: {tool.describe_call(inp)}")
        return Decision("allow", "user approved") if approved else Decision("deny", "user declined")
