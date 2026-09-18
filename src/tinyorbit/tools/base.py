"""Tool base — the self-describing action interface.

Built in Chapter 6. Three ideas from the book live here:

    1. A tool is self-describing: name + description + JSON schema go to the
       model verbatim; the same schema validates what the model sends back.
    2. Fail-closed defaults: a tool that forgets to override is_read_only()
       or is_concurrency_safe() is treated as a serial write.
    3. Safety is input-dependent: the flags take the parsed input, because
       `ls` and `rm -rf` are the same tool with different inputs.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid an import cycle; permissions imports Tool
    from tinyorbit.permissions import PermissionPolicy


@dataclass
class ToolResult:
    """What a tool hands back. `data` becomes the tool_result content block."""

    data: str
    is_error: bool = False


@dataclass
class ToolUseContext:
    """Everything a tool may need at call time — the book's 'god object', kept small.

    `file_state` is the staleness ledger: path -> mtime at the last Read.
    Write and Edit refuse to touch a file the model has not read (or that
    changed since), so the agent never clobbers edits made outside the
    session.
    """

    cwd: Path
    abort: asyncio.Event
    permissions: "PermissionPolicy"
    data_dir: Path
    file_state: dict[Path, float] = field(default_factory=dict)
    # Set only when subagent dispatch is wired (a SubagentContext; Ch 8). None
    # for plain tools and inside a subagent — kept as Any so base.py stays free
    # of the agents import.
    subagent: Any = None
    # PreToolUse guard hooks (list[HookSpec]; Ch 12). Empty by default; run
    # before the permission check. Any keeps base.py free of the hooks import.
    hooks: list = field(default_factory=list)

    def resolve(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        return (path if path.is_absolute() else self.cwd / path).resolve()


class Tool:
    """Base class. Subclasses set name/description/input_schema and implement call()."""

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    # ── fail-closed defaults ─────────────────────────────────────────────
    def is_read_only(self, inp: dict) -> bool:
        return False

    def is_concurrency_safe(self, inp: dict) -> bool:
        return False

    # ── hooks a tool may override ────────────────────────────────────────
    def validate(self, inp: dict) -> str | None:
        """Semantic validation beyond the schema. Return an error string or None."""
        return None

    def describe_call(self, inp: dict) -> str:
        """One-line summary for the UI and permission prompts."""
        return json.dumps(inp, sort_keys=True)[:120]

    async def call(self, inp: dict, ctx: ToolUseContext) -> ToolResult:
        raise NotImplementedError

    # ── what the API sees ────────────────────────────────────────────────
    def to_api(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def check_schema(schema: dict, inp: Any) -> str | None:
    """Minimal JSON-schema check: required keys and top-level types.

    A full JSON-schema validator is a dependency we do not need. This catches
    the errors that actually occur (missing field, wrong type) and lets
    the tool's own validate() handle the rest.
    """
    if not isinstance(inp, dict):
        return "input must be an object"
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in inp:
            return f"missing required field '{key}'"
    for key, value in inp.items():
        want = props.get(key, {}).get("type")
        py_type = _JSON_TYPES.get(want)
        if py_type is None:
            continue
        # bool is a subclass of int in Python; the API's "integer" is not.
        if want in ("integer", "number") and isinstance(value, bool):
            return f"field '{key}' must be a {want}"
        if not isinstance(value, py_type):
            return f"field '{key}' must be a {want}"
    return None
