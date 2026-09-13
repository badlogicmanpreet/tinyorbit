"""Read / Write / Edit — the file tools.

The interesting part is staleness detection (Ch 6, FileEditTool). Every
Read records the file's mtime in ctx.file_state. Write and Edit refuse to
run unless the file was read this session *and* has not changed since.
"""

from __future__ import annotations

from pathlib import Path

from tinyorbit.tools.base import Tool, ToolResult, ToolUseContext

DEFAULT_READ_LIMIT = 2000


def _staleness_error(path: Path, ctx: ToolUseContext) -> str | None:
    if not path.exists():
        return None
    last_read = ctx.file_state.get(path)
    if last_read is None:
        return f"{path} exists but has not been read this session. Read it before modifying it."
    if path.stat().st_mtime > last_read:
        return f"{path} changed on disk since it was last read. Read it again before modifying it."
    return None


class ReadTool(Tool):
    name = "Read"
    description = (
        "Read a text file. Returns numbered lines (like cat -n). "
        "Use offset/limit to page through large files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path, or relative to the working directory"},
            "offset": {"type": "integer", "description": "1-based line number to start from"},
            "limit": {"type": "integer", "description": f"Maximum lines to return (default {DEFAULT_READ_LIMIT})"},
        },
        "required": ["file_path"],
    }

    def is_read_only(self, inp: dict) -> bool:
        return True

    def is_concurrency_safe(self, inp: dict) -> bool:
        return True

    def describe_call(self, inp: dict) -> str:
        return inp.get("file_path", "")

    async def call(self, inp: dict, ctx: ToolUseContext) -> ToolResult:
        path = ctx.resolve(inp["file_path"])
        if not path.exists():
            return ToolResult(f"File not found: {path}", is_error=True)
        if path.is_dir():
            return ToolResult(f"{path} is a directory. Use Glob or Bash `ls` instead.", is_error=True)
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            return ToolResult(f"Could not read {path}: {exc}", is_error=True)

        ctx.file_state[path] = path.stat().st_mtime

        lines = text.splitlines()
        offset = max(int(inp.get("offset", 1)), 1)
        limit = max(int(inp.get("limit", DEFAULT_READ_LIMIT)), 1)
        chunk = lines[offset - 1 : offset - 1 + limit]
        if not lines:
            return ToolResult("(empty file)")
        if not chunk:
            return ToolResult(f"offset {offset} is past the end of the file ({len(lines)} lines)", is_error=True)

        out = "\n".join(f"{n:6}\t{line}" for n, line in enumerate(chunk, start=offset))
        remaining = len(lines) - (offset - 1 + len(chunk))
        if remaining > 0:
            out += f"\n... {remaining} more lines. Use offset={offset + limit} to continue."
        return ToolResult(out)


class WriteTool(Tool):
    name = "Write"
    description = (
        "Create a file or overwrite it completely. Existing files must be Read first. "
        "Prefer Edit for changes to existing files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path, or relative to the working directory"},
            "content": {"type": "string", "description": "Full file contents"},
        },
        "required": ["file_path", "content"],
    }

    def describe_call(self, inp: dict) -> str:
        return inp.get("file_path", "")

    async def call(self, inp: dict, ctx: ToolUseContext) -> ToolResult:
        path = ctx.resolve(inp["file_path"])
        if (err := _staleness_error(path, ctx)) is not None:
            return ToolResult(err, is_error=True)
        if path.is_dir():
            return ToolResult(f"{path} is a directory", is_error=True)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(inp["content"])
        except OSError as exc:
            return ToolResult(f"Could not write {path}: {exc}", is_error=True)
        ctx.file_state[path] = path.stat().st_mtime
        return ToolResult(f"Wrote {len(inp['content'])} characters to {path}")


class EditTool(Tool):
    name = "Edit"
    description = (
        "Replace an exact string in a file. old_string must match exactly once "
        "(include surrounding lines to disambiguate) unless replace_all is true. "
        "The file must be Read first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path, or relative to the working directory"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def validate(self, inp: dict) -> str | None:
        if inp["old_string"] == inp["new_string"]:
            return "old_string and new_string are identical; nothing to do"
        if inp["old_string"] == "":
            return "old_string must not be empty (use Write to create files)"
        return None

    def describe_call(self, inp: dict) -> str:
        return inp.get("file_path", "")

    async def call(self, inp: dict, ctx: ToolUseContext) -> ToolResult:
        path = ctx.resolve(inp["file_path"])
        if not path.is_file():
            return ToolResult(f"File not found: {path}", is_error=True)
        if (err := _staleness_error(path, ctx)) is not None:
            return ToolResult(err, is_error=True)

        text = path.read_text(errors="replace")
        old, new = inp["old_string"], inp["new_string"]
        count = text.count(old)
        if count == 0:
            return ToolResult("old_string not found in file. Re-read the file and match the text exactly.", is_error=True)
        replace_all = bool(inp.get("replace_all", False))
        if count > 1 and not replace_all:
            return ToolResult(
                f"old_string matches {count} places. Include more surrounding context "
                "to make it unique, or set replace_all=true.",
                is_error=True,
            )
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        path.write_text(updated)
        ctx.file_state[path] = path.stat().st_mtime
        replaced = count if replace_all else 1
        return ToolResult(f"Edited {path}: replaced {replaced} occurrence(s)")
