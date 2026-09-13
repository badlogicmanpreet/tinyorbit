"""Glob and Grep — read-only, concurrency-safe search tools.

Both cap their output (Ch 6, result budgeting): the model gets the first N
hits and a note about how many more exist, never an unbounded dump.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from tinyorbit.tools.base import Tool, ToolResult, ToolUseContext

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".astro", "dist", "build", ".tinyorbit"}
MAX_HITS = 200


def _skipped(rel: Path) -> bool:
    return any(part in SKIP_DIRS for part in rel.parts)


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(1024)
    except OSError:
        return True


class GlobTool(Tool):
    name = "Glob"
    description = "Find files by glob pattern (e.g. '**/*.py', 'src/**/*.ts'). Newest first."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern relative to path"},
            "path": {"type": "string", "description": "Directory to search (default: working directory)"},
        },
        "required": ["pattern"],
    }

    def is_read_only(self, inp: dict) -> bool:
        return True

    def is_concurrency_safe(self, inp: dict) -> bool:
        return True

    def describe_call(self, inp: dict) -> str:
        return inp.get("pattern", "")

    async def call(self, inp: dict, ctx: ToolUseContext) -> ToolResult:
        base = ctx.resolve(inp.get("path") or ".")
        if not base.is_dir():
            return ToolResult(f"Not a directory: {base}", is_error=True)

        def search() -> list[Path]:
            hits = [
                p for p in base.glob(inp["pattern"])
                if p.is_file() and not _skipped(p.relative_to(base))
            ]
            hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return hits

        hits = await asyncio.to_thread(search)
        if not hits:
            return ToolResult("No files matched.")
        shown = hits[:MAX_HITS]
        out = "\n".join(str(p.relative_to(base)) for p in shown)
        if len(hits) > MAX_HITS:
            out += f"\n... {len(hits) - MAX_HITS} more files not shown. Narrow the pattern."
        return ToolResult(out)


class GrepTool(Tool):
    name = "Grep"
    description = "Search file contents with a regular expression. Returns path:line:text matches."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regular expression"},
            "path": {"type": "string", "description": "Directory or file to search (default: working directory)"},
            "glob": {"type": "string", "description": "Only search files matching this glob (e.g. '*.py')"},
            "max_results": {"type": "integer", "description": f"Cap on matches (default {MAX_HITS})"},
        },
        "required": ["pattern"],
    }

    def is_read_only(self, inp: dict) -> bool:
        return True

    def is_concurrency_safe(self, inp: dict) -> bool:
        return True

    def validate(self, inp: dict) -> str | None:
        try:
            re.compile(inp["pattern"])
        except re.error as exc:
            return f"invalid regular expression: {exc}"
        return None

    def describe_call(self, inp: dict) -> str:
        return inp.get("pattern", "")

    async def call(self, inp: dict, ctx: ToolUseContext) -> ToolResult:
        base = ctx.resolve(inp.get("path") or ".")
        if not base.exists():
            return ToolResult(f"Path not found: {base}", is_error=True)
        regex = re.compile(inp["pattern"])
        file_glob = inp.get("glob") or "*"
        cap = max(int(inp.get("max_results", MAX_HITS)), 1)

        def search() -> tuple[list[str], int]:
            files = [base] if base.is_file() else (
                p for p in base.rglob(file_glob) if p.is_file() and not _skipped(p.relative_to(base))
            )
            hits: list[str] = []
            total = 0
            for path in files:
                if _is_binary(path):
                    continue
                try:
                    lines = path.read_text(errors="replace").splitlines()
                except OSError:
                    continue
                rel = path.relative_to(base) if path != base else path.name
                for n, line in enumerate(lines, start=1):
                    if regex.search(line):
                        total += 1
                        if len(hits) < cap:
                            hits.append(f"{rel}:{n}:{line.strip()[:200]}")
            return hits, total

        hits, total = await asyncio.to_thread(search)
        if not hits:
            return ToolResult("No matches.")
        out = "\n".join(hits)
        if total > len(hits):
            out += f"\n... {total - len(hits)} more matches not shown. Narrow the pattern or path."
        return ToolResult(out)
