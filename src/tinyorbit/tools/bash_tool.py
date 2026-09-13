"""Bash — the most complex tool, reduced to its essentials.

Two book patterns show up here:

    * Input-dependent safety (Ch 6): the same tool is read-only for
      `git status` and destructive for `rm -rf`. We classify the command
      before deciding whether it needs permission or may run in parallel.
    * Cooperative abort (Ch 5): the subprocess races against the abort
      event so Ctrl+C kills the child instead of orphaning it.
"""

from __future__ import annotations

import asyncio
import re
import shlex

from tinyorbit.tools.base import Tool, ToolResult, ToolUseContext

DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000

READ_ONLY_COMMANDS = {
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "pwd", "echo",
    "which", "file", "stat", "du", "tree", "env", "printenv", "uname", "date",
    "whoami", "diff", "sort", "uniq", "cut", "awk", "sed", "true", "type",
}
GIT_READ_ONLY = {"status", "log", "diff", "show", "branch", "blame", "remote", "rev-parse", "ls-files", "describe"}
_SPLIT = re.compile(r"\|\||&&|\||;")


def classify_read_only(command: str) -> bool:
    """Heuristic: True only when every segment of the pipeline is a known reader.

    Any redirect, `sudo`, `rm`, or unknown command makes the whole thing a write.
    Fail-closed: parse failures return False.
    """
    if ">" in command or "sudo" in command:
        return False
    for segment in _SPLIT.split(command):
        try:
            words = shlex.split(segment)
        except ValueError:
            return False
        if not words:
            continue
        head = words[0]
        if head == "git":
            if len(words) < 2 or words[1] not in GIT_READ_ONLY:
                return False
        elif head not in READ_ONLY_COMMANDS:
            return False
    return True


class BashTool(Tool):
    name = "Bash"
    description = (
        "Run a shell command in the working directory and return its output. "
        "Use for builds, tests, git, and anything the other tools do not cover. "
        "Avoid interactive commands."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "timeout_ms": {"type": "integer", "description": f"Timeout in ms (default {DEFAULT_TIMEOUT_MS}, max {MAX_TIMEOUT_MS})"},
            "description": {"type": "string", "description": "Short description of what the command does"},
        },
        "required": ["command"],
    }

    def is_read_only(self, inp: dict) -> bool:
        return classify_read_only(inp.get("command", ""))

    def is_concurrency_safe(self, inp: dict) -> bool:
        return self.is_read_only(inp)

    def validate(self, inp: dict) -> str | None:
        if not inp["command"].strip():
            return "command is empty"
        return None

    def describe_call(self, inp: dict) -> str:
        return inp.get("command", "")

    async def call(self, inp: dict, ctx: ToolUseContext) -> ToolResult:
        timeout_s = min(int(inp.get("timeout_ms", DEFAULT_TIMEOUT_MS)), MAX_TIMEOUT_MS) / 1000
        proc = await asyncio.create_subprocess_shell(
            inp["command"],
            cwd=ctx.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        output_task = asyncio.ensure_future(proc.communicate())
        abort_task = asyncio.ensure_future(ctx.abort.wait())
        try:
            done, _ = await asyncio.wait(
                {output_task, abort_task}, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
            )
            if output_task not in done:
                proc.kill()
                await proc.wait()
                reason = "interrupted by user" if ctx.abort.is_set() else f"timed out after {timeout_s:.0f}s"
                return ToolResult(f"Command {reason}", is_error=True)
            stdout, stderr = output_task.result()
        finally:
            for task in (output_task, abort_task):
                if not task.done():
                    task.cancel()

        text = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        if err.strip():
            text = f"{text}\n[stderr]\n{err}" if text.strip() else err
        if proc.returncode != 0:
            return ToolResult(f"{text.rstrip()}\n[exit code {proc.returncode}]", is_error=True)
        return ToolResult(text.rstrip() or "(no output)")
