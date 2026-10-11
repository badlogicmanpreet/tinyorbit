"""System prompt construction (Ch 4).

One rule: everything before the cache boundary must be byte-identical from
turn to turn. Identity, tool guidance, cwd and memory are stable for a
session, so they go in the cached block. The date goes after the boundary.
"""

from __future__ import annotations

import platform
from datetime import date
from pathlib import Path

from tinyorbit.memory import MemoryFile, render_memory

STATIC_TEMPLATE = """\
You are tinyorbit, a coding agent that runs in a terminal. You help with software \
engineering tasks in the working directory by reading, searching, and editing files \
and running shell commands with the tools provided.

Working directory: {cwd}
Platform: {platform}

How to work:
- Look at code with Read, Glob and Grep before changing it. Never guess file contents.
- Read a file before you Edit or Write it; both refuse otherwise.
- Prefer Edit for targeted changes. Use Write only for new files or full rewrites.
- Use Bash for builds, tests and git. Do not run destructive commands unless asked.
- Independent tool calls can go in one turn; they run in parallel when safe.
- Keep replies brief. When you finish, say what changed and what you did not do.
"""


def render_template(template: str, cwd: Path) -> str:
    """Fill the two placeholders by plain replacement, not str.format.

    A template may come from an evolved genome (see tinyorbit.evolve), and
    model-written text can carry stray braces that str.format would reject.
    """
    host = f"{platform.system()} {platform.release()}"
    return template.replace("{cwd}", str(cwd)).replace("{platform}", host)


def build_system_prompt(cwd: Path, memory: list[MemoryFile], template: str | None = None) -> list[dict]:
    static = render_template(template if template is not None else STATIC_TEMPLATE, cwd)
    rendered = render_memory(memory)
    if rendered:
        static += "\n" + rendered + "\n"

    return [
        # ── cached prefix ──
        {"type": "text", "text": static, "cache_control": {"type": "ephemeral"}},
        # ── dynamic boundary: anything below may change between turns ──
        {"type": "text", "text": f"Today's date: {date.today().isoformat()}"},
    ]
