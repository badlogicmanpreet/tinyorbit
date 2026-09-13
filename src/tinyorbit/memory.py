"""Memory — persistent instructions loaded into every prompt (Ch 11).

We read an instructions file from the user's home, then from every ancestor
of the working directory, root first, so project rules override global ones
by appearing later in the prompt. AGENTS.md is the cross-agent convention;
TINYORBIT.md wins when both exist in the same directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MEMORY_FILENAMES = ("TINYORBIT.md", "AGENTS.md")
MAX_MEMORY_CHARS = 40_000


@dataclass(frozen=True)
class MemoryFile:
    path: Path
    content: str


def _first_present(directory: Path) -> Path | None:
    for name in MEMORY_FILENAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def load_memory(cwd: Path, home: Path | None = None) -> list[MemoryFile]:
    home = home or Path.home()
    candidates: list[Path] = []

    global_file = _first_present(home / ".tinyorbit")
    if global_file:
        candidates.append(global_file)

    ancestors = [cwd, *cwd.parents]
    for directory in reversed(ancestors):           # root → cwd
        if directory == home:                       # home is the global slot, handled above
            continue
        found = _first_present(directory)
        if found and found not in candidates:
            candidates.append(found)

    files = []
    for path in candidates:
        try:
            files.append(MemoryFile(path, path.read_text(errors="replace")[:MAX_MEMORY_CHARS]))
        except OSError:
            continue
    return files


def render_memory(files: list[MemoryFile]) -> str:
    if not files:
        return ""
    parts = [f"## {f.path}\n\n{f.content.strip()}" for f in files]
    return "# Project instructions\n\n" + "\n\n".join(parts)
