"""Genome — the evolvable, text-only surface of the harness (self-evolving layer 1).

The self-evolving layer starts at the narrowest surface that can fix a failure:
editable text around a frozen model. A genome is a small directory of Markdown
artifacts that the harness reads *on top of* its built-in defaults. Nothing in
it is code, so every variant is inspectable, diffable and safe to version.

    system_prompt.md        replaces prompt.STATIC_TEMPLATE ({cwd}, {platform} placeholders)
    memory.md               harness-level instructions, loaded like a global AGENTS.md
    lessons.md              distilled lessons from past runs (written by reflect.py)
    skills/<name>/SKILL.md  extra skills, same layouts as .tinyorbit/skills/
    skills/<name>.md
    tools/<ToolName>.md     replaces that tool's description

The same directory is what the outer loop mutates (loop.py) and what the inner
loop writes lessons into (reflect.py), so both loops evolve one artifact set.
A run picks a genome up with `--genome DIR` or `TINYORBIT_GENOME=DIR`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from tinyorbit.memory import MAX_MEMORY_CHARS, MemoryFile
from tinyorbit.skills import Skill, load_skills_dir

SYSTEM_PROMPT = "system_prompt.md"
MEMORY = "memory.md"
LESSONS = "lessons.md"

# A whitelist, not a blacklist: an evolved edit can only touch paths the harness
# knows how to read, and can never escape the genome directory.
ALLOWED_PATH = re.compile(
    r"^(system_prompt\.md|memory\.md|lessons\.md"
    r"|skills/[A-Za-z0-9_-]+/SKILL\.md|skills/[A-Za-z0-9_-]+\.md"
    r"|tools/[A-Za-z0-9_]+\.md)$"
)


def is_allowed_path(rel: str) -> bool:
    return bool(ALLOWED_PATH.match(rel))


@dataclass
class Genome:
    """Pure content: relative path -> text. Lineage and scores live in the archive."""

    files: dict[str, str] = field(default_factory=dict)

    def size(self) -> int:
        return sum(len(t) for t in self.files.values())

    def fingerprint(self) -> str:
        """Content hash, so the loop never pays to re-evaluate an identical child."""
        h = hashlib.sha256()
        for rel in sorted(self.files):
            h.update(rel.encode() + b"\0" + self.files[rel].encode() + b"\0")
        return h.hexdigest()[:16]

    def with_files(self, updates: dict[str, str | None]) -> "Genome":
        """A new genome with files replaced; a None value deletes the file."""
        files = dict(self.files)
        for rel, text in updates.items():
            if text is None:
                files.pop(rel, None)
            else:
                files[rel] = text
        return Genome(files)

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for rel, text in self.files.items():
            if not is_allowed_path(rel):
                raise ValueError(f"path not allowed in a genome: {rel}")
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    @classmethod
    def load(cls, root: Path) -> "Genome":
        files: dict[str, str] = {}
        if root.is_dir():
            for path in sorted(root.rglob("*.md")):
                rel = path.relative_to(root).as_posix()
                if is_allowed_path(rel):
                    files[rel] = path.read_text(errors="replace")
        return cls(files)


def seed_genome() -> Genome:
    """Generation 0: today's built-in prompt and tool descriptions, made editable."""
    from tinyorbit.prompt import STATIC_TEMPLATE
    from tinyorbit.tools import get_all_base_tools

    files = {SYSTEM_PROMPT: STATIC_TEMPLATE}
    for tool in get_all_base_tools():
        files[f"tools/{tool.name}.md"] = tool.description
    return Genome(files)


# ── the read side: how a run consumes a genome ───────────────────────────────

@dataclass(frozen=True)
class Overlay:
    """What bootstrap.setup() layers over the defaults. Empty fields change nothing."""

    system_template: str | None = None
    memory: list[MemoryFile] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    tool_descriptions: dict[str, str] = field(default_factory=dict)


def load_overlay(root: Path) -> Overlay:
    root = Path(root)
    genome = Genome.load(root)
    files = genome.files
    template = files.get(SYSTEM_PROMPT)
    # memory first, lessons second: lessons are the more specific, more recent
    # layer, and later memory sections take precedence in the prompt.
    memory = [
        MemoryFile(root / name, files[name][:MAX_MEMORY_CHARS])
        for name in (MEMORY, LESSONS)
        if files.get(name, "").strip()
    ]
    tools = {
        Path(rel).stem: text.strip()
        for rel, text in files.items()
        if rel.startswith("tools/") and text.strip()
    }
    return Overlay(
        system_template=template if template and template.strip() else None,
        memory=memory,
        skills=load_skills_dir(root / "skills"),
        tool_descriptions=tools,
    )


def apply_tool_descriptions(tools: list, descriptions: dict[str, str]) -> None:
    """Shadow each named tool's class-level description on its instance.

    Unknown names are ignored: a genome evolved against one tool pool must still
    load when a tool is missing (e.g. hidden by --disallowed-tools).
    """
    for tool in tools:
        if tool.name in descriptions:
            tool.description = descriptions[tool.name]
