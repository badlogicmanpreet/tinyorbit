"""Skills — packaged instructions the agent can pull in on demand (Ch 8).

A skill is a named bundle of instructions (a `SKILL.md`) for a repeatable task:
a deploy runbook, a review checklist, a house style. Only the *name and
one-line description* sit in the system prompt; the full body enters the
conversation only when the model invokes the Skill tool. That is the whole
point — dozens of skills stay available without their text bloating every turn.

Discovery mirrors subagents and MCP: empty by default (a plain repo pays
nothing and the Skill tool never appears), populated from Markdown under
`.tinyorbit/skills/`. Both layouts are accepted:

    .tinyorbit/skills/<name>/SKILL.md     (Claude Code's convention)
    .tinyorbit/skills/<name>.md           (flat, for a one-file skill)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tinyorbit.frontmatter import parse_frontmatter


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str


def parse_skill_file(path: Path, fallback_name: str) -> Skill | None:
    """One Markdown file -> one Skill. None if it carries no instructions."""
    meta, body = parse_frontmatter(path.read_text())
    instructions = body.strip()
    if not instructions:
        return None
    return Skill(
        name=meta.get("name") or fallback_name,
        description=meta.get("description", ""),
        instructions=instructions,
    )


def discover_skills(cwd: Path) -> list[Skill]:
    """Load skills from `<cwd>/.tinyorbit/skills/`, or none if the dir is absent."""
    root = Path(cwd) / ".tinyorbit" / "skills"
    if not root.is_dir():
        return []
    found: dict[str, Skill] = {}
    # `<name>/SKILL.md` first, then flat `<name>.md`; a flat file never shadows
    # a directory skill of the same name.
    for path in sorted(root.glob("*/SKILL.md")):
        skill = _safe_parse(path, path.parent.name)
        if skill:
            found.setdefault(skill.name, skill)
    for path in sorted(root.glob("*.md")):
        skill = _safe_parse(path, path.stem)
        if skill:
            found.setdefault(skill.name, skill)
    return list(found.values())


def _safe_parse(path: Path, fallback_name: str) -> Skill | None:
    try:
        return parse_skill_file(path, fallback_name)
    except OSError:
        return None
