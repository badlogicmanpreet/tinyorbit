"""A tiny YAML-ish frontmatter parser, shared by agents and skills (Ch 8).

Deliberately minimal — `key: value` lines between `---` fences, nothing nested.
A real YAML parser is a dependency the open-source core does not need, and the
two features that read frontmatter (subagents, skills) only ever use flat keys
and comma-separated lists. Keeping it here means both parse identically.
"""

from __future__ import annotations


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a `---`-fenced frontmatter block from the body.

    Returns ({}, whole-text) when there is no frontmatter, so a plain Markdown
    file still yields a usable body.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return {}, text
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip()
    body = "\n".join(lines[end + 1:]).strip()
    return meta, body
