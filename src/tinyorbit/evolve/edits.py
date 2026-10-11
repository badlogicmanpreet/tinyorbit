"""SEARCH/REPLACE edits — how the mutator changes a genome.

Asking a model for targeted diffs instead of whole rewrites keeps children
close to their parent (small, attributable steps, the AlphaEvolve-style
mutation operator) and keeps the model's output short. One block per change:

    <<<<<<< SEARCH system_prompt.md
    - Keep replies brief.
    =======
    - Keep replies brief. Before finishing, run the project's existing tests.
    >>>>>>> REPLACE

An empty SEARCH writes the whole file, which is how a new skill or lesson
file is created. Matching is exact first, then whitespace-tolerant line by
line, because models routinely re-indent or trim the text they quote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tinyorbit.evolve.genome import Genome, is_allowed_path

_BLOCK = re.compile(
    r"^<{7} SEARCH[ \t]+(?P<path>\S+)[ \t]*\n(?P<search>.*?)^={7}[ \t]*\n(?P<replace>.*?)^>{7} REPLACE[ \t]*$",
    re.S | re.M,
)


class EditError(ValueError):
    """An edit that cannot be applied; the child is discarded, the reason logged."""


@dataclass(frozen=True)
class Edit:
    path: str
    search: str
    replace: str


def parse_edits(text: str) -> list[Edit]:
    return [Edit(m["path"], m["search"], m["replace"]) for m in _BLOCK.finditer(text)]


def _fuzzy_replace(current: str, search: str, replace: str) -> str | None:
    """Match `search` against `current` line by line, ignoring surrounding whitespace."""
    want = [ln.strip() for ln in search.strip("\n").splitlines()]
    have = current.splitlines(keepends=True)
    hits = [
        i for i in range(len(have) - len(want) + 1)
        if [ln.strip() for ln in have[i:i + len(want)]] == want
    ]
    if len(hits) != 1:
        return None
    i = hits[0]
    tail = "" if replace.endswith("\n") or i + len(want) >= len(have) else "\n"
    return "".join(have[:i]) + replace + tail + "".join(have[i + len(want):])


def apply_edits(genome: Genome, edits: list[Edit]) -> Genome:
    if not edits:
        raise EditError("proposal contained no SEARCH/REPLACE blocks")
    files = dict(genome.files)
    for edit in edits:
        if not is_allowed_path(edit.path):
            raise EditError(f"path not allowed in a genome: {edit.path}")
        if not edit.search.strip():
            files[edit.path] = edit.replace
            continue
        current = files.get(edit.path)
        if current is None:
            raise EditError(f"{edit.path} does not exist; use an empty SEARCH to create it")
        count = current.count(edit.search)
        if count == 1:
            files[edit.path] = current.replace(edit.search, edit.replace, 1)
            continue
        if count > 1:
            raise EditError(f"SEARCH text matches {count} places in {edit.path}; quote more context")
        fuzzy = _fuzzy_replace(current, edit.search, edit.replace)
        if fuzzy is None:
            raise EditError(f"SEARCH text not found in {edit.path}")
        files[edit.path] = fuzzy
    return Genome(files)
