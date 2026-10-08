"""Event log — every step of an evolution run, append-only, for audit and reports.

The archive (archive.json) is the *state* of a run; the ledger is one line per
attempt. Neither says how the run got there step by step: which island was
searched, why that parent was picked, what the mutator was asked and answered,
when each benchmark stage started and ended, which grid cell a child took from
whom, why a candidate was or was not promoted. events.jsonl records exactly
that (an event-sourced log), so report.py can replay a run for a reader who
was not watching, and a live run can be followed while it is still going.

Bulky artifacts stay out of the log and are referenced by path:
    iterations/<NNNN>/prompt.md      what the mutator was shown
    iterations/<NNNN>/response.md    what it answered
    iterations/<NNNN>/diff.patch     parent → child, unified diff
"""

from __future__ import annotations

import difflib
import json
import time
from pathlib import Path

from tinyorbit.evolve.genome import Genome

EVENTS = "events.jsonl"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class EventLog:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def path(self) -> Path:
        return self.root / EVENTS

    def emit(self, kind: str, **data) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        event = {"ts": now(), "kind": kind, **data}
        with self.path.open("a") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
        return event

    def read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def artifact(self, iteration: int, name: str, text: str) -> str:
        """Write a bulky artifact for an iteration; returns its path relative to the run dir."""
        rel = Path("iterations") / f"{iteration:04d}" / name
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        return str(rel)


def genome_diff(parent: Genome, child: Genome) -> str:
    """Unified diff over every file in either genome (created and deleted files included)."""
    chunks: list[str] = []
    for path in sorted(set(parent.files) | set(child.files)):
        before, after = parent.files.get(path, ""), child.files.get(path, "")
        if before == after:
            continue
        chunks.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{path}" if path in parent.files else "/dev/null",
            tofile=f"b/{path}" if path in child.files else "/dev/null",
        ))
    return "".join(c if c.endswith("\n") else c + "\n" for c in chunks)
