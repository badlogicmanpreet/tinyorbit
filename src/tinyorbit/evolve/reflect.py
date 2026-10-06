"""Inner loop — distill reusable lessons from one finished session.

The outer loop (loop.py) improves the harness across many evaluated runs; this
is the cheap per-run step. After a session ends, one model call reads a
condensed transcript plus whatever outcome signal exists and proposes a few
*general* lessons. They land in the genome's `lessons.md`, which the next run
loads as memory: online self-improvement with no weights touched.

Two guards keep the file from rotting, both borrowed from the skill-evolution
literature:
  * a failure-aware prompt — a verified outcome (tests passed/failed) is worth
    far more than the agent's own "done", so the caller passes it when known;
  * a bounded, deduplicated book — same-titled lessons replace each other and
    the oldest fall off past MAX_LESSONS, so lessons.md never grows without
    limit (the outer loop can still prune or rewrite it like any artifact).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tinyorbit.evolve.genome import LESSONS
from tinyorbit.evolve.llm import Complete

MAX_LESSONS = 20
MAX_NEW_PER_RUN = 3
TRANSCRIPT_BUDGET = 24_000

LESSONS_HEADER = (
    "# Lessons learned\n\n"
    "Distilled from past runs of this agent. Apply a lesson when its situation matches.\n"
)
_LESSON_LINE = re.compile(r"^- \*\*(?P<title>.+?)\*\*: (?P<body>.*?)(?: <!-- (?P<meta>.*?) -->)?\s*$")


@dataclass(frozen=True)
class Lesson:
    title: str
    body: str
    outcome: str = ""
    source: str = ""

    @property
    def key(self) -> str:
        return re.sub(r"[^a-z0-9]+", " ", self.title.lower()).strip()


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


# ── the lessons book (lessons.md) ────────────────────────────────────────────

def parse_lessons(text: str) -> list[Lesson]:
    out = []
    for line in text.splitlines():
        m = _LESSON_LINE.match(line)
        if not m:
            continue
        meta = dict(kv.split("=", 1) for kv in (m["meta"] or "").split() if "=" in kv)
        out.append(Lesson(m["title"], m["body"], meta.get("outcome", ""), meta.get("source", "")))
    return out


def render_lessons(lessons: Iterable[Lesson]) -> str:
    lines = []
    for lesson in lessons:
        meta = " ".join(f"{k}={v}" for k, v in (("outcome", lesson.outcome), ("source", lesson.source)) if v)
        suffix = f" <!-- {meta} -->" if meta else ""
        lines.append(f"- **{_one_line(lesson.title)}**: {_one_line(lesson.body)}{suffix}")
    return LESSONS_HEADER + "\n" + "\n".join(lines) + "\n"


def merge_lessons(existing: list[Lesson], new: list[Lesson], limit: int = MAX_LESSONS) -> list[Lesson]:
    """Newest last. A same-titled lesson is replaced and moved to the end."""
    merged = list(existing)
    for lesson in new:
        merged = [m for m in merged if m.key != lesson.key]
        merged.append(lesson)
    return merged[-limit:]


def record_lessons(genome_dir: Path, new: list[Lesson], limit: int = MAX_LESSONS) -> list[Lesson]:
    path = Path(genome_dir) / LESSONS
    existing = parse_lessons(path.read_text()) if path.is_file() else []
    merged = merge_lessons(existing, new, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_lessons(merged))
    return merged


# ── transcript condensing ────────────────────────────────────────────────────

def _get(block: Any, key: str, default: Any = None) -> Any:
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(_get(b, "text", "")) for b in content)
    return str(content or "")


def condense_transcript(messages: list[Any], budget: int = TRANSCRIPT_BUDGET) -> str:
    """One short line per step; head and tail kept when over budget.

    The head holds the task and the agent's first plan, the tail holds how it
    ended — the two places a failure is usually diagnosable.
    """
    lines: list[str] = []
    for i, msg in enumerate(messages):
        role, content = _get(msg, "role", "?"), _get(msg, "content", "")
        if isinstance(content, str):
            cap = 2_000 if i == 0 else 800
            lines.append(f"{role.upper()}: {_one_line(content)[:cap]}")
            continue
        for block in content or []:
            kind = _get(block, "type", "")
            if kind == "text":
                lines.append(f"{role.upper()}: {_one_line(_get(block, 'text', ''))[:600]}")
            elif kind == "tool_use":
                args = json.dumps(_get(block, "input", {}), default=str)
                lines.append(f"TOOL {_get(block, 'name', '?')} {args[:300]}")
            elif kind == "tool_result":
                tag = "RESULT(error)" if _get(block, "is_error") else "RESULT"
                lines.append(f"{tag}: {_one_line(_result_text(_get(block, 'content')))[:400]}")

    text = "\n".join(lines)
    if len(text) <= budget:
        return text
    head_budget, tail_budget = budget // 3, budget - budget // 3
    head, tail = [], []
    used = 0
    for line in lines:
        if used + len(line) > head_budget:
            break
        head.append(line)
        used += len(line) + 1
    used = 0
    for line in reversed(lines[len(head):]):
        if used + len(line) > tail_budget:
            break
        tail.append(line)
        used += len(line) + 1
    omitted = len(lines) - len(head) - len(tail)
    return "\n".join([*head, f"... [{omitted} steps omitted] ...", *reversed(tail)])


# ── the reflection call ──────────────────────────────────────────────────────

REFLECT_SYSTEM = """\
You review one session of a coding agent and distill lessons that will make \
future sessions of the same agent succeed more often. You are editing the \
agent's own long-term memory, so be selective: a lesson must change behavior \
on many future tasks, not restate this task. Never include file names, issue \
numbers, function names or other facts specific to this repository.\
"""


def build_reflect_prompt(transcript: str, outcome: str, feedback: str, known: list[Lesson]) -> str:
    known_text = "\n".join(f"- {k.title}: {k.body}" for k in known) or "(none yet)"
    return f"""\
Outcome: {outcome or "unknown"}
External feedback (ground truth when present): {feedback.strip() or "(none)"}

Lessons the agent already has (do not repeat or rephrase these):
{known_text}

Session transcript (condensed):
{transcript}

What, if anything, should the agent do differently next time? Focus on the \
decision that most affected the outcome: how it explored, verified, or decided \
it was done.

Reply with only a JSON array of at most {MAX_NEW_PER_RUN} objects, each \
{{"title": "<short imperative>", "lesson": "<one or two sentences: when it \
applies and what to do>"}}. Reply [] if nothing new is worth keeping."""


def parse_reflection(text: str) -> list[dict]:
    """Tolerant JSON-array extraction: models wrap JSON in prose or fences."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


async def reflect(messages: list[Any], complete: Complete, *, outcome: str = "", feedback: str = "",
                  known: list[Lesson] | None = None, source: str = "") -> list[Lesson]:
    known = known or []
    prompt = build_reflect_prompt(condense_transcript(messages), outcome, feedback, known)
    raw = await complete(REFLECT_SYSTEM, prompt)
    known_keys = {k.key for k in known}
    out: list[Lesson] = []
    for item in parse_reflection(raw)[:MAX_NEW_PER_RUN]:
        title, body = _one_line(item.get("title", "")), _one_line(item.get("lesson", ""))
        if not title or not body:
            continue
        lesson = Lesson(title[:120], body[:600], _one_line(outcome).replace(" ", "_")[:40], source)
        if lesson.key not in known_keys:
            out.append(lesson)
    return out


async def reflect_into_genome(genome_dir: Path, messages: list[Any], complete: Complete, *,
                              outcome: str = "", feedback: str = "", source: str = "") -> list[Lesson]:
    """reflect() + record_lessons(): the one call a post-run hook makes."""
    path = Path(genome_dir) / LESSONS
    known = parse_lessons(path.read_text()) if path.is_file() else []
    new = await reflect(messages, complete, outcome=outcome, feedback=feedback, known=known, source=source)
    if new:
        record_lessons(genome_dir, new)
    return new
