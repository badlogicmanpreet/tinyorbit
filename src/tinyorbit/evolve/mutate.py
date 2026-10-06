"""Mutator — an LLM that reads failures and proposes edits to a genome.

The proposer sees four things, each borrowed from a published self-improvement
pattern:
  * the parent's full genome and scores (what to change);
  * the parent's failure reports from evaluation (trace-mined diagnosis:
    fix the failure that recurs, not the one that is loudest);
  * the attempt ledger — recent edits and whether they were kept or rejected,
    so it does not re-propose a fix that already failed (the failure wiki);
  * a couple of strong, different elites from the archive (inspirations).

It must answer with a one-line summary and SEARCH/REPLACE blocks (edits.py).
Brevity is part of the instructions because the loop also penalizes growth:
the regularized objective, not just the prompt, discourages bloat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tinyorbit.evolve.archive import Candidate
from tinyorbit.evolve.edits import Edit, parse_edits
from tinyorbit.evolve.genome import Genome
from tinyorbit.evolve.llm import Complete

MAX_FAILURES_SHOWN = 6
MAX_FAILURE_DETAIL = 1_500
MAX_LEDGER_SHOWN = 20
MAX_INSPIRATION_CHARS = 3_000

MUTATE_SYSTEM = """\
You improve the harness of an autonomous coding agent. The model is fixed; you \
edit the text the harness gives it. The artifacts are:

- system_prompt.md: the agent's core system prompt. Keep the {cwd} and \
{platform} placeholders.
- memory.md: standing instructions, injected like a project AGENTS.md.
- lessons.md: lessons distilled from past runs. You may rewrite, merge or prune them.
- skills/<name>/SKILL.md: on-demand playbooks (frontmatter: name, description).
- tools/<ToolName>.md: the description the model sees for that tool.

Rules:
- Diagnose the failure that recurs across tasks, then make 1 to 3 focused edits that address it.
- Generalize. Never mention a specific repository, issue, file, test or answer from the reports.
- Shorter is better. Prefer rewriting or deleting text over appending; growth is penalized.
- Do not repeat an attempt the ledger shows was rejected.

Answer with exactly this shape:
<summary>one line: what you changed and why</summary>
<<<<<<< SEARCH <path>
<exact text to replace; empty to write the whole file>
=======
<new text>
>>>>>>> REPLACE
(repeat the block for each edit)\
"""


@dataclass
class MutationContext:
    parent: Candidate
    inspirations: list[tuple[Candidate, Genome]] = field(default_factory=list)
    ledger: list[dict] = field(default_factory=list)


@dataclass
class Proposal:
    summary: str
    edits: list[Edit]
    raw: str = ""


def _render_genome(genome: Genome) -> str:
    return "\n\n".join(
        f'<file path="{rel}">\n{genome.files[rel]}\n</file>' for rel in sorted(genome.files)
    ) or "(empty genome: every artifact is at its built-in default)"


def _render_scores(c: Candidate) -> str:
    parts = [f"{k}={v:.3f}" for k, v in c.stage_scores.items()]
    if c.fitness is not None:
        parts.append(f"fitness={c.fitness:.3f}")
    parts.append(f"size={c.size} chars")
    return ", ".join(parts)


def _render_failures(c: Candidate) -> str:
    if not c.failures:
        return "(no failure reports; look for ways to make success more reliable or cheaper)"
    out = []
    for f in c.failures[:MAX_FAILURES_SHOWN]:
        detail = str(f.get("detail", ""))[:MAX_FAILURE_DETAIL]
        out.append(f"### {f.get('task', '?')}: {f.get('summary', '')}\n{detail}".rstrip())
    return "\n\n".join(out)


def _render_ledger(entries: list[dict]) -> str:
    if not entries:
        return "(no attempts yet)"
    lines = []
    for e in entries[-MAX_LEDGER_SHOWN:]:
        delta = e.get("delta")
        shown = f" ({delta:+.3f})" if isinstance(delta, (int, float)) else ""
        why = f" [{e['reason']}]" if e.get("reason") else ""
        lines.append(f"- {e.get('outcome', '?')}{shown}: {e.get('summary', '')}{why}")
    return "\n".join(lines)


def _render_inspirations(parent: Genome, items: list[tuple[Candidate, Genome]]) -> str:
    if not items:
        return "(none yet)"
    out = []
    for cand, genome in items:
        changed = {
            rel: text for rel, text in genome.files.items() if parent.files.get(rel) != text
        }
        body = "\n\n".join(f'<file path="{rel}">\n{text}\n</file>' for rel, text in sorted(changed.items()))
        out.append(f"## {cand.id} ({_render_scores(cand)}): {cand.summary}\n"
                   f"{body[:MAX_INSPIRATION_CHARS] or '(same files as the parent)'}")
    return "\n\n".join(out)


def build_mutation_prompt(genome: Genome, ctx: MutationContext) -> str:
    return f"""\
# Parent {ctx.parent.id}: {_render_scores(ctx.parent)}

{_render_genome(genome)}

# Failure reports from evaluating the parent

{_render_failures(ctx.parent)}

# Attempt ledger (most recent last)

{_render_ledger(ctx.ledger)}

# Other strong variants (changed files only)

{_render_inspirations(genome, ctx.inspirations)}

Propose your edits now."""


_SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.S)


def parse_proposal(raw: str) -> Proposal:
    m = _SUMMARY.search(raw)
    summary = " ".join(m.group(1).split()) if m else ""
    return Proposal(summary=summary[:300], edits=parse_edits(raw), raw=raw)


class LLMMutator:
    def __init__(self, complete: Complete) -> None:
        self._complete = complete

    async def __call__(self, genome: Genome, ctx: MutationContext) -> Proposal:
        raw = await self._complete(MUTATE_SYSTEM, build_mutation_prompt(genome, ctx))
        return parse_proposal(raw)
