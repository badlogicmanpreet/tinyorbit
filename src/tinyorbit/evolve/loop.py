"""Outer loop — evolve the harness genome against an evaluator.

    seed ──► evaluate (all stages + held-out) ──► champion₀
    repeat:
        island  = next in round-robin
        parent  = archive.sample_parent(island)
        child   = apply(mutator(parent, failures, ledger, inspirations))
        guards  : edits apply, size cap, placeholders kept, not seen before
        cascade : stage by stage, stop at the first missed threshold
        fitness = last-stage score − size_penalty × growth over the seed
        archive : record; take a MAP-Elites cell if best in it
        promote : if it beats the champion on train fitness, run held-out;
                  it becomes champion only if held-out improves by > margin
        migrate every N iterations; checkpoint after every iteration

Three choices here exist to stop a self-improving loop from fooling itself:
the held-out gate (accept only what generalizes), the size penalty (the
regularized objective that keeps prompts from bloating to fit the train set),
and the ledger (rejected ideas are remembered and shown to the next proposer).
Everything is checkpointed, so a benchmark-scale run can be killed and resumed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from tinyorbit.evolve.archive import INVALID, POOLED, REJECTED, Archive, Candidate
from tinyorbit.evolve.edits import EditError, apply_edits
from tinyorbit.evolve.evaluate import EvalResult, Evaluator, Stage, run_cascade
from tinyorbit.evolve.genome import SYSTEM_PROMPT, Genome
from tinyorbit.evolve.mutate import MutationContext, Proposal

Mutator = Callable[[Genome, MutationContext], Awaitable[Proposal]]
Log = Callable[[str], None]

PLACEHOLDERS = ("{cwd}", "{platform}")
MAX_FAILURES_KEPT = 8


@dataclass
class EvolveConfig:
    iterations: int = 10
    stages: list[Stage] = field(default_factory=lambda: [Stage("train")])
    holdout: str | None = None          # stage name scored only for would-be champions
    holdout_margin: float = 0.0         # held-out must beat the champion by more than this
    size_penalty: float = 0.01          # fitness points per 1,000 chars grown over the seed
    max_genome_chars: int = 60_000
    num_islands: int = 2
    migration_interval: int = 5
    feature_bins: int = 4
    rng_seed: int = 0


def fitness(score: float, size: int, seed_size: int, penalty: float) -> float:
    growth = max(0, size - seed_size) / 1_000
    return score - penalty * growth


def _failures(results: dict[str, EvalResult]) -> list[dict]:
    out: list[dict] = []
    for result in results.values():
        out.extend(f.to_dict() for f in result.failures)
    return out[:MAX_FAILURES_KEPT]


def _metrics(results: dict[str, EvalResult]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for result in results.values():
        merged.update(result.metrics)
    return merged


def _guard(child: Genome, parent: Genome, cfg: EvolveConfig, archive: Archive) -> str | None:
    """Cheap, model-free checks before paying for an evaluation."""
    if child.fingerprint() == parent.fingerprint():
        return "no-op: edits left the genome unchanged"
    if child.size() > cfg.max_genome_chars:
        return f"genome too large ({child.size()} > {cfg.max_genome_chars} chars)"
    before, after = parent.files.get(SYSTEM_PROMPT, ""), child.files.get(SYSTEM_PROMPT, "")
    for ph in PLACEHOLDERS:
        if ph in before and ph not in after:
            return f"system prompt lost the {ph} placeholder"
    if (twin := archive.seen(child.fingerprint())) is not None:
        return f"duplicate of {twin.id}"
    return None


async def evolve(seed: Genome, evaluator: Evaluator, mutator: Mutator, out_dir: Path,
                 cfg: EvolveConfig | None = None, log: Log = print) -> Archive:
    cfg = cfg or EvolveConfig()
    archive = Archive.open(out_dir, num_islands=cfg.num_islands, feature_bins=cfg.feature_bins,
                           rng=random.Random(cfg.rng_seed))
    # A resumed run must not replay the same random choices from iteration 0.
    archive.rng.seed(f"{cfg.rng_seed}:{archive.iteration}")

    if not archive.candidates:
        await _evaluate_seed(seed, evaluator, archive, cfg, log)

    seed_size = int(archive.reference.get("size", seed.size()))

    while archive.iteration < cfg.iterations:
        it = archive.iteration
        island = it % archive.num_islands
        parent = archive.sample_parent(island)
        parent_genome = archive.genome(parent.id)
        ctx = MutationContext(
            parent=parent,
            inspirations=[(c, archive.genome(c.id)) for c in archive.inspirations(parent)],
            ledger=archive.ledger(last=20),
        )
        entry: dict = {"iteration": it, "parent": parent.id, "island": island}

        try:
            proposal = await mutator(parent_genome, ctx)
            entry["summary"] = proposal.summary
            child = apply_edits(parent_genome, proposal.edits)
            reason = _guard(child, parent_genome, cfg, archive)
        except EditError as exc:
            child, reason = None, str(exc)
        except Exception as exc:  # noqa: BLE001 — a failed proposal is one lost iteration
            child, reason = None, f"mutator error: {exc}"
            entry.setdefault("summary", "")

        if child is None or reason:
            entry.update(outcome=INVALID, reason=reason)
            archive.log(entry)
            log(f"[{it}] invalid from {parent.id}: {reason}")
            archive.iteration = it + 1
            archive.save()
            continue

        results, passed = await run_cascade(evaluator, child, cfg.stages)
        last = list(results)[-1]
        score = results[last].score
        cand = Candidate(
            id=archive.next_id(), parent=parent.id, generation=parent.generation + 1,
            island=island, fingerprint=child.fingerprint(), size=child.size(), iteration=it,
            summary=entry["summary"], status=POOLED if passed else REJECTED,
            stage_scores={k: r.score for k, r in results.items()}, metrics=_metrics(results),
            failures=_failures(results), cost_usd=sum(r.cost_usd for r in results.values()),
        )
        if passed:
            cand.fitness = fitness(score, cand.size, seed_size, cfg.size_penalty)
        archive.add(cand, child)

        outcome = cand.status
        if passed and await _maybe_promote(cand, child, evaluator, archive, cfg):
            outcome = "champion"
        parent_fit = parent.fitness or 0.0
        entry.update(
            outcome=outcome, candidate=cand.id, stage=last, score=score,
            fitness=cand.fitness, holdout=cand.holdout,
            delta=(cand.fitness - parent_fit) if cand.fitness is not None else None,
            reason="" if passed else f"below {last} threshold",
        )
        archive.log(entry)
        log(f"[{it}] {cand.id} ← {parent.id}: {outcome} {last}={score:.3f}"
            + (f" fitness={cand.fitness:.3f}" if cand.fitness is not None else "")
            + (f" holdout={cand.holdout:.3f}" if cand.holdout is not None else "")
            + f"  {cand.summary}")

        archive.iteration = it + 1
        if archive.iteration % max(1, cfg.migration_interval) == 0:
            archive.migrate()
        archive.save()

    return archive


async def _evaluate_seed(seed: Genome, evaluator: Evaluator, archive: Archive,
                         cfg: EvolveConfig, log: Log) -> None:
    # The baseline runs every stage regardless of thresholds: it is the
    # yardstick, and the loop needs a full train score to compare against.
    results, _ = await run_cascade(evaluator, seed, cfg.stages, stop_early=False)
    last = list(results)[-1]
    archive.reference = {"size": float(seed.size()), "cost_usd": _metrics(results).get("cost_usd", 0.0)}
    cand = Candidate(
        id=archive.next_id(), parent=None, generation=0, island=0,
        fingerprint=seed.fingerprint(), size=seed.size(), summary="seed",
        stage_scores={k: r.score for k, r in results.items()}, metrics=_metrics(results),
        failures=_failures(results), cost_usd=sum(r.cost_usd for r in results.values()),
        fitness=results[last].score,
    )
    if cfg.holdout:
        cand.holdout = (await evaluator.evaluate(seed, cfg.holdout)).score
    archive.add(cand, seed)
    # Every island starts from the seed, so none begins empty.
    for island in range(1, archive.num_islands):
        archive.migrants.setdefault(island, []).append(cand.id)
        archive._place(island, cand)
    archive.set_champion(cand.id)
    archive.log({"iteration": -1, "candidate": cand.id, "outcome": "seed", "summary": "seed",
                 "score": results[last].score, "fitness": cand.fitness, "holdout": cand.holdout})
    archive.save()
    log(f"[seed] {cand.id}: {last}={results[last].score:.3f}"
        + (f" holdout={cand.holdout:.3f}" if cand.holdout is not None else ""))


async def _maybe_promote(cand: Candidate, genome: Genome, evaluator: Evaluator,
                         archive: Archive, cfg: EvolveConfig) -> bool:
    champ = archive.candidates.get(archive.champion or "")
    if champ is not None and (cand.fitness or 0.0) <= (champ.fitness or 0.0):
        return False
    if cfg.holdout:
        cand.holdout = (await evaluator.evaluate(genome, cfg.holdout)).score
        if champ is not None and cand.holdout <= (champ.holdout or 0.0) + cfg.holdout_margin:
            return False
    archive.set_champion(cand.id)
    return True
