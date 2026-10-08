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
Everything is checkpointed, so a benchmark-scale run can be killed and resumed,
and every step is written to events.jsonl (events.py) so report.py can replay it.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from tinyorbit.evolve.archive import INVALID, POOLED, REJECTED, Archive, Candidate
from tinyorbit.evolve.edits import EditError, apply_edits
from tinyorbit.evolve.evaluate import EvalResult, Evaluator, Stage, run_cascade
from tinyorbit.evolve.events import EventLog, genome_diff
from tinyorbit.evolve.genome import SYSTEM_PROMPT, Genome
from tinyorbit.evolve.mutate import MutationContext, Proposal

Mutator = Callable[[Genome, MutationContext], Awaitable[Proposal]]
Log = Callable[[str], None]
OnIteration = Callable[[Archive], None]

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
    budget_usd: float | None = None     # stop before an attempt that could overshoot this


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


class _Recorder:
    """Evaluator wrapper that logs each stage as it starts and ends (live progress)."""

    def __init__(self, inner: Evaluator, events: EventLog, thresholds: dict[str, float]) -> None:
        self.inner, self.events, self.thresholds = inner, events, thresholds
        self.candidate, self.iteration = "", -1

    async def evaluate(self, genome: Genome, stage: str) -> EvalResult:
        self.events.emit("stage_start", iteration=self.iteration, candidate=self.candidate, stage=stage,
                         fingerprint=genome.fingerprint())
        started = time.monotonic()
        result = await self.inner.evaluate(genome, stage)
        min_score = self.thresholds.get(stage)
        self.events.emit(
            "stage_end", iteration=self.iteration, candidate=self.candidate, stage=stage,
            score=result.score, min_score=min_score,
            passed=None if min_score is None else result.score >= min_score,
            seconds=round(time.monotonic() - started, 1), cost_usd=result.cost_usd,
            metrics=result.metrics, info=result.info,
            failures=[{"task": f.task, "summary": f.summary} for f in result.failures],
        )
        return result


async def evolve(seed: Genome, evaluator: Evaluator, mutator: Mutator, out_dir: Path,
                 cfg: EvolveConfig | None = None, log: Log = print,
                 on_iteration: OnIteration | None = None) -> Archive:
    cfg = cfg or EvolveConfig()
    archive = Archive.open(out_dir, num_islands=cfg.num_islands, feature_bins=cfg.feature_bins,
                           rng=random.Random(cfg.rng_seed))
    # A resumed run must not replay the same random choices from iteration 0.
    archive.rng.seed(f"{cfg.rng_seed}:{archive.iteration}")
    events = EventLog(archive.root)
    thresholds = {s.name: s.min_score for s in cfg.stages}
    recorder = _Recorder(evaluator, events, thresholds)
    cfg_dict = asdict(cfg)
    events.emit("run_start", resumed=bool(archive.candidates), iteration=archive.iteration, config=cfg_dict)

    def checkpoint() -> None:
        archive.save()
        if on_iteration is not None:
            try:
                on_iteration(archive)
            except Exception as exc:  # noqa: BLE001 — a report failure must not stop the search
                log(f"on_iteration hook failed: {exc}")

    if not archive.candidates:
        await _evaluate_seed(seed, recorder, archive, cfg, log, events)
        checkpoint()

    seed_size = int(archive.reference.get("size", seed.size()))

    while archive.iteration < cfg.iterations:
        if cfg.budget_usd is not None:
            spent, reserve = spend(events)
            if spent + reserve > cfg.budget_usd:
                events.emit("budget_stop", iteration=archive.iteration, spent_usd=spent,
                            reserve_usd=reserve, budget_usd=cfg.budget_usd)
                log(f"budget: spent ${spent:.2f}, one more attempt could cost ${reserve:.2f}, "
                    f"cap ${cfg.budget_usd:.2f}; stopping")
                break
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
        events.emit("iteration_start", iteration=it, island=island, parent=parent.id, pick=archive.last_pick,
                    parent_fitness=parent.fitness, inspirations=[c.id for c, _ in ctx.inspirations])

        proposal = None
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
        _log_mutation(events, it, mutator, proposal, parent_genome, child)

        if child is None or reason:
            entry.update(outcome=INVALID, reason=reason)
            archive.log(entry)
            events.emit("invalid", iteration=it, parent=parent.id, reason=reason)
            log(f"[{it}] invalid from {parent.id}: {reason}")
            archive.iteration = it + 1
            events.emit("iteration_end", iteration=it, outcome=INVALID)
            checkpoint()
            continue

        recorder.candidate, recorder.iteration = archive.next_id(), it
        results, passed = await run_cascade(recorder, child, cfg.stages)
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
        champion_before = archive.champion
        archive.add(cand, child)
        events.emit("candidate", iteration=it, candidate=cand.id, parent=parent.id, island=island,
                    status=cand.status, stage_scores=cand.stage_scores, fitness=cand.fitness,
                    size=cand.size, size_delta=cand.size - parent.size, growth_over_seed=cand.size - seed_size,
                    penalty=(score - cand.fitness) if cand.fitness is not None else None,
                    summary=cand.summary, cost_usd=cand.cost_usd, fingerprint=cand.fingerprint,
                    place=archive.last_place)

        outcome = cand.status
        promoted, why = (False, "did not clear every stage") if not passed else \
            await _maybe_promote(cand, child, recorder, archive, cfg)
        if passed:
            events.emit("promotion", iteration=it, candidate=cand.id, promoted=promoted, reason=why,
                        champion_before=champion_before, champion_after=archive.champion,
                        fitness=cand.fitness, holdout=cand.holdout)
        if promoted:
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
            events.emit("migration", iteration=it, moves=archive.migrate())
        events.emit("iteration_end", iteration=it, outcome=outcome, candidate=cand.id)
        checkpoint()

    events.emit("run_end", iteration=archive.iteration, champion=archive.champion)
    return archive


def spend(events: EventLog) -> tuple[float, float]:
    """(USD spent so far, worst cost of one attempt seen so far).

    Read back from the event log, so a resumed run counts what earlier sessions
    spent. Reused (cached) stage results cost nothing now. The seed ran every
    stage, so its bill is the first worst-case estimate (a pre-flight budget
    check rather than a hard stop mid-stage, which would waste the stage).
    """
    per_step: dict[int, float] = {}
    for ev in events.read():
        if ev["kind"] == "stage_end" and not (ev.get("info") or {}).get("cached"):
            usd = float(ev.get("cost_usd") or 0.0)
        elif ev["kind"] == "mutation":
            usd = float(ev.get("cost_usd") or 0.0)
        else:
            continue
        it = ev.get("iteration", -1)
        per_step[it] = per_step.get(it, 0.0) + usd
    return sum(per_step.values()), max(per_step.values(), default=0.0)


def _log_mutation(events: EventLog, it: int, mutator, proposal: Proposal | None,
                  parent: Genome, child: Genome | None) -> None:
    prompt = getattr(mutator, "last_prompt", "")
    paths = {
        "prompt": events.artifact(it, "prompt.md", prompt) if prompt else None,
        "response": events.artifact(it, "response.md", proposal.raw) if proposal and proposal.raw else None,
        "diff": events.artifact(it, "diff.patch", genome_diff(parent, child)) if child is not None else None,
    }
    events.emit(
        "mutation", iteration=it, summary=proposal.summary if proposal else "",
        edits=[{"path": e.path, "create": not e.search, "removed_chars": len(e.search),
                "added_chars": len(e.replace)} for e in (proposal.edits if proposal else [])],
        cost_usd=getattr(mutator, "last_cost_usd", None), artifacts=paths,
    )


async def _evaluate_seed(seed: Genome, evaluator: _Recorder, archive: Archive,
                         cfg: EvolveConfig, log: Log, events: EventLog) -> None:
    # The baseline runs every stage regardless of thresholds: it is the
    # yardstick, and the loop needs a full train score to compare against.
    evaluator.candidate, evaluator.iteration = archive.next_id(), -1
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
    events.emit("seed", candidate=cand.id, stage_scores=cand.stage_scores, fitness=cand.fitness,
                holdout=cand.holdout, size=cand.size, cost_usd=cand.cost_usd, reference=archive.reference,
                islands=archive.num_islands, fingerprint=cand.fingerprint)
    log(f"[seed] {cand.id}: {last}={results[last].score:.3f}"
        + (f" holdout={cand.holdout:.3f}" if cand.holdout is not None else ""))


async def _maybe_promote(cand: Candidate, genome: Genome, evaluator: Evaluator,
                         archive: Archive, cfg: EvolveConfig) -> tuple[bool, str]:
    """Returns (promoted, why), the why in words a report can show as is."""
    champ = archive.candidates.get(archive.champion or "")
    if champ is not None and (cand.fitness or 0.0) <= (champ.fitness or 0.0):
        return False, f"fitness {cand.fitness:.3f} does not beat champion {champ.id} ({champ.fitness:.3f})"
    if cfg.holdout:
        cand.holdout = (await evaluator.evaluate(genome, cfg.holdout)).score
        if champ is not None and cand.holdout <= (champ.holdout or 0.0) + cfg.holdout_margin:
            return False, (f"held-out {cand.holdout:.3f} does not beat champion {champ.id} "
                           f"({champ.holdout or 0.0:.3f}) by more than {cfg.holdout_margin}")
    archive.set_champion(cand.id)
    return True, "beat the champion on train fitness" + (" and on held-out" if cfg.holdout else "")
