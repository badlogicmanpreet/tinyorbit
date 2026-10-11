"""SWE-bench Verified as an evolution evaluator.

    uv run python -m tinyorbit.evolve run --out bench/logs/evo-1 --seed bench/logs/g0 \
        --evaluator bench/evolve_eval.py:make --arg jobs=2 \
        --stages smoke:0.67,train --holdout holdout --iterations 10

Each (genome, stage) becomes one run.py invocation in the bench venv:
`run.py --tasks <stage file> --genome <dir> --run-id evo-<genome fp>-<stage>-<settings> --score`.
The run id is content-addressed, so the evolve loop resuming after a crash, or
two lineages arriving at the same genome, reuse the finished run instead of
paying for it again (memoised evaluation). run.py itself skips instances it has
already predicted, so even a half-finished stage resumes where it stopped.

Stage files are frozen so scores stay comparable across generations:
    smoke    tasks-smoke3.json     3 easy tasks the default harness solves: a cheap "did the edit break it" gate
    train    tasks-hard10.json     the hard tail; its failures are what the mutator reads
    holdout  tasks-holdout10.json  same distribution, disjoint; only its score is ever used

Score = resolved / total. With 10 tasks one flip moves the score by 0.1, so set
--holdout-margin with that in mind, and re-run a champion before believing it.
An instance that was never scored (docker failure, swebench error) raises
instead of counting as unresolved: infrastructure noise is not a verdict on the
genome, and the loop is resumable, so rerunning retries just that instance.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import sys
from pathlib import Path

from tinyorbit.evolve.evaluate import EvalResult, Failure
from tinyorbit.evolve.genome import Genome
from tinyorbit.evolve.reflect import condense_transcript

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scoring  # noqa: E402 — sibling stdlib module shared with run.py

STAGE_TASKS = {"smoke": "tasks-smoke3.json", "train": "tasks-hard10.json", "holdout": "tasks-holdout10.json"}
DETAIL_BUDGET = 6_000         # chars of condensed transcript per failure handed to the mutator
MAX_TESTS_SHOWN = 8


class HarnessError(RuntimeError):
    """The benchmark plumbing failed; the genome was not judged."""


def _num(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rows(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "results.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {row["instance_id"]: row for row in csv.DictReader(fh)}     # last row per instance wins


def _summary(iid: str, verdict: dict, row: dict) -> str:
    status = verdict["status"]
    reason = row.get("exit_reason", "?")
    if status == "empty_patch":
        return f"no patch produced (agent ended with '{reason}')"
    f2p, p2p = verdict.get("fail_to_pass_failed") or [], verdict.get("pass_to_pass_failed") or []
    if verdict.get("patch_applied") is False:
        return f"patch did not apply (agent ended with '{reason}')"
    parts = [f"agent ended with '{reason}' but the hidden tests disagree"]
    if f2p:
        parts.append(f"{len(f2p)} target test(s) still fail: {', '.join(f2p[:MAX_TESTS_SHOWN])}")
    if p2p:
        parts.append(f"{len(p2p)} previously passing test(s) broke: {', '.join(p2p[:MAX_TESTS_SHOWN])}")
    return "; ".join(parts)


def result_from_run(run_dir: Path, instance_ids: list[str]) -> EvalResult:
    """Turn one scored run directory into the EvalResult the loop consumes."""
    scores = json.loads((run_dir / scoring.SCORES).read_text())
    verdicts = {iid: scores["instances"].get(iid, {"status": "missing"}) for iid in instance_ids}
    missing = sorted(i for i, v in verdicts.items() if v["status"] in ("missing", "error"))
    if missing:
        raise HarnessError(f"{run_dir.name}: not scored {missing}; rerun to retry them")

    rows = _rows(run_dir)
    failures = []
    for iid, verdict in verdicts.items():
        if verdict["status"] == "resolved":
            continue
        transcript = run_dir / "transcripts" / f"{iid}.messages.json"
        detail = ""
        if transcript.exists():
            data = json.loads(transcript.read_text())
            detail = condense_transcript(data["messages"] if isinstance(data, dict) else data, DETAIL_BUDGET)
        failures.append(Failure(iid, _summary(iid, verdict, rows.get(iid, {})), detail))

    total = len(instance_ids)
    resolved = total - len(failures)
    cost = sum(_num(rows.get(i, {}).get("cost_usd")) for i in instance_ids)
    minutes = sum(_num(rows.get(i, {}).get("minutes")) for i in instance_ids)
    return EvalResult(
        score=resolved / total if total else 0.0,
        # per task, not per stage: the archive bins on cost_usd, and stages differ in size
        metrics={"cost_usd": cost / total if total else 0.0, "minutes": minutes / total if total else 0.0,
                 "resolved": float(resolved), "total": float(total)},
        failures=failures,
        cost_usd=cost,
        # provenance for the event log: where to look, and the per-task verdicts behind the score
        info={"run_id": run_dir.name, "run_dir": str(run_dir),
              "tasks": {iid: v["status"] for iid, v in verdicts.items()}},
    )


class BenchEvaluator:
    def __init__(self, *, python: str | None = None, jobs: int = 1, model: str = "claude-opus-5",
                 effort: str | None = None, max_turns: int = 60, timeout_min: int = 45,
                 stages: dict[str, str] | None = None):
        self.python = python or str(HERE / ".venv" / "bin" / "python")
        self.jobs, self.model, self.effort = jobs, model, effort
        self.max_turns, self.timeout_min = max_turns, timeout_min
        self.stages = stages or STAGE_TASKS
        self.logs = HERE / "logs"                  # where run.py writes; not configurable there either
        # settings that change what a run means are part of the cache key
        self.settings = hashlib.sha256(f"{model}|{effort}|{max_turns}|{timeout_min}".encode()).hexdigest()[:6]

    def run_id(self, genome: Genome, stage: str) -> str:
        return f"evo-{genome.fingerprint()}-{stage}-{self.settings}"

    def command(self, tasks_file: Path, run_id: str, genome_dir: Path) -> list[str]:
        cmd = [self.python, str(HERE / "run.py"), "--tasks", str(tasks_file), "--run-id", run_id,
               "--genome", str(genome_dir), "--score", "--jobs", str(self.jobs), "--model", self.model,
               "--max-turns", str(self.max_turns), "--timeout-min", str(self.timeout_min)]
        return cmd + (["--effort", self.effort] if self.effort else [])

    async def evaluate(self, genome: Genome, stage: str) -> EvalResult:
        if stage not in self.stages:
            raise ValueError(f"unknown stage {stage!r}; known: {sorted(self.stages)}")
        tasks_file = HERE / self.stages[stage]
        ids = [t["instance_id"] for t in json.loads(tasks_file.read_text())]
        run_id = self.run_id(genome, stage)
        run_dir = self.logs / run_id
        try:
            result = result_from_run(run_dir, ids)            # memoised: this genome already ran here
            result.info["cached"] = True                      # cost_usd was spent by the earlier run
            return result
        except (FileNotFoundError, HarnessError):
            pass
        genome_dir = self.logs / "evo-genomes" / genome.fingerprint()
        if not genome_dir.exists():
            genome.save(genome_dir)
        proc = await asyncio.create_subprocess_exec(*self.command(tasks_file, run_id, genome_dir), cwd=HERE)
        await proc.wait()
        result = result_from_run(run_dir, ids)
        result.info["cached"] = False
        return result


def make(**kwargs: str) -> BenchEvaluator:
    """Factory for `--evaluator bench/evolve_eval.py:make --arg key=value` (values arrive as strings)."""
    ints = {"jobs", "max_turns", "timeout_min"}
    return BenchEvaluator(**{k: int(v) if k in ints else v for k, v in kwargs.items()})
