"""Evaluator contract and the staged cascade.

The loop never knows what a benchmark is. An Evaluator scores one genome on one
named stage and returns a score in [0, 1], optional metrics, and the failures
it saw — the failures are what the mutator reads next, so they are the most
valuable part of the result.

Stages form a cascade (cheap filter first): a child that cannot clear a 3-task
smoke set never pays for the 30-task train set. The held-out stage is separate
and run only for a would-be champion, so promotion is judged on tasks the
mutator never saw failures from — the validation gate that keeps the loop from
overfitting its own training tasks.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tinyorbit.evolve.genome import Genome


@dataclass
class Failure:
    task: str
    summary: str
    detail: str = ""          # condensed transcript, test output — whatever explains it

    def to_dict(self) -> dict:
        return {"task": self.task, "summary": self.summary, "detail": self.detail}


@dataclass
class EvalResult:
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    cost_usd: float = 0.0


@dataclass(frozen=True)
class Stage:
    name: str
    min_score: float = 0.0    # must reach this to advance to the next stage

    @classmethod
    def parse(cls, spec: str) -> "Stage":
        """'smoke:0.2' -> Stage('smoke', 0.2); 'train' -> Stage('train')."""
        name, _, threshold = spec.partition(":")
        return cls(name.strip(), float(threshold) if threshold else 0.0)


class Evaluator(Protocol):
    async def evaluate(self, genome: Genome, stage: str) -> EvalResult: ...


async def run_cascade(evaluator: Evaluator, genome: Genome, stages: list[Stage],
                      *, stop_early: bool = True) -> tuple[dict[str, EvalResult], bool]:
    """Evaluate stage by stage. Returns (results so far, cleared every gate)."""
    results: dict[str, EvalResult] = {}
    passed = True
    for stage in stages:
        result = await evaluator.evaluate(genome, stage.name)
        results[stage.name] = result
        if result.score < stage.min_score:
            passed = False
            if stop_early:
                break
    return results, passed


def load_evaluator(spec: str, kwargs: dict[str, Any] | None = None) -> Evaluator:
    """'package.module:factory' or 'path/to/file.py:factory' -> factory(**kwargs).

    Keeps benchmarks out of core: an evaluator can live in a file that is not
    part of any installed package (bench/evolve_eval.py), loaded by path.
    """
    module_name, _, attr = spec.rpartition(":")
    if not module_name or not attr:
        raise ValueError(f"evaluator must look like 'module:factory' or 'file.py:factory', got {spec!r}")
    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        if not path.is_file():
            raise ValueError(f"evaluator file not found: {path}")
        mod_spec = importlib.util.spec_from_file_location(f"_tinyorbit_evaluator_{path.stem}", path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, attr)(**(kwargs or {}))
