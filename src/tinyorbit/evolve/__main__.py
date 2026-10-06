"""CLI for the self-evolving layer: seed, run, reflect, show."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from tinyorbit.evolve.archive import Archive
from tinyorbit.evolve.evaluate import Stage, load_evaluator
from tinyorbit.evolve.genome import Genome, seed_genome
from tinyorbit.evolve.loop import EvolveConfig, evolve


def _kv(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--arg expects key=value, got {pair!r}")
        out[key] = value
    return out


def _complete(args: argparse.Namespace):
    from tinyorbit.evolve.llm import provider_complete
    from tinyorbit.providers import resolve_provider
    from tinyorbit.state import CostTracker

    provider = resolve_provider(provider=args.provider, model=args.model)
    cost = CostTracker()
    return provider, cost, provider_complete(provider, args.model, cost=cost, effort=args.effort)


def cmd_seed(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty", file=sys.stderr)
        return 2
    seed_genome().save(out)
    print(f"seed genome written to {out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from tinyorbit.evolve.mutate import LLMMutator

    seed = Genome.load(Path(args.seed)) if args.seed else seed_genome()
    cfg = EvolveConfig(
        iterations=args.iterations,
        stages=[Stage.parse(s) for s in args.stages.split(",") if s.strip()],
        holdout=args.holdout or None,
        holdout_margin=args.holdout_margin,
        size_penalty=args.size_penalty,
        num_islands=args.islands,
        rng_seed=args.rng_seed,
    )
    evaluator = load_evaluator(args.evaluator, _kv(args.arg))

    async def main() -> Archive:
        provider, cost, complete = _complete(args)
        try:
            return await evolve(seed, evaluator, LLMMutator(complete), Path(args.out), cfg)
        finally:
            print(f"mutator: {cost.summary(args.model)}", file=sys.stderr)
            await provider.close()

    archive = asyncio.run(main())
    champ = archive.candidates[archive.champion] if archive.champion else None
    if champ:
        print(f"champion {champ.id}: fitness={champ.fitness} holdout={champ.holdout} → {archive.root / 'champion'}")
    return 0


def cmd_reflect(args: argparse.Namespace) -> int:
    from tinyorbit.evolve.reflect import reflect_into_genome

    data = json.loads(Path(args.transcript).read_text())
    messages = data["messages"] if isinstance(data, dict) else data

    async def main() -> list:
        provider, cost, complete = _complete(args)
        try:
            return await reflect_into_genome(
                Path(args.genome), messages, complete, outcome=args.outcome, feedback=args.feedback,
                source=args.source or (data.get("session_id", "") if isinstance(data, dict) else ""),
            )
        finally:
            await provider.close()

    for lesson in asyncio.run(main()):
        print(f"+ {lesson.title}: {lesson.body}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    archive = Archive.open(Path(args.out))
    print(f"iteration {archive.iteration}, champion {archive.champion}")
    rows = sorted(archive.candidates.values(), key=lambda c: c.fitness if c.fitness is not None else -1, reverse=True)
    for c in rows:
        fit = f"{c.fitness:.3f}" if c.fitness is not None else "  -  "
        hold = f"{c.holdout:.3f}" if c.holdout is not None else "  -  "
        star = "*" if c.id == archive.champion else " "
        print(f"{star}{c.id}  {c.status:8}  fit={fit} hold={hold} size={c.size:6}  ← {c.parent or '-':6} {c.summary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tinyorbit.evolve")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default="claude-opus-5")
        p.add_argument("--provider", default=None)
        p.add_argument("--effort", default=None)

    p = sub.add_parser("seed", help="write the built-in harness text as an editable genome")
    p.add_argument("out")
    p.set_defaults(fn=cmd_seed)

    p = sub.add_parser("run", help="evolve a genome against an evaluator")
    p.add_argument("--out", required=True, help="run directory (resumes if it exists)")
    p.add_argument("--seed", default=None, help="seed genome dir (default: built-in harness text)")
    p.add_argument("--evaluator", required=True, help="module:factory returning an Evaluator")
    p.add_argument("--arg", action="append", default=[], metavar="KEY=VALUE", help="evaluator factory kwarg")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--stages", default="train", help="cascade, e.g. 'smoke:0.2,train'")
    p.add_argument("--holdout", default=None, help="held-out stage name gating promotion")
    p.add_argument("--holdout-margin", type=float, default=0.0)
    p.add_argument("--size-penalty", type=float, default=0.01)
    p.add_argument("--islands", type=int, default=2)
    p.add_argument("--rng-seed", type=int, default=0)
    model_flags(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("reflect", help="distill lessons from a saved transcript into a genome")
    p.add_argument("--transcript", required=True, help="TINYORBIT_TRANSCRIPT json or a messages list")
    p.add_argument("--genome", required=True)
    p.add_argument("--outcome", default="", help="e.g. 'resolved' or 'failed: tests still red'")
    p.add_argument("--feedback", default="", help="ground-truth detail, e.g. failing test output")
    p.add_argument("--source", default="")
    model_flags(p)
    p.set_defaults(fn=cmd_reflect)

    p = sub.add_parser("show", help="summarize a run directory")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_show)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
