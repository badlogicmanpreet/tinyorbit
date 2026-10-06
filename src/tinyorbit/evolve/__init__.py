"""The self-evolving layer: the harness improves its own text from feedback.

Two loops over one artifact set (a genome, see genome.py):

    inner  reflect.py   after a run, distill general lessons into lessons.md
    outer  loop.py      evolve the whole genome against an evaluator, keeping
                        an archive of variants and promoting only what holds up
                        on held-out tasks

Opt-in end to end and stdlib-only: a run without --genome reads nothing from
here, and the core never imports this package unless asked.

    python -m tinyorbit.evolve seed  runs/g0
    python -m tinyorbit.evolve run   --seed runs/g0 --out runs/evo1 --evaluator pkg.mod:factory
    python -m tinyorbit.evolve reflect --transcript t.json --genome runs/evo1/champion --outcome failed
    python main.py --genome runs/evo1/champion "fix the failing test"
"""

from tinyorbit.evolve.archive import Archive, Candidate
from tinyorbit.evolve.evaluate import EvalResult, Evaluator, Failure, Stage
from tinyorbit.evolve.genome import Genome, load_overlay, seed_genome
from tinyorbit.evolve.loop import EvolveConfig, evolve
from tinyorbit.evolve.mutate import LLMMutator, MutationContext, Proposal
from tinyorbit.evolve.reflect import Lesson, reflect, reflect_into_genome

__all__ = [
    "Archive", "Candidate", "EvalResult", "Evaluator", "EvolveConfig", "Failure", "Genome",
    "LLMMutator", "Lesson", "MutationContext", "Proposal", "Stage", "evolve", "load_overlay",
    "reflect", "reflect_into_genome", "seed_genome",
]
