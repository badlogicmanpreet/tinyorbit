"""Archive — every variant ever tried, plus the structures that pick the next parent.

Greedy hill-climbing on a noisy benchmark gets stuck on the first lucky
variant. The archive keeps an open-ended population instead (the Darwin Gödel
Machine argument) and organizes it the MAP-Elites way: each candidate is
binned by behavior features — here genome size and run cost, both relative to
the seed — and each cell keeps only its best occupant. A lean variant and a
verbose one therefore survive side by side even when one scores a bit lower,
which is what lets a later mutation build on the lean one.

Islands are independent sub-populations searched in turn; every few
iterations each island's best migrates to the next, so a good idea spreads
without one lineage taking over the whole archive.

On disk (all under the run directory):
    archive.json          candidates, grid, champion, iteration counter
    ledger.jsonl          one line per attempt: what was tried and how it went
    genomes/<id>/         each candidate's files
    champion/             a copy of the current champion, ready for --genome
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tinyorbit.evolve.genome import Genome

POOLED, REJECTED, INVALID = "pooled", "rejected", "invalid"


@dataclass
class Candidate:
    id: str
    parent: str | None
    generation: int
    island: int
    fingerprint: str
    size: int
    iteration: int = 0
    summary: str = ""
    status: str = POOLED
    stage_scores: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    fitness: float | None = None
    holdout: float | None = None
    failures: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0


class Archive:
    def __init__(self, root: Path, *, num_islands: int = 2, feature_bins: int = 4,
                 rng: random.Random | None = None) -> None:
        self.root = Path(root)
        self.num_islands = max(1, num_islands)
        self.feature_bins = max(1, feature_bins)
        self.rng = rng or random.Random(0)
        self.candidates: dict[str, Candidate] = {}
        self.grid: dict[str, str] = {}               # "island|size_bin|cost_bin" -> candidate id
        self.migrants: dict[int, list[str]] = {}     # island -> ids that migrated in
        self.champion: str | None = None
        self.iteration = 0
        self.reference: dict[str, float] = {}        # seed size/cost, the feature yardstick

    # ── persistence ──────────────────────────────────────────────────────────

    @classmethod
    def open(cls, root: Path, **kw) -> "Archive":
        archive = cls(root, **kw)
        path = archive.root / "archive.json"
        if path.is_file():
            data = json.loads(path.read_text())
            archive.candidates = {c["id"]: Candidate(**c) for c in data["candidates"]}
            archive.grid = data["grid"]
            archive.migrants = {int(k): v for k, v in data.get("migrants", {}).items()}
            archive.champion = data.get("champion")
            archive.iteration = data.get("iteration", 0)
            archive.reference = data.get("reference", {})
        return archive

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "iteration": self.iteration,
            "champion": self.champion,
            "reference": self.reference,
            "grid": self.grid,
            "migrants": self.migrants,
            "candidates": [asdict(c) for c in self.candidates.values()],
        }
        tmp = self.root / "archive.json.tmp"
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.root / "archive.json")    # atomic: a killed run resumes cleanly

    def log(self, entry: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "ledger.jsonl").open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def ledger(self, last: int | None = None) -> list[dict]:
        path = self.root / "ledger.jsonl"
        if not path.is_file():
            return []
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return entries[-last:] if last else entries

    def genome(self, cid: str) -> Genome:
        return Genome.load(self.root / "genomes" / cid)

    def next_id(self) -> str:
        return f"g{len(self.candidates):04d}"

    # ── insertion ────────────────────────────────────────────────────────────

    def add(self, candidate: Candidate, genome: Genome) -> bool:
        """Record a candidate. Returns True if it took (or kept) a grid cell."""
        self.candidates[candidate.id] = candidate
        genome.save(self.root / "genomes" / candidate.id)
        if candidate.status != POOLED:
            return False
        return self._place(candidate.island, candidate)

    def _place(self, island: int, candidate: Candidate) -> bool:
        cell = self._cell(island, candidate)
        incumbent = self.candidates.get(self.grid.get(cell, ""))
        if incumbent is None or _better(candidate, incumbent):
            self.grid[cell] = candidate.id
            return True
        return False

    def _bin(self, value: float, ref: float) -> int:
        # Bins span 0..2x the seed's value; the seed itself lands mid-range.
        if ref <= 0:
            return 0
        return max(0, min(self.feature_bins - 1, int(value / ref * self.feature_bins / 2)))

    def _cell(self, island: int, c: Candidate) -> str:
        size_bin = self._bin(c.size, self.reference.get("size", 0))
        cost_bin = self._bin(c.metrics.get("cost_usd", 0.0), self.reference.get("cost_usd", 0))
        return f"{island}|{size_bin}|{cost_bin}"

    def seen(self, fingerprint: str) -> Candidate | None:
        return next((c for c in self.candidates.values() if c.fingerprint == fingerprint), None)

    # ── selection ────────────────────────────────────────────────────────────

    def pool(self, island: int | None = None) -> list[Candidate]:
        pooled = [c for c in self.candidates.values() if c.status == POOLED]
        if island is None:
            return pooled
        members = set(self.migrants.get(island, []))
        return [c for c in pooled if c.island == island or c.id in members]

    def elites(self, island: int | None = None) -> list[Candidate]:
        ids = [cid for cell, cid in self.grid.items() if island is None or cell.startswith(f"{island}|")]
        return [self.candidates[cid] for cid in dict.fromkeys(ids)]

    def best(self, island: int | None = None) -> Candidate | None:
        pool = self.pool(island)
        if not pool:
            return None
        best = pool[0]
        for c in pool[1:]:
            if _better(c, best):
                best = c
        return best

    def sample_parent(self, island: int, *, exploit: float = 0.5, explore: float = 0.3) -> Candidate:
        """Exploit an elite, explore the island at random, or take the overall best."""
        r = self.rng.random()
        if r < exploit and (elites := self.elites(island)):
            return self.rng.choice(elites)
        if r < exploit + explore and (pool := self.pool(island)):
            return self.rng.choice(pool)
        best = self.best() or self.best(island)
        if best is None:
            raise RuntimeError("archive has no pooled candidate to mutate")
        return best

    def inspirations(self, parent: Candidate, k: int = 2) -> list[Candidate]:
        """Other strong, different elites — shown to the mutator as prior art."""
        others = [c for c in self.elites() if c.id != parent.id and c.fingerprint != parent.fingerprint]
        others.sort(key=lambda c: c.fitness or 0.0, reverse=True)
        return others[:k]

    def migrate(self) -> None:
        """Ring migration: each island's best joins the next island."""
        if self.num_islands < 2:
            return
        for island in range(self.num_islands):
            best = self.best(island)
            if best is None:
                continue
            target = (island + 1) % self.num_islands
            members = self.migrants.setdefault(target, [])
            if best.id not in members and best.island != target:
                members.append(best.id)
                self._place(target, best)

    def set_champion(self, cid: str) -> None:
        self.champion = cid
        dest = self.root / "champion"
        if dest.exists():
            shutil.rmtree(dest)
        self.genome(cid).save(dest)


def _better(a: Candidate, b: Candidate) -> bool:
    """Higher fitness wins; on a tie the smaller genome does (simplicity bias)."""
    fa, fb = a.fitness or 0.0, b.fitness or 0.0
    if fa != fb:
        return fa > fb
    return a.size < b.size
