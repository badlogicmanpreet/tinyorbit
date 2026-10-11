"""Freeze a new SWE-bench Verified task split, disjoint from the existing ones.

    python make_split.py --out tasks-holdout10.json --difficulty "1-4 hours" ">4 hours" --n 10 --seed 7

Evolution needs a held-out set the mutator never sees failures from, drawn
from the same distribution as the train set. This samples it reproducibly:
fetch the 500 Verified rows, keep the requested difficulty bands, drop every
instance already in a tasks-*.json here, then pick round-robin across repos
(seeded shuffle within each repo) so one repository cannot dominate. The
output has the same keys as the other task files, so run.py reads it as-is.

Stdlib only (the Hugging Face datasets-server rows API), so it runs in any venv.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = "SWE-bench/SWE-bench_Verified"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100
KEYS = ("instance_id", "repo", "base_commit", "version", "difficulty", "image",
        "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS")


def fetch_rows() -> list[dict]:
    rows: list[dict] = []
    offset, total = 0, None
    while total is None or offset < total:
        query = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "test",
                                        "offset": offset, "length": PAGE})
        with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=60) as resp:
            page = json.load(resp)
        total = page["num_rows_total"]
        rows.extend(r["row"] for r in page["rows"])
        offset += PAGE
    return rows


def existing_ids() -> set[str]:
    ids: set[str] = set()
    for path in HERE.glob("tasks*.json"):
        ids.update(t["instance_id"] for t in json.loads(path.read_text()))
    return ids


def to_task(row: dict) -> dict:
    task = {k: row[k] for k in KEYS}
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):        # the API ships these as JSON strings
        if isinstance(task[k], str):
            task[k] = json.loads(task[k])
    return task


def stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["instance_id"]):
        by_repo[row["repo"]].append(row)
    for bucket in by_repo.values():
        rng.shuffle(bucket)
    repos = sorted(by_repo)
    rng.shuffle(repos)
    picked: list[dict] = []
    while len(picked) < n and any(by_repo.values()):
        for repo in repos:
            if by_repo[repo] and len(picked) < n:
                picked.append(by_repo[repo].pop())
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--difficulty", nargs="+", default=["1-4 hours", ">4 hours"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = HERE / args.out
    if out.exists():
        print(f"{out.name} exists; frozen splits are never overwritten", file=sys.stderr)
        return 2
    taken = existing_ids()
    pool = [r for r in fetch_rows() if r["difficulty"] in args.difficulty and r["instance_id"] not in taken]
    picked = stratified(pool, args.n, args.seed)
    out.write_text(json.dumps([to_task(r) for r in picked], indent=1) + "\n")
    print(f"{out.name}: {len(picked)} of {len(pool)} eligible (seed {args.seed}, excluded {len(taken)})")
    for r in picked:
        print(f"  {r['instance_id']:40} {r['difficulty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
