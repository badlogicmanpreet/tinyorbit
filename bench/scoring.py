"""Read what `swebench eval` wrote and reduce it to one scores.json per run.

    swebench eval verified -p predictions.jsonl --run-id R     (cwd = the run dir)
        R/<model>.<run-id>.json                                     summary: resolved_ids, ...
        R/logs/run_evaluation/<run-id>/<model>/<iid>/report.json    per instance: tests_status

Both layouts are found by glob rather than fixed paths, so a minor swebench
version change in directory naming does not silently zero a run. The
per-instance report wins when present (it carries the failing test names the
evolution mutator reads); the summary fills gaps. An instance in neither is
"missing": it was never scored, which is a harness problem, not a verdict.

Stdlib only: imported by run.py (bench venv) and by evolve_eval.py (tinyorbit venv).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

MODEL_NAME = "tinyorbit"          # model_name_or_path in every prediction line
SCORES = "scores.json"
# summary key -> status, checked in order (resolved beats everything)
SUMMARY_STATUS = (("resolved_ids", "resolved"), ("unresolved_ids", "unresolved"),
                  ("empty_patch_ids", "empty_patch"), ("error_ids", "error"))


def swebench_cmd() -> str:
    """The swebench CLI next to the running interpreter (bench/.venv), else PATH."""
    local = Path(sys.executable).parent / "swebench"
    return str(local) if local.exists() else (shutil.which("swebench") or "swebench")


def eval_command(run_id: str, jobs: int = 1) -> list[str]:
    return [swebench_cmd(), "eval", "verified", "-p", "predictions.jsonl", "--run-id", run_id, "-j", str(jobs)]


def _failed(tests_status: dict, key: str) -> list[str]:
    bucket = tests_status.get(key) or {}
    return list(bucket.get("failure") or [])


def parse_scores(run_dir: Path, run_id: str, instance_ids: list[str]) -> dict:
    """Merge summary and per-instance reports into {instances: {iid: {...}}, resolved, total}."""
    instances: dict[str, dict] = {iid: {"status": "missing"} for iid in instance_ids}

    for summary_path in sorted(run_dir.glob(f"*.{run_id}.json")):
        summary = json.loads(summary_path.read_text())
        for key, status in reversed(SUMMARY_STATUS):     # reversed so "resolved" is written last
            for iid in summary.get(key) or []:
                if iid in instances:
                    instances[iid] = {"status": status}

    for report_path in sorted(run_dir.glob(f"logs/run_evaluation/{run_id}/**/report.json")):
        for iid, rep in json.loads(report_path.read_text()).items():
            if iid not in instances:
                continue
            ts = rep.get("tests_status") or {}
            status = "resolved" if rep.get("resolved") else (
                "empty_patch" if rep.get("patch_exists") is False else "unresolved")
            instances[iid] = {
                "status": status,
                "fail_to_pass_failed": _failed(ts, "FAIL_TO_PASS"),
                "pass_to_pass_failed": _failed(ts, "PASS_TO_PASS"),
                "patch_applied": rep.get("patch_successfully_applied"),
            }

    resolved = sorted(i for i, r in instances.items() if r["status"] == "resolved")
    return {"run_id": run_id, "total": len(instance_ids), "resolved": len(resolved),
            "resolved_ids": resolved, "instances": instances}


def write_scores(run_dir: Path, scores: dict) -> Path:
    path = run_dir / SCORES
    path.write_text(json.dumps(scores, indent=1) + "\n")
    return path
