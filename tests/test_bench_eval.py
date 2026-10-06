"""bench/ scoring and the SWE-bench evolution evaluator, against fake swebench output.

No docker, no network: each test writes the files `swebench eval` and run.py
would have written, then checks what the evolve loop would be told.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tinyorbit.evolve.evaluate import load_evaluator
from tinyorbit.evolve.genome import Genome

BENCH = Path(__file__).resolve().parent.parent / "bench"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"bench_{name}", BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(BENCH))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(BENCH))
    return module


scoring = load("scoring")
evolve_eval = load("evolve_eval")

IDS = ["a__a-1", "b__b-2", "c__c-3"]


def fake_run(run_dir: Path, run_id: str, *, scored=IDS) -> None:
    """What run.py --score leaves behind: results, transcripts, swebench output, scores.json."""
    (run_dir / "transcripts").mkdir(parents=True)
    with (run_dir / "results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance_id", "difficulty", "exit_reason", "api_calls", "cost_usd", "minutes", "diff_chars"])
        w.writerow(["a__a-1", "1-4 hours", "completed", 10, "1.50", "4.0", 900])
        w.writerow(["b__b-2", "1-4 hours", "completed", 20, "?", "6.0", 300])        # cost unparsed
        w.writerow(["c__c-3", ">4 hours", "max_turns", 60, "3.00", "20.0", 0])
    (run_dir / "transcripts" / "b__b-2.messages.json").write_text(json.dumps({"messages": [
        {"role": "user", "content": "fix the parser"},
        {"role": "assistant", "content": [{"type": "text", "text": "Done, my new test passes."}]},
    ]}))
    summary = {"resolved_ids": ["a__a-1"], "unresolved_ids": ["b__b-2"], "empty_patch_ids": ["c__c-3"]}
    summary = {k: [i for i in v if i in scored] for k, v in summary.items()}
    (run_dir / f"tinyorbit.{run_id}.json").write_text(json.dumps(summary))
    report = run_dir / "logs" / "run_evaluation" / run_id / "tinyorbit" / "b__b-2" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"b__b-2": {
        "resolved": False, "patch_exists": True, "patch_successfully_applied": True,
        "tests_status": {"FAIL_TO_PASS": {"success": [], "failure": ["test_parse_edge"]},
                         "PASS_TO_PASS": {"success": ["test_ok"], "failure": ["test_roundtrip"]}}}}))
    scoring.write_scores(run_dir, scoring.parse_scores(run_dir, run_id, IDS))


def test_parse_scores_merges_summary_and_reports(tmp_path):
    fake_run(tmp_path, "r1")
    scores = json.loads((tmp_path / "scores.json").read_text())
    assert scores["resolved"] == 1 and scores["total"] == 3
    inst = scores["instances"]
    assert inst["a__a-1"]["status"] == "resolved"
    assert inst["c__c-3"]["status"] == "empty_patch"
    assert inst["b__b-2"]["fail_to_pass_failed"] == ["test_parse_edge"]          # report.json wins
    assert inst["b__b-2"]["pass_to_pass_failed"] == ["test_roundtrip"]


def test_result_from_run_feeds_failures_to_the_mutator(tmp_path):
    fake_run(tmp_path, "r1")
    result = evolve_eval.result_from_run(tmp_path, IDS)
    assert result.score == pytest.approx(1 / 3)
    assert result.cost_usd == pytest.approx(4.5)
    assert result.metrics["cost_usd"] == pytest.approx(1.5)        # per task, comparable across stages
    by_task = {f.task: f for f in result.failures}
    assert set(by_task) == {"b__b-2", "c__c-3"}
    assert "test_parse_edge" in by_task["b__b-2"].summary and "broke: test_roundtrip" in by_task["b__b-2"].summary
    assert "my new test passes" in by_task["b__b-2"].detail       # the condensed transcript rides along
    assert "no patch" in by_task["c__c-3"].summary and "max_turns" in by_task["c__c-3"].summary


def test_unscored_instance_is_a_harness_error_not_a_failure(tmp_path):
    fake_run(tmp_path, "r1", scored=IDS[:2])
    with pytest.raises(evolve_eval.HarnessError, match="c__c-3"):
        evolve_eval.result_from_run(tmp_path, IDS)


def test_load_evaluator_from_file_spec_and_kwargs():
    ev = load_evaluator(f"{BENCH / 'evolve_eval.py'}:make", {"jobs": "2", "model": "m"})
    assert ev.jobs == 2 and ev.model == "m"
    g = Genome({"memory.md": "x"})
    assert ev.run_id(g, "train") == f"evo-{g.fingerprint()}-train-{ev.settings}"
    other = load_evaluator(f"{BENCH / 'evolve_eval.py'}:make", {"max_turns": "30"})
    assert other.settings != ev.settings                      # different settings never share a cache entry
    with pytest.raises(ValueError, match="not found"):
        load_evaluator("missing/eval.py:make")


async def test_evaluate_reuses_a_finished_run_without_spawning(tmp_path):
    ev = evolve_eval.make(python="/nonexistent/python")      # spawning would fail loudly
    ev.logs = tmp_path
    ev.stages = {"train": "tasks.json"}
    ids = [t["instance_id"] for t in json.loads((BENCH / "tasks.json").read_text())]
    g = Genome({"memory.md": "x"})
    run_id = ev.run_id(g, "train")
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    scoring.write_scores(run_dir, {"run_id": run_id, "total": len(ids), "resolved": len(ids),
                                   "instances": {i: {"status": "resolved"} for i in ids}})
    result = await ev.evaluate(g, "train")
    assert result.score == 1.0 and result.failures == []
    with pytest.raises(ValueError, match="unknown stage"):
        await ev.evaluate(g, "nope")


def test_frozen_splits_are_disjoint_and_well_formed():
    sets = {name: json.loads((BENCH / f).read_text()) for name, f in evolve_eval.STAGE_TASKS.items()}
    ids = {name: {t["instance_id"] for t in tasks} for name, tasks in sets.items()}
    assert not ids["train"] & ids["holdout"] and not ids["smoke"] & ids["holdout"]
    assert len(ids["holdout"]) == 10 and len(ids["smoke"]) == 3
    for task in sets["holdout"]:
        assert isinstance(task["FAIL_TO_PASS"], list) and task["image"].startswith("swebench/")
