"""Self-evolving layer: genome overlay, edits, reflection, and the outer loop.

Network-free: the model is a scripted `complete` function, the benchmark is a
fake evaluator whose score depends on what the genome says.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tinyorbit.bootstrap import Config, setup
from tinyorbit.evolve.archive import Archive
from tinyorbit.evolve.edits import Edit, EditError, apply_edits, parse_edits
from tinyorbit.evolve.evaluate import EvalResult, Failure, Stage, run_cascade
from tinyorbit.evolve.genome import Genome, load_overlay, seed_genome
from tinyorbit.evolve.loop import EvolveConfig, evolve, fitness
from tinyorbit.evolve.mutate import LLMMutator, MutationContext, Proposal, build_mutation_prompt, parse_proposal
from tinyorbit.evolve.reflect import (
    Lesson,
    condense_transcript,
    merge_lessons,
    parse_lessons,
    reflect,
    reflect_into_genome,
    render_lessons,
)
from tinyorbit.providers.base import ModelResponse

# ── genome + overlay ─────────────────────────────────────────────────────────

def test_seed_genome_exports_prompt_and_tool_docs():
    seed = seed_genome()
    assert "{cwd}" in seed.files["system_prompt.md"]
    assert "tools/Bash.md" in seed.files and seed.files["tools/Bash.md"].strip()


def test_genome_roundtrip_and_path_whitelist(tmp_path):
    g = Genome({"memory.md": "be careful", "skills/verify/SKILL.md": "---\nname: verify\n---\nrun tests"})
    g.save(tmp_path / "g")
    (tmp_path / "g" / "notes.md").write_text("ignored: not a genome path")
    loaded = Genome.load(tmp_path / "g")
    assert loaded.files == g.files
    assert loaded.fingerprint() == g.fingerprint()
    with pytest.raises(ValueError):
        Genome({"../escape.md": "x"}).save(tmp_path / "bad")


def test_bootstrap_applies_genome_overlay(tmp_path):
    genome_dir = tmp_path / "genome"
    Genome({
        "system_prompt.md": "You are EVOLVED in {cwd} on {platform}. Braces {ok}.",
        "memory.md": "Harness rule: prefer small diffs.",
        "lessons.md": render_lessons([Lesson("Run the suite", "Run existing tests before finishing.")]),
        "skills/verify.md": "---\nname: verify\ndescription: check work\n---\nRun the tests.",
        "tools/Bash.md": "EVOLVED bash description",
    }).save(genome_dir)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("Project rule wins.")
    config = Config(cwd=repo, model="m", trusted=True, genome_dir=genome_dir, data_dir=tmp_path / ".d")
    setup(config)

    prompt = config.capabilities["system_prompt"][0]["text"]
    assert prompt.startswith(f"You are EVOLVED in {repo}")
    assert "Braces {ok}." in prompt                          # no str.format crash on stray braces
    # genome memory first, project memory last (project overrides harness)
    assert prompt.index("prefer small diffs") < prompt.index("Run the suite") < prompt.index("Project rule wins")
    tools = {t.name: t for t in config.capabilities["tools"]}
    assert tools["Bash"].description == "EVOLVED bash description"
    assert "Skill" in tools and "verify" in tools["Skill"].description


def test_seed_genome_is_behavior_identical(tmp_path):
    """Generation 0 must reproduce the default harness byte for byte, or every
    evolved score would be measured against a baseline that never shipped."""
    seed_genome().save(tmp_path / "g0")
    with_seed = Config(cwd=tmp_path, model="m", trusted=True, genome_dir=tmp_path / "g0", data_dir=tmp_path / ".d")
    plain = Config(cwd=tmp_path, model="m", trusted=True, data_dir=tmp_path / ".d")
    setup(with_seed)
    setup(plain)
    assert with_seed.capabilities["system_prompt"] == plain.capabilities["system_prompt"]
    assert [t.to_api() for t in with_seed.capabilities["tools"]] == [t.to_api() for t in plain.capabilities["tools"]]


def test_no_genome_means_no_change(tmp_path):
    from tinyorbit.prompt import STATIC_TEMPLATE

    config = Config(cwd=tmp_path, model="m", trusted=True, data_dir=tmp_path / ".d")
    setup(config)
    text = config.capabilities["system_prompt"][0]["text"]
    assert text.startswith(STATIC_TEMPLATE.split("{cwd}")[0])
    assert load_overlay(tmp_path / "missing").system_template is None


# ── edits ────────────────────────────────────────────────────────────────────

BLOCK = "<<<<<<< SEARCH {path}\n{search}=======\n{replace}>>>>>>> REPLACE\n"


def test_parse_and_apply_exact_edit():
    g = Genome({"system_prompt.md": "a\n- Keep replies brief.\nz\n"})
    text = "<summary>x</summary>\n" + BLOCK.format(
        path="system_prompt.md", search="- Keep replies brief.\n", replace="- Keep replies brief. Test first.\n")
    edits = parse_edits(text)
    assert edits == [Edit("system_prompt.md", "- Keep replies brief.\n", "- Keep replies brief. Test first.\n")]
    assert apply_edits(g, edits).files["system_prompt.md"] == "a\n- Keep replies brief. Test first.\nz\n"


def test_fuzzy_whitespace_match_and_create():
    g = Genome({"memory.md": "  - one\n  - two\nend\n"})
    child = apply_edits(g, [
        Edit("memory.md", "- one\n- two\n", "- merged\n"),
        Edit("skills/new.md", "", "fresh skill\n"),
    ])
    assert child.files["memory.md"] == "- merged\nend\n"
    assert child.files["skills/new.md"] == "fresh skill\n"


@pytest.mark.parametrize("edit, message", [
    (Edit("memory.md", "missing", "x"), "not found"),
    (Edit("memory.md", "dup", "x"), "matches 2"),
    (Edit("main.py", "", "x"), "not allowed"),
    (Edit("lessons.md", "x", "y"), "does not exist"),
])
def test_edit_errors(edit, message):
    with pytest.raises(EditError, match=message):
        apply_edits(Genome({"memory.md": "dup dup"}), [edit])


# ── reflection (inner loop) ──────────────────────────────────────────────────

def test_lessons_roundtrip_dedupe_and_cap():
    a, b = Lesson("Run tests", "Use the repo's suite.", "failed", "s1"), Lesson("Read first", "Read before edit.")
    assert parse_lessons(render_lessons([a, b])) == [a, b]
    replaced = merge_lessons([a, b], [Lesson("run  TESTS!", "Updated.")])
    assert [x.body for x in replaced] == ["Read before edit.", "Updated."]   # replaced and moved last
    many = merge_lessons([], [Lesson(f"t{i}", "b") for i in range(30)], limit=5)
    assert [x.title for x in many] == ["t25", "t26", "t27", "t28", "t29"]    # oldest evicted


def test_condense_transcript_keeps_head_and_tail():
    msgs = [{"role": "user", "content": "THE TASK"}]
    for i in range(400):
        msgs.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": f"step {i}"}}]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok", "is_error": i == 399}]})
    text = condense_transcript(msgs, budget=4_000)
    assert len(text) <= 4_200
    assert text.startswith("USER: THE TASK") and "steps omitted" in text and "RESULT(error)" in text


async def test_reflect_filters_known_and_malformed():
    seen = {}

    async def complete(system, user):
        seen["user"] = user
        return 'Sure:\n```json\n[{"title": "Run tests", "lesson": "dup"}, {"title": "Verify with repo tests",' \
               ' "lesson": "Run the existing suite, not only new tests."}, {"title": ""}]\n```'

    known = [Lesson("Run tests", "Use the suite.")]
    out = await reflect([{"role": "user", "content": "fix bug"}], complete,
                        outcome="failed", feedback="FAIL_TO_PASS still red", known=known, source="i1")
    assert [x.title for x in out] == ["Verify with repo tests"]
    assert out[0].outcome == "failed" and out[0].source == "i1"
    assert "FAIL_TO_PASS still red" in seen["user"] and "Run tests: Use the suite." in seen["user"]


async def test_reflect_into_genome_writes_lessons_file(tmp_path):
    async def complete(system, user):
        return '[{"title": "Check edge cases", "lesson": "Test empty input."}]'

    new = await reflect_into_genome(tmp_path, [{"role": "user", "content": "x"}], complete, outcome="resolved")
    assert len(new) == 1
    assert "**Check edge cases**" in (tmp_path / "lessons.md").read_text()
    # second identical reflection adds nothing: the title is already known
    assert await reflect_into_genome(tmp_path, [{"role": "user", "content": "x"}], complete) == []


async def test_print_mode_reflect_hook_uses_provider(tmp_path):
    from tinyorbit.repl import _reflect
    from tinyorbit.state import SessionState

    class FakeProvider:
        async def stream(self, req, abort):
            usage = SimpleNamespace(input_tokens=3, output_tokens=2, cache_read_input_tokens=0,
                                    cache_creation_input_tokens=0)
            msg = SimpleNamespace(content=[SimpleNamespace(type="text", text='[{"title": "T", "lesson": "L"}]')],
                                  usage=usage)
            yield ModelResponse(msg)

    config = Config(cwd=tmp_path, model="m", genome_dir=tmp_path / "g", reflect=True)
    session = SessionState(messages=[{"role": "user", "content": "task"}])
    await _reflect(config, session, FakeProvider(), "completed")
    assert "**T**: L" in (tmp_path / "g" / "lessons.md").read_text()
    assert session.cost.api_calls == 1                 # reflection cost lands on the session ledger


# ── mutation prompt ──────────────────────────────────────────────────────────

def test_mutation_prompt_carries_failures_ledger_and_inspirations():
    from tinyorbit.evolve.archive import Candidate

    parent = Candidate("g0001", "g0000", 1, 0, "fp", 10, stage_scores={"train": 0.5}, fitness=0.5,
                       failures=[{"task": "django-1", "summary": "declared done early", "detail": "trace"}])
    other = Candidate("g0002", "g0000", 1, 1, "fp2", 12, summary="added verify skill", fitness=0.6)
    prompt = build_mutation_prompt(
        Genome({"memory.md": "m"}),
        MutationContext(parent, [(other, Genome({"memory.md": "m", "skills/v.md": "verify"}))],
                        [{"outcome": "invalid", "summary": "tried X", "reason": "not found"}]),
    )
    assert "declared done early" in prompt and "tried X" in prompt and "[not found]" in prompt
    assert '<file path="skills/v.md">' in prompt and "added verify skill" in prompt
    proposal = parse_proposal("<summary> tighten\n prompt </summary>\n" + BLOCK.format(
        path="memory.md", search="m\n", replace="n\n"))
    assert proposal.summary == "tighten prompt" and len(proposal.edits) == 1


# ── outer loop ───────────────────────────────────────────────────────────────

class FakeEvaluator:
    """Train rewards 'run the tests' and (spuriously) 'OVERFIT'; held-out punishes OVERFIT."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def evaluate(self, genome, stage):
        self.calls.append((genome.fingerprint(), stage))
        p = genome.files.get("system_prompt.md", "")
        good = 0.3 if "run the tests" in p else 0.0
        if stage == "smoke":
            score = 0.0 if "BROKEN" in p else 1.0
        elif stage == "holdout":
            score = 0.4 + good - (0.2 if "OVERFIT" in p else 0.0)
        else:
            score = 0.4 + good + (0.1 if "OVERFIT" in p else 0.0)
        fails = [Failure("task-1", "declared success after its own tests", "...")] if score < 0.7 else []
        return EvalResult(score, metrics={"cost_usd": 1.0}, failures=fails)


def edit_prompt(add: str, summary: str) -> Proposal:
    return Proposal(summary, [Edit("system_prompt.md", "- Keep replies brief.", f"- Keep replies brief. {add}")])


def scripted_mutator(proposals):
    queue = list(proposals)
    contexts: list[MutationContext] = []

    async def mutate(genome, ctx):
        contexts.append(ctx)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return mutate, contexts


PROPOSALS = [
    edit_prompt("Always run the tests before finishing.", "verify before done"),
    Proposal("bad quote", [Edit("system_prompt.md", "text that is not there", "x")]),
    edit_prompt("BROKEN", "breaks smoke"),
    edit_prompt("OVERFIT run the tests", "overfits train"),
]


def loop_config(iterations):
    return EvolveConfig(iterations=iterations, stages=[Stage("smoke", 0.5), Stage("train")],
                        holdout="holdout", num_islands=1, rng_seed=1)


async def test_evolve_promotes_only_what_generalizes(tmp_path):
    evaluator = FakeEvaluator()
    mutator, contexts = scripted_mutator(PROPOSALS)
    archive = await evolve(seed_genome(), evaluator, mutator, tmp_path / "run", loop_config(4), log=lambda _: None)

    outcomes = [e["outcome"] for e in archive.ledger()]
    assert outcomes == ["seed", "champion", "invalid", "rejected", "pooled"]
    assert archive.champion == "g0001"
    champ = archive.candidates["g0001"]
    assert champ.holdout == pytest.approx(0.7) and champ.fitness < 0.7      # size penalty applied
    assert "run the tests" in (tmp_path / "run" / "champion" / "system_prompt.md").read_text()
    # BROKEN never reached the train stage: the cascade stopped it at smoke
    broken = archive.candidates["g0002"]
    assert broken.status == "rejected" and list(broken.stage_scores) == ["smoke"]
    # later proposers saw the earlier rejection in their ledger
    assert any(e.get("reason") for e in contexts[-1].ledger)


async def test_evolve_resumes_from_checkpoint(tmp_path):
    evaluator = FakeEvaluator()
    mutator, _ = scripted_mutator(PROPOSALS)
    await evolve(seed_genome(), evaluator, mutator, tmp_path / "run", loop_config(2), log=lambda _: None)
    archive = await evolve(seed_genome(), evaluator, mutator, tmp_path / "run", loop_config(4), log=lambda _: None)
    assert archive.iteration == 4 and len(archive.ledger()) == 5
    seed_fp = seed_genome().fingerprint()
    assert sum(1 for fp, _ in evaluator.calls if fp == seed_fp) == 3     # smoke, train, holdout: once
    on_disk = json.loads((tmp_path / "run" / "archive.json").read_text())
    assert on_disk["champion"] == "g0001"


async def test_duplicate_and_mutator_errors_are_logged_not_fatal(tmp_path):
    write = Proposal("add memory", [Edit("memory.md", "", "Run the tests.\n")])
    evaluator = FakeEvaluator()
    mutator, _ = scripted_mutator([write, write, RuntimeError("boom"), PROPOSALS[2]])
    archive = await evolve(seed_genome(), evaluator, mutator, tmp_path / "run", loop_config(3),
                           log=lambda _: None)
    reasons = [e.get("reason", "") for e in archive.ledger()]
    # The repeat is a duplicate of g0001 (parent = seed) or a no-op (parent =
    # g0001); either way it is rejected before any evaluation is spent.
    assert any("duplicate of g0001" in r or "no-op" in r for r in reasons)
    assert any("mutator error: boom" in r for r in reasons)
    # the failed call did not use up iteration 2: it was retried and produced g0002
    assert archive.iteration == 3 and len(archive.candidates) == 3
    assert len(evaluator.calls) == 3 + 2 + 1       # seed: 3 stages; g0001: smoke+train; g0002: smoke


async def test_repeated_mutator_failures_abort_without_burning_iterations(tmp_path):
    evaluator = FakeEvaluator()
    mutator, _ = scripted_mutator([RuntimeError("401")] * 3)
    with pytest.raises(RuntimeError, match="3 times in a row"):
        await evolve(seed_genome(), evaluator, mutator, tmp_path / "run", loop_config(5), log=lambda _: None)
    assert Archive.open(tmp_path / "run").iteration == 0          # resume starts at attempt 0 again


def test_archive_grid_keeps_lean_and_verbose_variants(tmp_path):
    from tinyorbit.evolve.archive import Candidate

    archive = Archive(tmp_path, num_islands=1, feature_bins=4)
    archive.reference = {"size": 1000.0, "cost_usd": 1.0}
    lean = Candidate("a", None, 0, 0, "fa", 600, fitness=0.6)
    verbose = Candidate("b", None, 0, 0, "fb", 1900, fitness=0.7)
    worse_lean = Candidate("c", None, 0, 0, "fc", 610, fitness=0.5)
    for c in (lean, verbose, worse_lean):
        archive.add(c, Genome({"memory.md": c.id}))
    assert {c.id for c in archive.elites()} == {"a", "b"}
    assert archive.best().id == "b"


async def test_cascade_stops_at_first_missed_gate():
    evaluator = FakeEvaluator()
    g = Genome({"system_prompt.md": "BROKEN"})
    results, passed = await run_cascade(evaluator, g, [Stage("smoke", 0.5), Stage("train")])
    assert not passed and list(results) == ["smoke"]


def test_fitness_penalizes_growth_only():
    assert fitness(0.8, 900, 1000, 0.01) == 0.8
    assert fitness(0.8, 3000, 1000, 0.01) == pytest.approx(0.78)


async def test_llm_mutator_wraps_complete():
    async def complete(system, user):
        assert "<<<<<<< SEARCH" in system and "Propose your edits now." in user
        return "<summary>s</summary>\n" + BLOCK.format(path="memory.md", search="", replace="new\n")

    from tinyorbit.evolve.archive import Candidate

    proposal = await LLMMutator(complete)(Genome(), MutationContext(Candidate("g", None, 0, 0, "f", 0)))
    assert proposal.edits[0].path == "memory.md"


# ── event log + report ───────────────────────────────────────────────────────

async def test_every_step_is_in_the_event_log(tmp_path):
    from tinyorbit.evolve.events import EventLog

    mutator, _ = scripted_mutator(PROPOSALS)
    seen: list[int] = []
    cfg = loop_config(4)
    cfg.num_islands, cfg.migration_interval = 2, 2
    await evolve(seed_genome(), FakeEvaluator(), mutator, tmp_path / "run", cfg, log=lambda _: None,
                 on_iteration=lambda a: seen.append(a.iteration))
    events = EventLog(tmp_path / "run").read()
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and "seed" in kinds
    assert kinds.count("iteration_start") == kinds.count("iteration_end") == 4
    assert kinds.count("stage_start") == kinds.count("stage_end")
    assert seen == [0, 1, 2, 3, 4]                      # after the seed, then after every iteration
    by_kind = lambda k: [e for e in events if e["kind"] == k]          # noqa: E731
    assert {e["pick"] for e in by_kind("iteration_start")} <= {"exploit", "explore", "best"}
    assert by_kind("invalid")[0]["reason"] and by_kind("invalid")[0]["iteration"] == 1
    smoke_fail = [e for e in by_kind("stage_end") if e["stage"] == "smoke" and e["passed"] is False]
    assert smoke_fail and smoke_fail[0]["candidate"] == "g0002"
    promo = by_kind("promotion")
    assert promo[0]["promoted"] and promo[0]["champion_after"] == "g0001"
    assert any(not p["promoted"] and "held-out" in p["reason"] for p in promo)     # OVERFIT kept out
    assert by_kind("migration")                                                     # every 2 iterations
    first = by_kind("mutation")[0]
    diff = (tmp_path / "run" / first["artifacts"]["diff"]).read_text()
    assert "+- Keep replies brief. Always run the tests" in diff
    assert by_kind("candidate")[0]["place"]["cell"].count("|") == 2


async def test_report_renders_a_finished_run(tmp_path):
    import json as _json
    import re

    from tinyorbit.evolve.report import _PRE, write_report

    mutator, _ = scripted_mutator(PROPOSALS)
    await evolve(seed_genome(), FakeEvaluator(), mutator, tmp_path / "run", loop_config(4), log=lambda _: None)
    page = write_report(tmp_path / "run").read_text()
    for needle in ("Why evolve the harness?", "No manual work is needed", "In nature", "Darwin",
                   "Watch it happen", "The story in one minute", "Family tree", "Islands and niches", "Island A",
                   "verify before done", "new champion", "easy safety check", "unseen problems",
                   "Finished.", "Promoted:", "<svg", "+- Keep replies brief. Always run the tests"):
        assert needle in page, needle
    assert "http-equiv" not in page                                     # no refresh once finished
    assert "var rough=" in page and "<script src" not in page           # one offline file, rough.js inlined
    assert "Glossary" not in page and "AI coding agent" not in page
    assert not re.search("[\u2014\u2013]", _PRE.sub("", page))          # house style: commas, not dashes
    # the animated story is in the same page, its scenes computed from the run
    data = _json.loads(re.search(r'id="data">(.*?)</script>', page, re.S).group(1).replace("<\\/", "</"))
    captions = [s["caption"] for s in data["scenes"]]
    assert captions[0] == "Every species starts somewhere." and "g0001 is the new champion." in captions
    assert "A mutation that does not work." in captions                 # the bad-quote edit
    assert any(c.startswith("After 4 generations") for c in captions)
    assert [n["id"] for n in data["nodes"]][:2] == ["g0000", "g0001"]


def test_report_drops_dashes_outside_quoted_files():
    from tinyorbit.evolve.report import no_dashes

    page = "<p>idea \u2014 verify first</p><pre>keep \u2014 verbatim</pre><p>0.5\u20131.0</p>"
    assert no_dashes(page) == "<p>idea, verify first</p><pre>keep \u2014 verbatim</pre><p>0.5, 1.0</p>"


def test_report_handles_a_run_that_just_started(tmp_path):
    from tinyorbit.evolve.events import EventLog
    from tinyorbit.evolve.report import render

    log = EventLog(tmp_path)
    log.emit("run_start", resumed=False, iteration=0, config={"iterations": 5, "stages": [{"name": "smoke"}]})
    log.emit("stage_start", iteration=-1, candidate="g0000", stage="smoke", fingerprint="x")
    page = render(tmp_path)
    assert "Running now: <b>easy safety check</b> for the original" in page
    assert 'http-equiv="refresh"' in page


async def test_budget_stops_before_an_attempt_that_could_overshoot(tmp_path):
    from tinyorbit.evolve.report import render

    class Priced(FakeEvaluator):
        async def evaluate(self, genome, stage):
            result = await super().evaluate(genome, stage)
            result.cost_usd = 10.0
            return result

    cfg = loop_config(4)
    cfg.budget_usd = 45.0          # seed spends 30 (3 stages); one attempt could cost 30 more
    mutator, _ = scripted_mutator(PROPOSALS)
    archive = await evolve(seed_genome(), Priced(), mutator, tmp_path / "run", cfg, log=lambda _: None)
    assert archive.iteration == 0
    assert "Finished early to stay within budget" in render(tmp_path / "run")
    # Resumed with more money; spending carries over from the log: seed 30, g0001 30 (all
    # three stages), the invalid edit 0, BROKEN 10 (stopped at smoke) = 70, and 70 + 30 > 95.
    cfg.budget_usd = 95.0
    archive = await evolve(seed_genome(), Priced(), mutator, tmp_path / "run", cfg, log=lambda _: None)
    assert archive.iteration == 3
