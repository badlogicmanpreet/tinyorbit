"""Story, the animated, plain-language opening of the report.

report.py is the full record; this module is its first section, the explainer.
It turns a run into a short sequence of scenes (pick a parent, mutate, test,
select) played on one canvas: a family tree growing across the islands on the
left, a test lab filling in pass/fail squares on the right, and one sentence of
narration per scene. Before it come the why, and the analogy to evolution in
nature.

The scenes are computed here from the replayed run and embedded as JSON; a
small inline script only plays them (data-driven animation), so the report
stays a single file that works offline. This module takes the Run that
report.py loads and never imports report.py, so there is no import cycle.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

from tinyorbit.evolve.archive import INVALID

if TYPE_CHECKING:
    from tinyorbit.evolve.report import Run

LANE_NAMES = "ABCDEFGH"
STAGE_LABELS = {"smoke": "Easy check", "train": "Practice set", "holdout": "Unseen problems"}
PICK_WORDS = {
    "exploit": "one of the strongest on its island",
    "explore": "a random member of its island, to keep trying new directions",
    "best": "the best version found so far",
}


def _short(text: str, limit: int = 150) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def lane_name(island: int) -> str:
    return LANE_NAMES[island % len(LANE_NAMES)]


def _row(ev: dict, animate: bool) -> dict:
    tasks = (ev.get("info") or {}).get("tasks") or {}
    m = ev.get("metrics") or {}
    solved, total = int(m.get("resolved", round(ev["score"] * len(tasks)))), int(m.get("total", len(tasks)))
    gate = ev.get("min_score")
    result = f"{solved} of {total} solved"
    if gate:
        result += ", passed" if ev.get("passed") else f", needed {round(gate * total)}"
    return {"name": STAGE_LABELS.get(ev["stage"], ev["stage"]), "animate": animate, "result": result,
            "pass": ev.get("passed"), "tasks": [{"id": k, "ok": v == "resolved"} for k, v in tasks.items()]}


def _fraction(ev: dict | None) -> str:
    if not ev:
        return ""
    m = ev.get("metrics") or {}
    if "resolved" in m and "total" in m:
        return f"{int(m['resolved'])}/{int(m['total'])}"
    return f"{ev['score'] * 100:.0f}%"


def build(run: Run, costs: dict) -> dict:
    """Nodes of the family tree plus the ordered scenes that reveal them."""
    final, holdout = run.final_stage, run.holdout
    islands = max(1, run.archive.num_islands)
    attempts = sorted(i for i in run.steps if i >= 0)
    width, left, top, lane_h = 640, 70, 30, 420 / islands
    dx = (width - left - 60) / max(1, len(attempts))

    def lane_y(island: int) -> float:
        return top + lane_h * (island + 0.5)

    nodes: list[dict] = []
    index: dict[str, int] = {}

    def add_node(**node) -> None:
        index[node["id"]] = len(nodes)
        nodes.append(node)

    scenes: list[dict] = []
    champion = None
    migrations: list[dict] = []

    def scene(caption: str, sub: str = "", *, lab=None, focus=(), dur=4200, new=None, parent=None) -> None:
        scenes.append({"caption": caption, "sub": sub, "lab": lab, "focus": list(focus), "new": new,
                       "parent": parent, "nodes": len(nodes), "champion": champion,
                       "migrations": list(migrations), "dur": dur})

    seed = run.seed
    seed_step = run.steps.get(-1)
    by_stage = {ev["stage"]: ev for ev in (seed_step.stages if seed_step else [])}
    lanes = [{"name": f"Island {lane_name(i)}", "y": lane_y(i), "h": lane_h} for i in range(islands)]
    shape = {"islands": islands, "lanes": lanes, "width": width, "top": top, "height": top + 420 + 30}
    if seed is None:
        scene("The run has just started.", "Come back in a few minutes.")
        return {"nodes": nodes, "scenes": scenes, **shape}

    scene("Every species starts somewhere.",
          "Ours is the standard tinyorbit harness: the system prompt, memory, lessons, skills and tool "
          "descriptions the harness hands the model before it starts work. The model itself never "
          "changes. Only these artifacts will evolve.", dur=6500)
    add_node(id=seed.id, x=left - 20, y=top + 210, island=-1, label="original",
             score=_fraction(by_stage.get(final)), kind="seed", parent=None)
    rows = [_row(by_stage[s], True) for s in run.stages + ([holdout] if holdout else []) if s in by_stage]
    scene(f"First we measure the original, {seed.id}.",
          "It tries real programming problems taken from open-source projects. Each square is one "
          "problem: green means its fix passed the project's own tests, red means it did not.",
          lab={"title": f"Testing the original ({seed.id})", "rows": rows,
               "verdict": f"{_fraction(by_stage.get(final))} on practice. This is the score to beat.", "tone": "info"},
          focus=[seed.id], new=seed.id, dur=4000 + 140 * sum(len(r["tasks"]) for r in rows))
    champion = seed.id
    if islands > 1:
        scene(f"The population is split across {islands} islands.",
              "Like animals on separate islands, each group evolves on its own, so the search does not "
              "pile onto a single idea. The original starts on every island.", focus=[seed.id], dur=5500)

    for it in attempts:
        st = run.steps[it]
        x = left + dx * (attempts.index(it) + 1) - 10
        island = st.start.get("island", it % islands)
        parent = st.start.get("parent", seed.id)
        lane = lane_name(island)
        n = attempts.index(it) + 1
        scene(f"Generation {n}, island {lane}: choose a parent.",
              f"We pick {parent}: {PICK_WORDS.get(st.start.get('pick', ''), 'a member of the population')}.",
              focus=[parent], dur=3600)
        idea = _short(st.mutation.get("summary") or "")
        if st.outcome == INVALID or not st.candidate:
            reason = _short((st.invalid or {}).get("reason", "it could not be applied"), 110)
            add_node(id=f"x{it}", x=x, y=lane_y(island), island=island, label="discarded", score="",
                     kind="invalid", parent=parent)
            scene("A mutation that does not work.", f"The change was thrown out before testing: {reason}.",
                  focus=[f"x{it}"], new=f"x{it}", parent=parent, dur=4500)
            continue
        cand = st.candidate
        cid = cand["candidate"]
        stages = {ev["stage"]: ev for ev in st.stages}
        add_node(id=cid, x=x, y=lane_y(island), island=island, label=cid,
                 score=_fraction(stages.get(final)) or "stopped", kind=cand.get("status", "pooled"), parent=parent)
        scene("Mutation: the AI editor rewrites part of the harness.",
              f"It studies the problems {parent} got wrong and proposes a change. Its idea: “{idea}”",
              focus=[cid], new=cid, parent=parent, dur=7000)
        rows = [_row(stages[s], True) for s in run.stages if s in stages]
        reached_final = final in stages
        if not reached_final:
            verdict, tone = f"Failed the {rows[-1]['name'].lower()}, so it goes no further.", "fail"
        else:
            verdict, tone = f"{_fraction(stages[final])} on practice (the champion has {_champ_score(run, champion)}).", "info"
        scene(f"Natural selection: {cid} takes the tests.",
              "Cheap check first: if it cannot solve the easy problems, it is stopped before the expensive ones.",
              lab={"title": f"Testing {cid}", "rows": rows, "verdict": verdict, "tone": tone},
              focus=[cid], dur=3800 + 140 * sum(len(r["tasks"]) for r in rows))
        prom = st.promotion
        if holdout and holdout in stages:
            hold_rows = [dict(r, animate=False) for r in rows] + [_row(stages[holdout], True)]
            scene(f"{cid} beat the champion on practice. Now the real test.",
                  "It must also do better on problems it has never seen, the way a species must survive "
                  "outside its home valley. This catches changes that only memorised the practice set.",
                  lab={"title": f"Testing {cid} on unseen problems", "rows": hold_rows,
                       "verdict": "", "tone": "info"},
                  focus=[cid], dur=4000 + 140 * len(hold_rows[-1]["tasks"]))
        if prom.get("promoted"):
            champion = cid
            nodes[index[cid]]["kind"] = "champion"
            scene(f"{cid} is the new champion.",
                  f"It is the fittest so far: {_fraction(stages.get(final))} on practice"
                  + (f" and {_fraction(stages.get(holdout))} on unseen problems." if holdout in stages else "."),
                  lab=_lab_static(run, cid, stages, f"New champion: {cid}", "win"), focus=[cid], dur=5000)
        elif reached_final:
            why = ("It did well on practice but not on unseen problems, so it may have just got lucky."
                   if holdout in stages else "It is not better than the champion.")
            scene(f"{cid} survives, but does not win.",
                  why + " It stays in the population, and later generations may build on it.",
                  lab=_lab_static(run, cid, stages, "Kept in the population", "keep"), focus=[cid], dur=4800)
        for move in st.migration:
            migrations.append({"id": move["candidate"], "to": move["to"], "x": nodes[index[move["candidate"]]]["x"],
                               "y": lane_y(move["to"])})
            scene(f"Migration: {move['candidate']} travels to island {lane_name(move['to'])}.",
                  "Now and then the best of one island joins the next, so a good idea can spread, "
                  "like animals crossing to a neighbouring island.", focus=[move["candidate"]], dur=5200)

    scene(*_ending(run), lab={"title": "Where we ended", "summary": _summary(run, costs), "tone": "win"},
          focus=[champion] if champion else [], dur=12000)
    return {"nodes": nodes, "scenes": scenes, **shape}


def _champ_score(run: Run, cid: str | None, stage: str | None = None) -> str:
    """'7/10': the candidate's result on a stage (default: the final one), from its stage_end event."""
    stage = stage or run.final_stage
    for st in run.steps.values():
        for ev in st.stages:
            if ev.get("candidate") == cid and ev["stage"] == stage:
                return _fraction(ev)
    return "?"


def _lab_static(run: Run, cid: str, stages: dict, verdict: str, tone: str) -> dict:
    order = run.stages + ([run.holdout] if run.holdout else [])
    return {"title": f"Results for {cid}", "rows": [_row(stages[s], False) for s in order if s in stages],
            "verdict": verdict, "tone": tone}


def _ending(run: Run) -> tuple[str, str]:
    seed, champ = run.seed, run.champion
    if champ is None or seed is None or champ.id == seed.id:
        return ("No offspring beat the original this time.",
                "That is a result too: the current harness is hard to improve with small changes.")
    return (f"After {sum(1 for i in run.steps if i >= 0)} generations, {champ.id} is the fittest.",
            f"It went from {_champ_score(run, seed.id)} to {_champ_score(run, champ.id)} on the practice problems.")


def _summary(run: Run, costs: dict) -> list[str]:
    seed, champ = run.seed, run.champion
    attempts = [s for i, s in run.steps.items() if i >= 0]
    kept = sum(1 for s in attempts if s.candidate and s.candidate.get("status") == "pooled")
    lines = [f"{len(attempts)} offspring tried, {kept} survived every check."]
    if champ and seed and champ.id != seed.id:
        lines.append(f"Champion {champ.id}: practice {_champ_score(run, seed.id)} → {_champ_score(run, champ.id)}"
                     + (f", unseen {_champ_score(run, seed.id, run.holdout)} → {_champ_score(run, champ.id, run.holdout)}"
                        if run.holdout and champ.holdout is not None else ""))
        lines.append(f"The change: “{_short(champ.summary, 140)}”")
    total = int(seed.metrics.get("total", 0)) if seed else 0
    if total and total <= 20:
        lines.append(f"Caution: with only {total} problems per set, a gain of one or two can be luck. "
                     "Re-test the champion before trusting it.")
    lines.append(f"Total cost about ${costs['total']:,.0f}"
                 + (f", plus ${costs['cached']:,.0f} of earlier results reused here." if costs["cached"] else "."))
    return lines


# ── section ──────────────────────────────────────────────────────────────────

WHY = """
<p class="lead"><b>Why evolve the harness?</b> tinyorbit is a small agent SDK: the harness that turns a model
into an agent. How well that agent does depends heavily on the harness artifacts: its system prompt, memory,
lessons, skills and tool descriptions. Today those are tuned by hand: change a prompt, rerun a benchmark, read
the failures, guess the next change, repeat. Evolve does that loop on its own. You give it a goal (a set of tasks
and a way to score them) and a budget, and it automatically evolves the artifacts until they meet the goal with a
high score. <span class="hl">No manual work is needed</span>: nobody hand-edits the harness, reads failures or
picks winners. The same mechanism extends to settings and any other file the harness reads.</p>
"""

INTRO = """
<p class="lead">In nature, evolution needs three things: <b>variation</b> (children differ slightly from their
parents), <b>selection</b> (the environment decides who survives) and <b>inheritance</b> (survivors pass their
traits on). Repeat that for enough generations and living things get remarkably good at surviving where they live,
without anyone designing them.</p>
<p class="lead"><b>Evolve</b> applies the same three ingredients to an agent harness. Its “DNA” is the set of
harness artifacts the model reads before it starts work. An AI editor makes small changes to those artifacts
(variation). Each new version is put to work on real programming problems and scored by the projects' own tests
(selection). The versions that do well become the parents of the next round (inheritance). The model is the same
throughout; only the harness evolves.</p>
"""

COMPARE = [
    ("DNA", "The genome: the harness artifacts (system prompt, memory, lessons, skills, tool descriptions), "
            "a folder of text files"),
    ("A mutation", "A small edit to those files, proposed by an AI editor"),
    ("Mutations are random", "Mutations are guided: the editor first reads which problems the parent failed, "
                             "and what earlier attempts already tried"),
    ("The environment", "Real programming problems from open-source projects"),
    ("Survival", "Fixing a problem so that the project's own tests pass"),
    ("Fitness", "Problems solved, minus a small penalty for growing longer"),
    ("Isolated islands (like Darwin's finches)", "Separate populations that evolve side by side, so one idea "
                                                 "does not take over too early"),
    ("Animals migrating between islands", "Every few generations, an island's best version is copied to the next"),
    ("Ecological niches", "Slots for different kinds of solutions (short or long harness, cheap or "
                          "thorough), so variety survives"),
    ("Surviving in a new climate", "Doing well on unseen problems, not just the practice set"),
    ("Millions of years", "A few hours, and about the price of a dinner out"),
]


def idea() -> str:
    rows = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>" for a, b in COMPARE)
    return (f'{WHY}{INTRO}<div class="cmpwrap sk-box"><table class="cmp"><tr><th>In nature</th><th>In evolve</th></tr>'
            f'{rows}</table></div>')


def player(run: Run, costs: dict) -> str:
    data = build(run, costs)
    # ensure_ascii=False keeps text as text, so the page-wide dash scrub in report.py reaches it too
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""
<div class="stage sk-box">
  <div class="narr" id="narr"></div>
  <div class="canvas"><svg class="tree sk" id="tree" viewBox="0 0 {data['width']} {data['height']}"></svg>
  <div class="lab" id="lab"></div></div>
  <div class="legend"><b style="background:#6b7280"></b>original<b style="background:#3b82f6"></b>survived
  <b style="background:#f59e0b"></b>failed a test<b style="background:#d4a017"></b>champion 👑
  <b style="background:#d1d5db"></b>discarded</div>
  <div class="controls"><button id="prev">‹ Back</button><button id="play" class="primary">Pause</button>
  <button id="next">Next ›</button><button id="speed">Speed 1×</button>
  <div class="bar"><i id="bar"></i></div><span class="count" id="count"></span></div>
</div>
<p class="muted small">The left and right arrow keys step through the scenes.</p>
<script type="application/json" id="data">{blob}</script>"""


CSS = """
.lead{font-size:17.5px;max-width:940px}
.cmpwrap{max-width:940px;margin:14px 0;padding:6px 8px;background:#fffef9}table.cmp{border-collapse:collapse;width:100%}
table.cmp th,table.cmp td{text-align:left;padding:7px 12px;border-bottom:1px dashed #c9c3b4;vertical-align:top}
table.cmp th{font-family:var(--hand);font-size:19px;color:#2f5d9e}
table.cmp td:first-child{width:36%;color:#4b5563;font-family:var(--hand);font-size:17px}
.stage{margin:14px 0 6px;background:#fffdf7}
.narr{padding:18px 24px 10px;min-height:122px}
.narr .cap{font-family:var(--hand);font-size:27px;margin:0;color:#1f2937}
.narr .sub{font-size:16.5px;color:#4b5563;margin:6px 0 0;max-width:980px}
.fade{animation:fade .5s ease both}@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.canvas{display:grid;grid-template-columns:640px 1fr;gap:0}
svg.tree{width:640px;height:490px;display:block}
.lane{fill:#f2f6fd}.lane.alt{fill:#f3f9f1}.lane-name{font-family:var(--hand);font-size:17px;fill:#6b7280}
.edge{fill:none;stroke:#8a94a3;stroke-width:1.6}.edge.hot{stroke:#c8960c;stroke-width:3}
.edge.draw{stroke-dasharray:400;stroke-dashoffset:400;animation:draw 1.2s .2s ease forwards}
@keyframes draw{to{stroke-dashoffset:0}}
.node circle.body{stroke:#2b2b2b;stroke-width:1.4}
.node.seed circle.body{fill:#6b7280}.node.pooled circle.body{fill:#3b82f6}.node.rejected circle.body{fill:#f59e0b}
.node.invalid circle.body{fill:#c4c4c4}.node.champion circle.body{fill:#d4a017}
.node text.id{font-family:var(--hand);font-size:15px;fill:#2b2b2b}.node text.sc{font-family:var(--hand);font-size:14px;fill:#6b7280}
.node.dim{opacity:.35}.node.pop{animation:pop .7s cubic-bezier(.3,1.6,.5,1) both;transform-box:fill-box;transform-origin:center}
@keyframes pop{from{transform:scale(0)}to{transform:scale(1)}}
.ring{fill:none;stroke:#2f5d9e;stroke-width:2;animation:spin 8s linear infinite;transform-box:fill-box;transform-origin:center}
@keyframes spin{to{transform:rotate(360deg)}}
.crown{font-size:20px}
.ghost circle{fill:none;stroke:#c8960c;stroke-width:2}.ghost text{font-family:var(--hand);font-size:14px;fill:#92400e}
.mig{stroke:#c8960c;stroke-width:1.8;fill:none;marker-end:url(#arr)}
.lab{padding:18px 22px;border-left:1.5px dashed #c9c3b4}
.lab h3{margin:0 0 12px;font-family:var(--hand);font-size:22px;font-weight:normal}
.row{margin:0 0 14px}.row .nm{font-size:14px;font-weight:600;color:#374151;display:flex;justify-content:space-between}
.row .nm span{font-weight:500;color:#6b7280}
.sq{display:inline-block;width:24px;height:24px;margin:5px 4px 0 0;background:#e5e7eb;border:1.3px solid #333;
border-radius:4px 7px 3px 6px/6px 3px 7px 4px}
.sq.ok{background:repeating-linear-gradient(-45deg,#10b981 0 3px,#a7f3d0 3px 5px)}
.sq.no{background:repeating-linear-gradient(45deg,#ef4444 0 3px,#fecaca 3px 5px)}
.sq.anim{animation:sq .35s ease both}@keyframes sq{from{transform:scale(.2);opacity:0}to{transform:none;opacity:1}}
.verdict{margin-top:12px;padding:8px 12px;font-family:var(--hand);font-size:19px}
.verdict.win{background:linear-gradient(transparent 20%,#fde68a 20%,#fde68a 85%,transparent 85%);color:#78350f}
.verdict.keep{color:#1e40af}.verdict.fail{color:#9a3412}.verdict.info{color:#374151}
.lab ul{padding-left:18px;margin:6px 0}.lab li{margin:7px 0}
.stage .legend{font-size:13px;color:#6b7280;padding:4px 22px 0;margin:0}
.stage .legend b{display:inline-block;width:11px;height:11px;border-radius:50%;margin:0 5px 0 12px;vertical-align:-1px;border:1px solid #333}
.controls{display:flex;align-items:center;gap:10px;padding:12px 22px}
button{font-family:var(--hand);font-size:17px;padding:4px 14px;border:1.5px solid #333;background:#fffdf7;cursor:pointer;
border-radius:255px 15px 225px 15px/15px 225px 15px 255px}
button.primary{background:#2b2b2b;color:#fffdf7}
.bar{flex:1;height:8px;border:1.3px solid #333;border-radius:6px 3px 7px 4px;overflow:hidden}
.bar i{display:block;height:100%;background:repeating-linear-gradient(-45deg,#d4a017 0 3px,#fde68a 3px 6px);width:0;transition:width .4s}
.count{font-family:var(--hand);font-size:16px;color:#6b7280;min-width:70px;text-align:right}
@media (max-width:1050px){.canvas{grid-template-columns:1fr}.lab{border-left:0;border-top:1.5px dashed #c9c3b4}}
"""

SCRIPT = r"""
(() => {
const D = JSON.parse(document.getElementById('data').textContent);
const S = D.scenes; let k = 0, playing = true, timer = null, speed = 1;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function path(a, b) { const mx = (a.x + b.x) / 2; return `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`; }
function lineage(id) { const out = new Set(); const byId = Object.fromEntries(D.nodes.map(n => [n.id, n]));
  while (id && byId[id] && !out.has(id)) { out.add(id); id = byId[id].parent; } return out; }
function tree(sc) {
  const shown = D.nodes.slice(0, sc.nodes), byId = Object.fromEntries(shown.map(n => [n.id, n]));
  const hot = lineage(sc.champion), focus = new Set(sc.focus); let g = '';
  D.lanes.forEach((l, i) => { g += `<rect class="lane ${i % 2 ? 'alt' : ''}" x="4" y="${l.y - l.h / 2 + 6}" width="${D.width - 8}" height="${l.h - 12}"/>`
    + `<text class="lane-name" x="${D.width - 16}" y="${l.y - l.h / 2 + 28}" text-anchor="end">${esc(l.name)}</text>`; });
  shown.forEach(n => { if (n.parent && byId[n.parent]) {
    const cls = 'edge' + (hot.has(n.id) && hot.has(n.parent) ? ' hot' : '') + (n.id === sc.new ? ' draw' : '');
    g += `<path class="${cls}" d="${path(byId[n.parent], n)}"/>`; } });
  sc.migrations.forEach(m => { const src = byId[m.id]; if (!src) return;
    g += `<path class="mig" d="M${src.x},${src.y + 22} L${m.x},${m.y - 26}"/>`
      + `<g class="ghost"><circle cx="${m.x}" cy="${m.y}" r="16"/><text x="${m.x}" y="${m.y + 33}" text-anchor="middle">${esc(m.id)}</text>`
      + `<text x="${m.x}" y="${m.y + 48}" text-anchor="middle">(migrant)</text></g>`; });
  shown.forEach(n => {
    const dim = sc.focus.length && !focus.has(n.id) && n.id !== sc.parent ? ' dim' : '';
    // gold only while it holds the title: a node is drawn as the champion of *this* scene
    const kind = n.id === sc.champion && n.kind !== 'seed' ? 'champion' : (n.kind === 'champion' ? 'pooled' : n.kind);
    const cls = `node ${kind}${dim}${n.id === sc.new ? ' pop' : ''}`;
    const r = n.kind === 'seed' ? 20 : 17;
    g += `<g class="${cls}"><title>${esc(n.id)}</title>`
      + (focus.has(n.id) ? `<circle class="ring" cx="${n.x}" cy="${n.y}" r="${r + 7}"/>` : '')
      + `<circle class="body" cx="${n.x}" cy="${n.y}" r="${r}"/>`
      + (n.kind === 'invalid' ? `<text x="${n.x}" y="${n.y + 6}" text-anchor="middle" font-size="18" fill="#2b2b2b">✕</text>` : '')
      + (n.id === sc.champion ? `<text class="crown" x="${n.x}" y="${n.y - r - 6}" text-anchor="middle">👑</text>` : '')
      + `<text class="id" x="${n.x}" y="${n.y + r + 18}" text-anchor="middle">${esc(n.label)}</text>`
      + (n.score ? `<text class="sc" x="${n.x}" y="${n.y + r + 34}" text-anchor="middle">${esc(n.score)}</text>` : '')
      + `</g>`; });
  $('tree').innerHTML = `<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">`
    + `<path d="M0,0 L10,5 L0,10 z" fill="#c8960c"/></marker></defs>` + g;
  window.sketch && window.sketch($('tree'));
}
function lab(sc) {
  const L = sc.lab; if (!L) { $('lab').innerHTML = `<p class="muted">The test lab is quiet. Watch the family tree.</p>`; return; }
  let h = `<h3>${esc(L.title)}</h3>`, delay = 0;
  (L.rows || []).forEach(r => {
    h += `<div class="row"><div class="nm">${esc(r.name)}<span>${esc(r.result)}</span></div>`;
    r.tasks.forEach(t => { const st = r.animate ? ` style="animation-delay:${(delay += 0.14 / speed).toFixed(2)}s"` : '';
      h += `<span class="sq ${t.ok ? 'ok' : 'no'}${r.animate ? ' anim' : ''}"${st} title="${esc(t.id)}"></span>`; });
    h += `</div>`; });
  if (L.summary) h += `<ul>${L.summary.map(s => `<li>${esc(s)}</li>`).join('')}</ul>`;
  if (L.verdict) h += `<div class="verdict ${L.tone} fade" style="animation-delay:${(delay + 0.3).toFixed(2)}s">${esc(L.verdict)}</div>`;
  $('lab').innerHTML = h;
}
function show(i) {
  k = Math.max(0, Math.min(S.length - 1, i)); const sc = S[k];
  $('narr').innerHTML = `<p class="cap fade">${esc(sc.caption)}</p>` + (sc.sub ? `<p class="sub fade" style="animation-delay:.15s">${esc(sc.sub)}</p>` : '');
  tree(sc); lab(sc);
  $('bar').style.width = `${(k + 1) / S.length * 100}%`; $('count').textContent = `${k + 1} / ${S.length}`;
  clearTimeout(timer); if (playing && k < S.length - 1) timer = setTimeout(() => show(k + 1), sc.dur / speed);
  if (k === S.length - 1) setPlaying(false);
}
function setPlaying(p) { playing = p; $('play').textContent = p ? 'Pause' : (k === S.length - 1 ? 'Replay' : 'Play');
  if (!p) clearTimeout(timer); }
$('play').onclick = () => { if (!playing && k === S.length - 1) { setPlaying(true); show(0); }
  else { setPlaying(!playing); if (playing) show(k); } };
$('prev').onclick = () => { setPlaying(false); show(k - 1); };
$('next').onclick = () => { setPlaying(false); show(k + 1); };
$('speed').onclick = () => { speed = speed === 1 ? 2 : speed === 2 ? 0.5 : 1; $('speed').textContent = `Speed ${speed}×`; };
document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === 'ArrowRight') $('next').click(); if (e.key === 'ArrowLeft') $('prev').click(); });
show(0);
})();
"""
