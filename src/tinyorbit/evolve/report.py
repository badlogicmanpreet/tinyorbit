"""Report, the one page that replays an evolution run.

Reads what the loop leaves in a run directory (archive.json, ledger.jsonl,
events.jsonl, iterations/) and draws it for three readers at once: the why,
the analogy to nature and an animated replay for anyone (story.py), the score
chart, family tree and island grids for an architect, and every prompt, diff
and failure for whoever wants to audit a step.

It is styled as pencil notes on plain paper. rough.js (vendored, MIT) redraws every
SVG shape and box by hand after load; without it the page still reads, with
CSS borders (progressive enhancement). Everything is inlined, no network, so
the page opens anywhere, survives being emailed, and can be committed.

While a run is going the page refreshes itself and says which stage is
running; the CLI rewrites it after every iteration.
"""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from tinyorbit.evolve import story as story_section
from tinyorbit.evolve.archive import INVALID, POOLED, REJECTED, Archive, Candidate
from tinyorbit.evolve.events import EventLog, genome_diff
from tinyorbit.evolve.story import lane_name

REPORT = "report.html"
ROUGH_JS = Path(__file__).parent / "vendor" / "rough.js"
INLINE_LIMIT = 20_000          # chars of a diff or response embedded before "see file"

STAGE_NAMES = {"smoke": "easy safety check", "train": "practice set", "holdout": "unseen problems"}
PICK_NAMES = {
    "exploit": "one of this island's best versions (exploit)",
    "explore": "a random version from this island (explore)",
    "best": "the best version found anywhere so far",
}
COLORS = {"seed": "#6b7280", POOLED: "#2563eb", REJECTED: "#f59e0b", INVALID: "#9ca3af",
          "champion": "#d4a017", "holdout": "#7c3aed"}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def stage_name(stage: str) -> str:
    return STAGE_NAMES.get(stage, stage)


def pct(score: float | None) -> str:
    return "none" if score is None else f"{score * 100:.0f}%"


def money(usd: float | None) -> str:
    return "none" if usd is None else f"${usd:,.2f}"


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class Step:
    """Everything that happened in one iteration (or the seed, iteration -1)."""
    iteration: int
    start: dict = field(default_factory=dict)
    mutation: dict = field(default_factory=dict)
    stages: list[dict] = field(default_factory=list)
    candidate: dict = field(default_factory=dict)
    promotion: dict = field(default_factory=dict)
    invalid: dict = field(default_factory=dict)
    migration: list[dict] = field(default_factory=list)
    end: dict = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        if self.end.get("outcome"):
            return self.end["outcome"]
        if self.invalid:
            return INVALID
        return "running"


@dataclass
class Run:
    root: Path
    archive: Archive
    events: list[dict]
    steps: dict[int, Step]
    config: dict
    running: dict | None          # the stage_start still waiting for its stage_end
    finished: bool
    budget_stop: dict | None = None
    abort: dict | None = None

    @property
    def stages(self) -> list[str]:
        return [s["name"] for s in self.config.get("stages", [])] or ["train"]

    @property
    def final_stage(self) -> str:
        return self.stages[-1]

    @property
    def holdout(self) -> str | None:
        return self.config.get("holdout")

    @property
    def seed(self) -> Candidate | None:
        return next((c for c in self.archive.candidates.values() if c.parent is None), None)

    @property
    def champion(self) -> Candidate | None:
        return self.archive.candidates.get(self.archive.champion or "")


def load(root: Path) -> Run:
    root = Path(root)
    archive = Archive.open(root)
    events = EventLog(root).read()
    steps: dict[int, Step] = {}
    config: dict = {}
    running: dict | None = None
    finished = False
    budget_stop: dict | None = None
    abort: dict | None = None

    def step(it: int) -> Step:
        return steps.setdefault(it, Step(it))

    for ev in events:
        kind, it = ev["kind"], ev.get("iteration", -1)
        if kind == "run_start":
            config, finished = ev.get("config", {}), False
        elif kind == "run_end":
            finished, running = True, None
        elif kind == "budget_stop":
            budget_stop = ev
        elif kind == "run_abort":
            abort, running = ev, None
        elif kind == "stage_start":
            running = ev
        elif kind == "stage_end":
            running = None
            step(it).stages.append(ev)
        elif kind == "seed":
            step(-1).candidate = ev
            step(-1).end = {"outcome": "seed"}
        elif kind == "iteration_start":
            step(it).start = ev
        elif kind == "mutation":
            step(it).mutation = ev
        elif kind == "invalid":
            step(it).invalid = ev
        elif kind == "candidate":
            step(it).candidate = ev
        elif kind == "promotion":
            step(it).promotion = ev
        elif kind == "migration":
            step(it).migration = ev.get("moves", [])
        elif kind == "iteration_end":
            step(it).end = ev
    finished = finished and (budget_stop is not None or archive.iteration >= int(config.get("iterations", 0) or 0))
    return Run(root, archive, events, dict(sorted(steps.items())), config, running, finished, budget_stop,
               None if finished else abort)


def lineage(run: Run, cid: str | None) -> list[str]:
    path: list[str] = []
    while cid and cid in run.archive.candidates and cid not in path:
        path.append(cid)
        cid = run.archive.candidates[cid].parent
    return path[::-1]


def solved(stage_event: dict) -> str:
    m = stage_event.get("metrics") or {}
    if "resolved" in m and "total" in m:
        return f"{int(m['resolved'])} of {int(m['total'])}"
    return pct(stage_event.get("score"))


def cost_breakdown(run: Run) -> dict:
    by_stage: dict[str, float] = {}
    cached = 0.0
    for st in run.steps.values():
        for ev in st.stages:
            usd = float(ev.get("cost_usd") or 0.0)
            if (ev.get("info") or {}).get("cached"):
                cached += usd
                continue
            by_stage[ev["stage"]] = by_stage.get(ev["stage"], 0.0) + usd
    mutator = sum(float(st.mutation.get("cost_usd") or 0.0) for st in run.steps.values())
    return {"by_stage": by_stage, "mutator": mutator, "cached": cached,
            "total": sum(by_stage.values()) + mutator}


# ── story ────────────────────────────────────────────────────────────────────


def story(run: Run) -> list[str]:
    seed, champ = run.seed, run.champion
    final = stage_name(run.final_stage)
    out: list[str] = []
    if seed is None:
        return ["The run has started. The first measurement of the standard harness is still in progress."]
    seed_step = run.steps.get(-1, Step(-1))
    seed_final = next((e for e in seed_step.stages if e["stage"] == run.final_stage), {})
    out.append(
        f"We started from the standard tinyorbit harness, the artifacts that tell the model how to work "
        f"(version {seed.id}). On the {final} it solved {solved(seed_final) if seed_final else pct(seed.stage_scores.get(run.final_stage))} problems"
        + (f", and {pct(seed.holdout)} of the {stage_name(run.holdout)}." if seed.holdout is not None else ".")
    )
    attempts = [s for i, s in run.steps.items() if i >= 0]
    if not attempts:
        out.append("No changes have been tried yet.")
        return out
    counts = {k: sum(1 for s in attempts if s.outcome == k) for k in (INVALID, REJECTED, POOLED, "champion")}
    out.append(
        f"Then an AI editor proposed {len(attempts)} changes to those artifacts, one per attempt, each time "
        f"looking at which problems the version it started from got wrong."
    )
    parts = []
    if counts[INVALID]:
        parts.append(f"{counts[INVALID]} were thrown out before testing (broken or duplicate edits)")
    if counts[REJECTED]:
        parts.append(f"{counts[REJECTED]} failed the {stage_name(run.stages[0])}")
    kept = counts[POOLED] + counts["champion"]
    if kept:
        parts.append(f"{kept} passed every check and were kept as material for later attempts")
    if parts:
        out.append("Of those, " + "; ".join(parts) + ".")
    if champ is None or champ.id == seed.id:
        out.append("No change has yet done better than the original on both the practice set and the "
                   "unseen problems, so the original is still the champion.")
    else:
        steps = len(lineage(run, champ.id)) - 1
        out.append(
            f"The best version so far is {champ.id}, {steps} change{'s' if steps != 1 else ''} away from the original. "
            f"It scores {pct(champ.stage_scores.get(run.final_stage))} on the {final} "
            f"(the original: {pct(seed.stage_scores.get(run.final_stage))})"
            + (f" and {pct(champ.holdout)} on {stage_name(run.holdout)} it never trained on "
               f"(the original: {pct(seed.holdout)})." if champ.holdout is not None else ".")
        )
    costs = cost_breakdown(run)
    out.append(f"So far this has cost about {money(costs['total'])}.")
    if not run.finished:
        out.append("The run is still going (or was interrupted); this page updates after every attempt.")
    return out


# ── charts ───────────────────────────────────────────────────────────────────


def score_chart(run: Run) -> str:
    planned = int(run.config.get("iterations", 0) or 0)
    last = max([i for i in run.steps] + [planned - 1, 0])
    w, h, left, right, top, bottom = 820, 300, 56, 20, 20, 46
    pw, ph = w - left - right, h - top - bottom

    def x(i: float) -> float:
        return left + (i + 1) / (last + 1) * pw if last + 1 else left

    def y(score: float) -> float:
        return top + (1 - score) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart sk" role="img" aria-label="score per attempt">']
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y(tick)}" y2="{y(tick)}" class="grid"/>'
                     f'<text x="{left - 8}" y="{y(tick) + 4}" class="axis" text-anchor="end">{tick * 100:.0f}%</text>')
    step = max(1, (last + 2) // 12)
    for i in range(-1, last + 1, step):
        parts.append(f'<text x="{x(i)}" y="{h - bottom + 18}" class="axis" text-anchor="middle">'
                     f'{"start" if i < 0 else i}</text>')
    parts.append(f'<text x="{left + pw / 2}" y="{h - 6}" class="axis" text-anchor="middle">attempt</text>')

    # champion staircase: the final-stage score of whoever held the title after each step
    stair: list[tuple[float, float]] = []
    for i, st in run.steps.items():
        if i < 0 or st.promotion.get("promoted"):
            champ = run.archive.candidates.get(st.candidate.get("candidate", ""))
            if champ is not None:
                level = y(champ.stage_scores.get(run.final_stage, 0.0))
                if stair:
                    stair.append((x(i), stair[-1][1]))      # hold the old level, then step
                stair.append((x(i), level))
    if stair:
        stair.append((x(last), stair[-1][1]))
        parts.append(f'<polyline points="{" ".join(f"{px},{py}" for px, py in stair)}" class="stair"/>')

    for i, st in run.steps.items():
        cx = x(i)
        cand = st.candidate
        outcome = st.outcome
        if outcome == INVALID:
            parts.append(f'<g class="mark"><title>attempt {i}: thrown out, {esc(st.invalid.get("reason", ""))}</title>'
                         f'<path d="M{cx - 5},{y(0) - 5} l10,10 M{cx + 5},{y(0) - 5} l-10,10" stroke="{COLORS[INVALID]}" stroke-width="2"/></g>')
            continue
        if not cand:
            parts.append(f'<circle cx="{cx}" cy="{y(0)}" r="6" class="pending"><title>attempt {i}: running</title></circle>')
            continue
        scores = cand.get("stage_scores") or {}
        reached = run.final_stage if run.final_stage in scores else next(reversed(scores), run.final_stage)
        score = scores.get(reached, 0.0)
        color = COLORS.get(outcome, COLORS[POOLED])
        label = (f"{'start' if i < 0 else f'attempt {i}'}: {cand.get('candidate')}, {stage_name(reached)} {pct(score)}"
                 f", {esc(cand.get('summary', ''))}")
        r = 8 if outcome in ("champion", "seed") else 6
        # hollow = stopped before the final stage; its score there is a different yardstick
        fill = "white" if reached != run.final_stage else color
        parts.append(f'<g class="mark"><title>{label}</title><circle cx="{cx}" cy="{y(score)}" r="{r}" '
                     f'fill="{fill}" stroke="{color}" stroke-width="2"/></g>')
        hold = cand.get("holdout") if i < 0 else st.promotion.get("holdout")
        if hold is not None:
            hy = y(hold)
            parts.append(f'<g class="mark"><title>{cand.get("candidate")} on {stage_name(run.holdout or "")}: {pct(hold)}</title>'
                         f'<path d="M{cx},{hy - 7} l7,7 l-7,7 l-7,-7z" fill="none" stroke="{COLORS["holdout"]}" stroke-width="2"/></g>')
    parts.append("</svg>")
    legend = (
        f'<div class="legend"><span><i style="background:{COLORS["seed"]}"></i>original</span>'
        f'<span><i style="background:{COLORS[POOLED]}"></i>kept (passed every check)</span>'
        f'<span><i style="background:{COLORS["champion"]}"></i>new champion</span>'
        f'<span><i class="ring" style="border-color:{COLORS[REJECTED]}"></i>stopped at the {stage_name(run.stages[0])} (its score there)</span>'
        f'<span><b style="color:{COLORS[INVALID]}">✕</b> thrown out before testing</span>'
        f'<span><b style="color:{COLORS["holdout"]}">◇</b> score on {stage_name(run.holdout or "holdout")}</span>'
        f'<span><i class="line"></i>champion over time</span></div>'
    )
    return "".join(parts) + legend


def family_tree(run: Run) -> str:
    cands = sorted(run.archive.candidates.values(), key=lambda c: c.id)
    if not cands:
        return "<p class='muted'>Nothing yet.</p>"
    rows: dict[int, int] = {}
    pos: dict[str, tuple[float, float]] = {}
    colw, rowh, pad = 150, 74, 40
    for c in cands:
        row = rows.get(c.generation, 0)
        rows[c.generation] = row + 1
        pos[c.id] = (pad + c.generation * colw, pad + row * rowh)
    w = pad * 2 + max(rows) * colw + 60
    h = pad * 2 + (max(rows.values()) - 1) * rowh + 44
    best = set(lineage(run, run.archive.champion))
    parts = [f'<svg viewBox="0 0 {w} {h}" class="ftree sk" width="{w * 1.6:.0f}" height="{h * 1.6:.0f}" role="img" aria-label="family tree">']
    for g in sorted(rows):
        parts.append(f'<text x="{pad + g * colw}" y="16" class="axis" text-anchor="middle">'
                     f'{"original" if g == 0 else f"{g} change" + ("s" if g > 1 else "")}</text>')
    for c in cands:
        if c.parent in pos:
            (x1, y1), (x2, y2) = pos[c.parent], pos[c.id]
            hot = c.id in best and c.parent in best
            parts.append(f'<path d="M{x1 + 14},{y1} C{x1 + colw / 2},{y1} {x2 - colw / 2},{y2} {x2 - 14},{y2}" '
                         f'class="edge{" hot" if hot else ""}"/>')
    for c in cands:
        cx, cy = pos[c.id]
        kind = "seed" if c.parent is None else ("champion" if c.id == run.archive.champion else c.status)
        color = COLORS.get(kind, COLORS[POOLED])
        score = c.stage_scores.get(run.final_stage)
        sub = pct(score) if score is not None else f"stopped: {pct(next(iter(c.stage_scores.values()), None))}"
        tip = f"{c.id} ({c.status}) from {c.parent or '-'}: {c.summary}"
        ring = f'<circle cx="{cx}" cy="{cy}" r="18" fill="none" stroke="{COLORS["champion"]}" stroke-width="3"/>' \
            if c.id == run.archive.champion else ""
        parts.append(f'<g class="mark"><title>{esc(tip)}</title>{ring}<circle cx="{cx}" cy="{cy}" r="13" fill="{color}"/>'
                     f'<text x="{cx}" y="{cy + 30}" class="lbl" text-anchor="middle">{esc(c.id)}</text>'
                     f'<text x="{cx}" y="{cy + 45}" class="lbl muted" text-anchor="middle">{esc(sub)}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def island_grids(run: Run) -> str:
    a = run.archive
    bins = a.feature_bins
    out = ['<div class="islands">']
    for island in range(a.num_islands):
        cells = {k: v for k, v in a.grid.items() if k.startswith(f"{island}|")}
        rows = []
        for size_bin in range(bins):
            tds = []
            for cost_bin in range(bins):
                cid = cells.get(f"{island}|{size_bin}|{cost_bin}")
                if cid is None:
                    tds.append('<td class="empty"></td>')
                    continue
                c = a.candidates[cid]
                cls = "champ" if cid == a.champion else ("migrant" if c.island != island else "")
                tds.append(f'<td class="{cls}" title="{esc(c.summary)}"><b>{esc(cid)}</b><br>'
                           f'{pct(c.stage_scores.get(run.final_stage))}'
                           + (' <span class="tag">migrant</span>' if c.island != island else "") + "</td>")
            rows.append(f'<tr><th>{_bin_label(size_bin, bins, "size")}</th>{"".join(tds)}</tr>')
        head = "".join(f"<th>{_bin_label(b, bins, 'cost')}</th>" for b in range(bins))
        members = [c for c in a.candidates.values() if c.island == island]
        out.append(
            f'<div class="island sk-box"><h4>Island {lane_name(island)}</h4>'
            f'<p class="muted">{len(members)} versions born here, {len(cells)} of {bins * bins} niches filled'
            + (f", {len(a.migrants.get(island, []))} visitors from the neighbouring island" if a.migrants.get(island) else "")
            + f'</p><table class="grid"><tr><th class="corner">genome size ↓ / cost per problem →</th>{head}</tr>{"".join(rows)}</table></div>'
        )
    out.append("</div>")
    moves = [(i, m) for i, st in run.steps.items() for m in st.migration]
    if moves:
        out.append("<h4>Migrations</h4><ul>")
        for i, m in moves:
            out.append(f"<li>after attempt {i}: {esc(m['candidate'])} moved from island {lane_name(m['from'])} to island {lane_name(m['to'])}"
                       f" ({'took a niche' if m.get('placed') else 'joined as a parent option only'}"
                       + (f", replacing {esc(m['incumbent'])}" if m.get('placed') and m.get('incumbent') else "")
                       + ")</li>")
        out.append("</ul>")
    return "".join(out)


def _bin_label(b: int, bins: int, kind: str) -> str:
    # bins span 0..2x the seed's value (archive._bin); label each with its range
    lo, hi = b * 2 / bins, (b + 1) * 2 / bins
    word = "seed size" if kind == "size" else "seed cost"
    return f"{lo:.1f} to {hi:.1f}× {word}" if b < bins - 1 else f"≥{lo:.1f}× {word}"


# ── timeline ─────────────────────────────────────────────────────────────────


def _link(run: Run, out_path: Path, rel: str | None, label: str) -> str:
    if not rel:
        return ""
    href = os.path.relpath(run.root / rel, out_path.parent)
    return f'<a href="{esc(href)}">{esc(label)}</a>'


def _inline(run: Run, rel: str | None, summary: str, cls: str = "") -> str:
    if not rel or not (run.root / rel).is_file():
        return ""
    text = (run.root / rel).read_text()
    more = f"\n… {len(text) - INLINE_LIMIT:,} more chars in {rel}" if len(text) > INLINE_LIMIT else ""
    body = text[:INLINE_LIMIT] + more
    if cls == "diff":
        body_html = "".join(f'<span class="{_diff_cls(line)}">{esc(line)}</span>\n' for line in body.splitlines())
    else:
        body_html = esc(body)
    return f"<details><summary>{esc(summary)}</summary><pre class='{cls}'>{body_html}</pre></details>"


def _diff_cls(line: str) -> str:
    if line.startswith(("+++", "---")):
        return "dh"
    return {"+": "add", "-": "del", "@": "hunk"}.get(line[:1], "")


def _stage_chip(run: Run, ev: dict, out_path: Path) -> str:
    passed = ev.get("passed")
    cls = "pass" if passed else ("fail" if passed is False else "info")
    gate = f" (needed {pct(ev['min_score'])})" if ev.get("min_score") else ""
    info = ev.get("info") or {}
    cached = " · reused earlier result" if info.get("cached") else ""
    minutes = f" · {ev['seconds'] / 60:.0f} min" if ev.get("seconds") else ""
    run_dir = info.get("run_dir")
    link = f' · <a href="{esc(os.path.relpath(run_dir, out_path.parent))}">run files</a>' if run_dir else ""
    return (f'<span class="chip {cls}">{esc(stage_name(ev["stage"]))}: solved {esc(solved(ev))}{gate}'
            f' · {money(ev.get("cost_usd"))}{minutes}{cached}{link}</span>')


def _failures(ev_list: list[dict]) -> str:
    rows = [f"<li><b>{esc(stage_name(ev['stage']))}</b> · {esc(f['task'])}: {esc(f['summary'])}</li>"
            for ev in ev_list for f in ev.get("failures") or []]
    if not rows:
        return ""
    return f"<details><summary>problems it got wrong ({len(rows)})</summary><ul class='small'>{''.join(rows)}</ul></details>"


def _tasks_table(ev_list: list[dict]) -> str:
    rows = []
    for ev in ev_list:
        tasks = (ev.get("info") or {}).get("tasks") or {}
        cells = "".join(f'<span class="task {esc(status)}" title="{esc(iid)}: {esc(status)}"></span>'
                        for iid, status in tasks.items())
        if cells:
            rows.append(f"<div class='tasks'><span class='small'>{esc(stage_name(ev['stage']))}</span> {cells}</div>")
    return "".join(rows)


OUTCOME_WORDS = {
    INVALID: ("thrown out", "The edit could not be applied or broke a rule, so it was never tested."),
    REJECTED: ("failed a check", "It did not reach the score needed to go on, so it was set aside."),
    POOLED: ("kept", "It passed every check and joins the pool future attempts can start from."),
    "champion": ("new champion", "It beat the champion and was confirmed on problems it had never seen."),
    "running": ("running", "This attempt is still being measured."),
    "seed": ("starting point", "The standard harness, measured on every set to be the yardstick."),
}


def timeline(run: Run, out_path: Path) -> str:
    out = []
    for i, st in sorted(run.steps.items(), reverse=True):
        word, explain = OUTCOME_WORDS.get(st.outcome, (st.outcome, ""))
        cand = st.candidate.get("candidate", "")
        title = "The starting point" if i < 0 else f"Attempt {i}"
        head = (f'<div class="step-head"><span class="badge {esc(st.outcome)}">{esc(word)}</span>'
                f'<b>{title}</b>' + (f' → <code>{esc(cand)}</code>' if cand else "") + "</div>")
        lines = []
        if st.start:
            lines.append(f"Island {lane_name(st.start['island'])}. Started from <code>{esc(st.start['parent'])}</code>, "
                         f"picked as {esc(PICK_NAMES.get(st.start.get('pick', ''), st.start.get('pick', '?')))}."
                         + (f" Also shown for inspiration: {', '.join(f'<code>{esc(x)}</code>' for x in st.start['inspirations'])}."
                            if st.start.get("inspirations") else ""))
        if st.mutation:
            summary = st.mutation.get("summary") or "(no summary)"
            edits = st.mutation.get("edits") or []
            files = sorted({e["path"] for e in edits})
            lines.append(f"<span class='idea'>Idea: {esc(summary)}</span>")
            if edits:
                lines.append(f"{len(edits)} edit{'s' if len(edits) != 1 else ''} to {', '.join(f'<code>{esc(f)}</code>' for f in files)}"
                             f" (+{sum(e['added_chars'] for e in edits):,} / −{sum(e['removed_chars'] for e in edits):,} chars)"
                             + (f"; editor cost {money(st.mutation.get('cost_usd'))}" if st.mutation.get("cost_usd") is not None else ""))
        if st.invalid:
            lines.append(f"<span class='why'>Why thrown out: {esc(st.invalid.get('reason', ''))}</span>")
        if st.stages:
            lines.append(" ".join(_stage_chip(run, ev, out_path) for ev in st.stages))
        if st.candidate and i >= 0:
            c = st.candidate
            if c.get("fitness") is not None:
                lines.append(f"Fitness {c['fitness']:.3f} = score − size penalty {c.get('penalty') or 0:.3f} "
                             f"(genome {c.get('size_delta', 0):+,} chars vs parent, {c.get('growth_over_seed', 0):+,} vs original).")
            place = c.get("place") or {}
            if place:
                isl, sb, cb = place["cell"].split("|")
                lines.append(f"Niche size {sb} / cost {cb} on island {lane_name(int(isl))}: "
                             + ("took it" + (f" from <code>{esc(place['incumbent'])}</code>" if place.get("incumbent") else " (was empty)")
                                if place.get("placed") else f"kept by <code>{esc(place.get('incumbent'))}</code>, which is better there"))
        if st.promotion:
            lines.append(("<span class='win'>Promoted: " if st.promotion.get("promoted") else "Not promoted: ")
                         + esc(st.promotion.get("reason", "")) + ("</span>" if st.promotion.get("promoted") else ""))
        for m in st.migration:
            lines.append(f"Migration after this attempt: <code>{esc(m['candidate'])}</code> island {lane_name(m['from'])} → {lane_name(m['to'])}.")
        arts = st.mutation.get("artifacts") or {}
        links = " · ".join(x for x in (_link(run, out_path, arts.get("prompt"), "what the editor was shown"),
                                       _link(run, out_path, arts.get("response"), "its full answer"),
                                       _link(run, out_path, arts.get("diff"), "diff file")) if x)
        extra = (_inline(run, arts.get("diff"), "show the change (diff)", "diff")
                 + _failures(st.stages) + _tasks_table(st.stages)
                 + _inline(run, arts.get("response"), "show the editor's answer"))
        out.append(f'<div class="step {esc(st.outcome)}">{head}<p class="muted">{esc(explain)}</p>'
                   + "".join(f"<p>{line}</p>" for line in lines)
                   + (f"<p class='small'>{links}</p>" if links else "") + extra + "</div>")
    return "".join(out)


# ── page ─────────────────────────────────────────────────────────────────────


def champion_diff(run: Run) -> str:
    seed, champ = run.seed, run.champion
    if seed is None or champ is None or champ.id == seed.id:
        return "<p class='muted'>The champion is still the original harness, so there is nothing to compare.</p>"
    diff = genome_diff(run.archive.genome(seed.id), run.archive.genome(champ.id))
    path = " → ".join(f"<code>{esc(c)}</code>" for c in lineage(run, champ.id))
    steps = []
    for cid in lineage(run, champ.id)[1:]:
        c = run.archive.candidates[cid]
        steps.append(f"<li><code>{esc(cid)}</code>: {esc(c.summary)}</li>")
    body = "".join(f'<span class="{_diff_cls(line)}">{esc(line)}</span>\n' for line in diff[:INLINE_LIMIT * 3].splitlines())
    return (f"<p>Path: {path}</p><ol>{''.join(steps)}</ol>"
            f"<details open><summary>Everything that changed from the original to {esc(champ.id)}</summary>"
            f"<pre class='diff'>{body}</pre></details>")


def cards(run: Run) -> str:
    seed, champ = run.seed, run.champion
    attempts = sum(1 for i in run.steps if i >= 0)
    planned = run.config.get("iterations", "?")
    costs = cost_breakdown(run)
    items = [
        ("attempts", f"{attempts} / {planned}"),
        (f"original on {stage_name(run.final_stage)}", pct(seed.stage_scores.get(run.final_stage)) if seed else "none"),
        (f"champion on {stage_name(run.final_stage)}", pct(champ.stage_scores.get(run.final_stage)) if champ else "none"),
    ]
    if run.holdout:
        items += [(f"original on {stage_name(run.holdout)}", pct(seed.holdout) if seed else "none"),
                  (f"champion on {stage_name(run.holdout)}", pct(champ.holdout) if champ else "none")]
    budget = run.config.get("budget_usd")
    items += [("champion", champ.id if champ else "none"),
              ("spent so far" + (f" (cap {money(budget)})" if budget else ""), money(costs["total"]))]
    return '<div class="cards">' + "".join(
        f'<div class="card sk-box"><div class="num">{esc(v)}</div><div class="muted">{esc(k)}</div></div>' for k, v in items
    ) + "</div>"


def status_line(run: Run) -> str:
    if run.finished and run.budget_stop:
        b = run.budget_stop
        return (f'<div class="status done sk-box">Finished early to stay within budget: spent {money(b["spent_usd"])} '
                f'of {money(b["budget_usd"])}, and one more attempt could have cost up to {money(b["reserve_usd"])}.</div>')
    if run.finished:
        return ""                  # a finished run needs no banner; only live or stopped runs get one
    if run.abort:
        return (f'<div class="status live sk-box">Stopped at attempt {run.abort["iteration"]}: {esc(run.abort["reason"])}. '
                'Rerun the same command to resume.</div>')
    last_ts = run.events[-1]["ts"] if run.events else "never"
    if run.running:
        r = run.running
        who = "the original" if r.get("iteration", -1) < 0 else f"attempt {r['iteration']} ({r.get('candidate', '')})"
        return (f'<div class="status live sk-box">Running now: <b>{esc(stage_name(r["stage"]))}</b> for {esc(who)}, '
                f'started {esc(r["ts"])}. This page refreshes every minute.</div>')
    return f'<div class="status live sk-box">In progress or interrupted. Last update {esc(last_ts)}.</div>'


def costs_table(run: Run) -> str:
    c = cost_breakdown(run)
    rows = "".join(f"<tr><td>running tinyorbit on the {esc(stage_name(s))}</td><td>{money(v)}</td></tr>"
                   for s, v in c["by_stage"].items())
    rows += f"<tr><td>the AI editor proposing changes</td><td>{money(c['mutator'])}</td></tr>"
    rows += f"<tr class='total'><td>total</td><td>{money(c['total'])}</td></tr>"
    if c["cached"]:
        rows += f"<tr><td class='muted'>reused results (already paid earlier, not counted)</td><td class='muted'>{money(c['cached'])}</td></tr>"
    return f"<table class='costs'>{rows}</table>"


def conclusion(run: Run) -> list[str]:
    """Four or five plain sentences, computed from the run so they fit any run."""
    seed, champ = run.seed, run.champion
    if seed is None:
        return ["The run has not measured the original harness yet, so there is nothing to conclude."]
    score = story_section._champ_score
    attempts = sum(1 for i in run.steps if i >= 0)
    costs = cost_breakdown(run)
    out: list[str] = []
    if champ is None or champ.id == seed.id:
        out.append(f"In {attempts} attempts, no change beat the original harness on both the practice set and the "
                   "unseen problems, so the original stays in place.")
        out.append("That is a useful result: it says the current harness is hard to improve with small edits, "
                   "and it cost nothing in manual effort to find out.")
    else:
        holdout = (f" and from {score(run, seed.id, run.holdout)} to {score(run, champ.id, run.holdout)} on "
                   "unseen problems" if run.holdout and champ.holdout is not None else "")
        out.append(f"In {attempts} attempts, evolve moved the harness from {score(run, seed.id)} to "
                   f"{score(run, champ.id)} on the practice set{holdout}, with nobody editing a file by hand.")
        out.append(f"The winning change, {champ.id}: {champ.summary}")
    out.append(f"The whole search cost about {money(costs['total'])}, every step is logged, and every "
               "version can be inspected or rolled back.")
    total = int(seed.metrics.get("total", 0))
    if total and total <= 20:
        out.append(f"With {total} problems per set, a gain of one or two problems can be luck, so the champion "
                   "should be re-tested on a fresh set before it ships.")
    out.append("Next: confirm the champion, run more generations, and let evolve reach further artifacts such "
               "as skills and settings.")
    return out


REFERENCES = [
    ("AlphaEvolve: A coding agent for scientific and algorithmic discovery",
     "Novikov et al., Google DeepMind, 2025. The design evolve follows: an LLM proposes diffs, an evaluator "
     "scores them, and an evolutionary database of programs keeps the search going.",
     "https://arxiv.org/abs/2506.13131"),
    ("AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms",
     "Google DeepMind blog post introducing AlphaEvolve, May 2025.",
     "https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/"),
    ("OpenEvolve",
     "An open-source implementation of AlphaEvolve. Its island populations, MAP-Elites grid and "
     "cascade evaluation are the patterns evolve borrows; here the evolved program is the agent harness.",
     "https://github.com/algorithmicsuperintelligence/openevolve"),
]


def references() -> str:
    items = "".join(f'<li><a href="{esc(url)}">{esc(title)}</a><br><span class="muted">{esc(note)}</span></li>'
                    for title, note, url in REFERENCES)
    return f'<ol class="refs">{items}</ol>'


CSS = """
:root{--hand:"Bradley Hand","Segoe Print","Chalkboard SE","Comic Sans MS",cursive;--ink:#2b2b2b;--paper:#fdfbf3}
*{box-sizing:border-box}
body{margin:0;font:15.5px/1.6 "Avenir Next","Segoe UI",-apple-system,Roboto,sans-serif;color:#2b2b2b;
background:var(--paper)}
main{max-width:1120px;margin:0 auto;padding:30px 30px 80px}
h1,h2,h3,h4,.hand{font-family:var(--hand);font-weight:normal}
h1{font-size:42px;margin:0;line-height:1.2}
.kicker{font-family:var(--hand);font-size:18px;color:#2f5d9e}
h2{font-size:31px;margin:46px 0 8px;display:inline-block;padding:0 8px 0 4px;
background:linear-gradient(transparent 58%,rgba(253,224,71,.6) 58%,rgba(253,224,71,.6) 92%,transparent 92%)}
h2 .n{color:#2f5d9e;margin-right:8px}
h4{font-size:21px;margin:4px 0}
.hl{background:linear-gradient(transparent 55%,rgba(253,224,71,.65) 55%)}
.muted{color:#6b6b6b}.small{font-size:13.5px}
code{font:12.5px "SF Mono",Menlo,Consolas,monospace;background:rgba(0,0,0,.05);padding:0 4px;border-radius:3px}
.sk-box{position:relative;border:1.5px solid #555;border-radius:255px 14px 225px 14px/14px 225px 14px 255px}
.rough .sk-box{border-color:transparent}
svg.sk-frame{position:absolute;left:-5.5px;top:-5.5px;pointer-events:none;overflow:visible}
.status{padding:10px 16px;margin:16px 0;font-family:var(--hand);font-size:19px;max-width:760px;transform:rotate(-.4deg)}
.status.live{background:#fff7c2}.status.done{background:#e6f6ea}
.story p{font-size:17px;margin:8px 0;max-width:940px}
.cards{display:flex;flex-wrap:wrap;gap:16px;margin:18px 0}
.card{background:#fffef9;padding:10px 16px;min-width:138px}
.card:nth-child(odd){transform:rotate(-.7deg)}.card:nth-child(even){transform:rotate(.6deg)}
.card .num{font-family:var(--hand);font-size:30px;color:#2f5d9e;line-height:1.2}.card .muted{font-size:13px}
svg text{font-family:var(--hand)}
svg text.axis{font-size:14px;fill:#6b6b6b}svg line.grid{stroke:#d7dbe0;stroke-width:.8}
svg .stair{fill:none;stroke:#c8960c;stroke-width:2.5}
svg .pending{fill:none;stroke:#2f5d9e;stroke-width:1.5}
svg .edge{fill:none;stroke:#8a94a3;stroke-width:1.4}svg .edge.hot{stroke:#c8960c;stroke-width:3}
svg text.lbl{font-size:14px;fill:#2b2b2b}svg text.lbl.muted{fill:#7a7a7a}
.chart{width:100%;height:auto;display:block}.treewrap{overflow-x:auto;padding:6px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13.5px;color:#4b4b4b;margin:6px 0 0 56px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px;border:1px solid #333}
.legend i.ring{background:white;border:2px solid;width:9px;height:9px}
.legend i.line{border-radius:0;height:3px;width:16px;background:#c8960c;border:0;vertical-align:3px}
.islands{display:flex;flex-wrap:wrap;gap:28px}.island{padding:10px 16px 14px;background:#fffef9}
table.grid{border-collapse:collapse;font-size:12.5px}
table.grid td,table.grid th{border:1px dashed #b9b3a5;padding:6px;text-align:center;min-width:76px}
table.grid th{font-weight:500;color:#6b6b6b}table.grid td.empty{background:transparent}
table.grid td.champ{background:rgba(253,224,71,.45)}table.grid td.migrant{background:rgba(196,181,253,.35)}
th.corner{font-size:11.5px}.tag{font-size:11px;color:#6d28d9;font-family:var(--hand)}
.step{border-left:2.5px solid #c9c3b4;padding:6px 16px;margin:18px 0}.step p{margin:4px 0}
.step.champion{border-color:#d4a017}.step.pooled{border-color:#3b82f6}.step.rejected{border-color:#f59e0b}
.step.invalid{border-color:#c4c4c4}.step.seed{border-color:#6b7280}.step.running{border-color:#93c5fd}
.step-head{display:flex;gap:12px;align-items:center;font-size:16px}
.step-head b{font-family:var(--hand);font-weight:normal;font-size:23px}
.badge{font-family:var(--hand);font-size:16px;padding:0 10px;color:#6b7280;border:1.5px solid currentColor;
border-radius:255px 15px 225px 15px/15px 225px 15px 255px;transform:rotate(-2deg)}
.badge.champion{color:#a16207;background:rgba(253,224,71,.4)}.badge.pooled{color:#1d4ed8}.badge.rejected{color:#c2410c}
.badge.invalid{color:#7a7a7a}.badge.running{color:#2563eb}
.idea{font-family:var(--hand);font-size:18px;color:#1e3a8a}.why{color:#b45309}.win{color:#92400e;font-weight:600}
.chip{display:inline-block;font-size:12.5px;padding:2px 9px;margin:3px 5px 3px 0;border:1.3px solid;
border-radius:255px 12px 225px 12px/12px 225px 12px 255px}
.chip.pass{border-color:#059669;background:rgba(16,185,129,.08)}.chip.fail{border-color:#d97706;background:rgba(245,158,11,.08)}
.chip.info{border-color:#9ca3af}
.tasks{margin:3px 0}.task{display:inline-block;width:13px;height:13px;margin-right:3px;vertical-align:-2px;background:#e5e7eb;
border:1px solid #333;border-radius:2px 4px 2px 3px}
.task.resolved{background:repeating-linear-gradient(-45deg,#10b981 0 2px,#a7f3d0 2px 4px)}
.task.unresolved,.task.empty_patch{background:repeating-linear-gradient(45deg,#ef4444 0 2px,#fecaca 2px 4px)}
details{margin:6px 0}summary{cursor:pointer;color:#2f5d9e;font-family:var(--hand);font-size:17px}
a{color:#2f5d9e}
pre{background:rgba(255,255,255,.75);border:1px dashed #b9b3a5;padding:10px;overflow:auto;max-height:520px;
font:12px/1.5 "SF Mono",Menlo,Consolas,monospace;white-space:pre-wrap}
pre.diff .add{color:#047857;background:#ecfdf5}pre.diff .del{color:#b91c1c;background:#fef2f2}
pre.diff .hunk{color:#7c3aed}pre.diff .dh{color:#6b7280;font-weight:600}
table.costs{border-collapse:collapse}table.costs td{padding:3px 22px 3px 0}
.refs li{margin:8px 0}
table.costs tr.total td{font-family:var(--hand);font-size:20px;border-top:1.5px solid #555}
""" + story_section.CSS

# The pencil pass (progressive enhancement): rough.js redraws each SVG shape and
# each .sk-box frame by hand. Seeds come from the shape itself, so a redraw (the
# animation re-renders its tree every scene) wobbles the same way and does not jitter.
SKETCH = r"""
(() => {
if (typeof rough === 'undefined') return;
document.documentElement.classList.add('rough');
const hash = s => { let h = 7; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0; return (h >>> 0) % 2147483646 + 1; };
const light = c => { const m = (c.match(/[\d.]+/g) || []).map(Number); return m.length < 3 || 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2] > 225; };
const A = (el, n) => parseFloat(el.getAttribute(n) || 0);
window.sketch = svg => {
  const rc = rough.svg(svg);
  svg.querySelectorAll('circle,ellipse,rect,line,path,polyline,polygon').forEach(el => {
    if (el.closest('defs,marker,.plain')) return;
    const cs = getComputedStyle(el);
    const o = {seed: hash(el.outerHTML), roughness: 1.1, bowing: 1, stroke: cs.stroke === 'none' ? 'none' : cs.stroke,
               strokeWidth: parseFloat(cs.strokeWidth) || 1};
    if (cs.fill && cs.fill !== 'none' && !cs.fill.startsWith('rgba(0, 0, 0, 0')) {
      let box = {width: 0, height: 0}; try { box = el.getBBox(); } catch (e) { /* not rendered yet */ }
      const solid = light(cs.fill) || Math.min(box.width, box.height) < 14;
      Object.assign(o, {fill: cs.fill, fillStyle: solid ? 'solid' : 'hachure', hachureGap: 3.2, fillWeight: 1.4, hachureAngle: -41});
    }
    const t = el.tagName.toLowerCase(); let node;
    if (t === 'circle') node = rc.circle(A(el, 'cx'), A(el, 'cy'), A(el, 'r') * 2, o);
    else if (t === 'ellipse') node = rc.ellipse(A(el, 'cx'), A(el, 'cy'), A(el, 'rx') * 2, A(el, 'ry') * 2, o);
    else if (t === 'rect') node = rc.rectangle(A(el, 'x'), A(el, 'y'), A(el, 'width'), A(el, 'height'), o);
    else if (t === 'line') node = rc.line(A(el, 'x1'), A(el, 'y1'), A(el, 'x2'), A(el, 'y2'), o);
    else if (t === 'path') node = rc.path(el.getAttribute('d'), o);
    else { const pts = el.getAttribute('points').trim().split(/\s+/).map(p => p.split(',').map(Number));
           node = t === 'polygon' ? rc.polygon(pts, o) : rc.linearPath(pts, o); }
    if (el.getAttribute('class')) node.setAttribute('class', el.getAttribute('class'));   // keeps animation hooks
    if (cs.markerEnd && cs.markerEnd !== 'none' && node.lastElementChild) node.lastElementChild.setAttribute('marker-end', cs.markerEnd);
    while (el.firstChild) node.appendChild(el.firstChild);                                  // keeps <title> tooltips
    el.replaceWith(node);
  });
};
document.querySelectorAll('svg.sk').forEach(window.sketch);
const frame = el => {
  let s = el.querySelector(':scope > svg.sk-frame');
  if (!s) { s = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); s.setAttribute('class', 'sk-frame'); el.prepend(s); }
  const w = el.offsetWidth, h = el.offsetHeight;
  s.setAttribute('width', w + 8); s.setAttribute('height', h + 8); s.innerHTML = '';
  s.appendChild(rough.svg(s).rectangle(4, 4, w - 3, h - 3, {seed: hash(el.className + el.textContent.slice(0, 40)),
    roughness: 1.4, bowing: 0.7, stroke: '#3a3a3a', strokeWidth: 1.3}));
};
const ro = new ResizeObserver(es => es.forEach(e => frame(e.target)));
document.querySelectorAll('.sk-box').forEach(el => ro.observe(el));
})();
"""

_PRE = re.compile(r"(<pre\b.*?</pre>)", re.S)
_DASH = re.compile(r"\s*[—–]\s*")


def no_dashes(page: str) -> str:
    """House style: no em or en dashes, a comma instead. Run summaries and reasons are
    written by a model and often contain them. Quoted files (<pre>: diffs, the editor's
    answers) stay verbatim, since a report must not misquote what changed."""
    return "".join(part if part.startswith("<pre") else _DASH.sub(", ", part) for part in _PRE.split(page))


def render(root: Path, out_path: Path | None = None) -> str:
    run = load(root)
    out_path = Path(out_path or Path(root) / REPORT).resolve()
    run.root = run.root.resolve()
    refresh = "" if run.finished else '<meta http-equiv="refresh" content="60">'
    cfg = run.config
    costs = cost_breakdown(run)
    stages = ", ".join(f"{stage_name(s['name'])} ({s['name']}" + (f", needs {pct(s['min_score'])}" if s.get("min_score") else "") + ")"
                       for s in cfg.get("stages", []))
    tech = json.dumps(cfg, indent=1)

    def h2(n: int, title: str) -> str:
        return f'<div><h2><span class="n">{n}.</span>{esc(title)}</h2></div>'

    body = f"""
<div class="kicker">field notes</div>
<h1>How the tinyorbit harness Self Evolved</h1>
<div class="muted">tinyorbit is a small agent SDK, a harness around a model. This page replays one evolution run
from its {len(run.events)} logged events.</div>
{status_line(run)}
{h2(1, "The idea")}
{story_section.idea()}
{h2(2, "Watch it happen")}
<p class="muted">An animated replay of this run. It plays on its own; use Back and Next to step.</p>
{story_section.player(run, costs)}
{h2(3, "The story in one minute")}
<section class="story">{''.join(f'<p>{esc(s)}</p>' for s in story(run))}</section>
{cards(run)}
{h2(4, "Score after each attempt")}
<p class="muted">Each dot is one attempt, scored on the {esc(stage_name(run.final_stage))}. The gold line is the
champion at that moment. Purple diamonds show the score on {esc(stage_name(run.holdout or "holdout"))}, measured only for
contenders. Hover a dot for details.</p>
{score_chart(run)}
{h2(5, "Family tree")}
<p class="muted">Every version that was measured, left to right by how many changes it is from the original.
The gold path leads to the current champion. Hover a node for its idea.</p>
<div class="treewrap">{family_tree(run)}</div>
{h2(6, "Islands and niches")}
<p class="muted">The search runs on {run.archive.num_islands} islands in turn. Each island keeps the best version for every
mix of harness size and cost per problem, so different styles survive instead of one idea taking over.
Gold: the champion. Purple: a visitor from the other island.</p>
{island_grids(run)}
{h2(7, "What changed in the champion")}
{champion_diff(run)}
{h2(8, "Every attempt, newest first")}
<p class="muted">Green squares are problems solved, red are problems missed.</p>
{timeline(run, out_path)}
{h2(9, "Cost")}
{costs_table(run)}
{h2(10, "Conclusion")}
<section class="story">{''.join(f'<p>{esc(s)}</p>' for s in conclusion(run))}</section>
{h2(11, "Technical appendix")}
<p>Stages: {esc(stages) or 'none'}; promotion gate: {esc(stage_name(cfg.get('holdout') or '') or 'none')}
(margin {cfg.get('holdout_margin', 0)}); size penalty {cfg.get('size_penalty', '?')} per 1,000 chars;
{cfg.get('num_islands', '?')} islands, migration every {cfg.get('migration_interval', '?')} attempts;
{cfg.get('feature_bins', '?')}×{cfg.get('feature_bins', '?')} niches per island.</p>
<p>Files: <code>archive.json</code> (state), <code>ledger.jsonl</code> (one line per attempt),
<code>events.jsonl</code> (every step), <code>iterations/NNNN/</code> (editor prompt, answer, diff),
<code>genomes/</code> (every version), <code>champion/</code>.</p>
<details><summary>run configuration</summary><pre>{esc(tech)}</pre></details>
{h2(12, "References")}
{references()}
"""
    rough_js = ROUGH_JS.read_text() if ROUGH_JS.is_file() else ""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">{refresh}'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Evolution report, {esc(run.root.name)}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main>"
            f"<script>{rough_js}</script><script>{SKETCH}</script><script>{story_section.SCRIPT}</script>"
            f"</body></html>\n")
    return no_dashes(page)


def write_report(root: Path, out_path: Path | None = None) -> Path:
    out_path = Path(out_path or Path(root) / REPORT)
    text = render(root, out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(out_path)          # atomic, so a browser refresh never sees half a page
    return out_path
