"""Run tinyorbit on SWE-bench Verified instances, one container per task.

    python run.py --smoke                 # one container, no API key: proves the plumbing
    python run.py                         # all tasks in tasks.json
    python run.py --only django__django-11734
    python run.py --dry-run               # print the docker commands, run nothing

Per task: start the instance image, copy tinyorbit in, install it with uv,
run `main.py --print` with the issue as the prompt, capture the transcript,
take `git diff` from /testbed, append a prediction line, remove the container.

Outputs (all under bench/):
    transcripts/<id>.out      agent stdout (streamed text + tool activity)
    transcripts/<id>.err      agent stderr (cost line, tracebacks)
    transcripts/<id>.messages.json   full message history + token/cost ledger
    patches/<id>.diff         the model_patch
    predictions.jsonl         one line per task, swebench format
    results.csv               one row per task: turns, exit reason, cost, seconds, diff size

Then score with:
    .venv/bin/swebench eval verified -p predictions.jsonl --run-id tinyorbit-5 -j 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TINYORBIT_SRC = HERE.parent                       # python-impl/
STAGE = HERE / ".stage" / "tinyorbit"             # copy without .venv, bench, caches
CONTAINER_APP = "/opt/tinyorbit"
CONTAINER_TRANSCRIPT = "/tmp/tinyorbit-transcript.json"   # full message history, copied out per task
CONTAINER_TRACE = "/tmp/tinyorbit-trace.jsonl"           # one JSON line per SSE event, when --trace
UV = "/root/.local/bin/uv"
PLATFORM = "linux/amd64"                          # SWE-bench images are x86_64
DEFAULT_TIMEOUT_MIN = 45
COST_RE = re.compile(r"(\d+) API calls .*?~\$([\d.]+)")
ENDED_RE = re.compile(r"\[turn ended: (\w+)\]")


def sh(cmd: list[str], *, dry: bool, check: bool = True, capture: bool = False, timeout: float | None = None, input_text: str | None = None) -> subprocess.CompletedProcess:
    shown = [_redact(c) for c in cmd]
    print("  $", " ".join(c if len(c) < 80 else c[:77] + "..." for c in shown), flush=True)
    if dry:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, check=check, text=True, capture_output=capture, timeout=timeout, input=input_text)


def _redact(arg: str) -> str:
    # never let a credential reach the run log
    return re.sub(r"(ANTHROPIC_API_KEY=)\S+", r"\1<redacted>", arg)


def stage_source() -> Path:
    """Copy python-impl into a clean directory so docker cp does not drag .venv along."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    ignore = shutil.ignore_patterns(".venv", "bench", "content", ".git", ".github", "__pycache__", ".pytest_cache", ".DS_Store", "notes.md", ".tinyorbit", "key.txt", ".env")
    shutil.copytree(TINYORBIT_SRC, STAGE, ignore=ignore)
    return STAGE


def render_prompt(task: dict) -> str:
    template = (HERE / "prompt.txt").read_text()
    return template.format(repo=task["repo"], problem_statement=task["problem_statement"])


def start_container(name: str, image: str, api_key: str | None, dry: bool) -> None:
    sh(["docker", "rm", "-f", name], dry=dry, check=False, capture=True)
    sh(["docker", "pull", "--platform", PLATFORM, image], dry=dry)
    env = ["-e", f"ANTHROPIC_API_KEY={api_key}"] if api_key else []
    sh(["docker", "run", "-d", "--name", name, "--platform", PLATFORM, *env, "-w", "/testbed", image, "sleep", "infinity"], dry=dry, capture=True)


def install_tinyorbit(name: str, dry: bool) -> None:
    sh(["docker", "cp", str(STAGE), f"{name}:{CONTAINER_APP}"], dry=dry)
    # uv installs its own Python 3.13 inside the container; the image's conda python is the repo's, not ours.
    sh(["docker", "exec", name, "sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], dry=dry, capture=True)
    sh(["docker", "exec", name, UV, "sync", "--project", CONTAINER_APP, "--no-group", "dev"], dry=dry, capture=True, timeout=900)
    # a clean baseline so the final diff is only what the agent did
    sh(["docker", "exec", name, "git", "-C", "/testbed", "status", "--porcelain"], dry=dry, capture=True)


def run_agent(name: str, prompt: str, args: argparse.Namespace, out_path: Path, err_path: Path) -> tuple[float, bool]:
    cmd = [
        "docker", "exec", "-w", "/testbed", "-e", f"TINYORBIT_TRANSCRIPT={CONTAINER_TRANSCRIPT}",
        *(["-e", f"TINYORBIT_TRACE={CONTAINER_TRACE}"] if args.trace else []), name,
        UV, "run", "--project", CONTAINER_APP, "python", f"{CONTAINER_APP}/main.py",
        "--print", "--permission-mode", "bypassPermissions",
        "--max-turns", str(args.max_turns), "--model", args.model,
    ]
    if args.effort:
        cmd += ["--effort", args.effort]
    if args.thinking_display != "omitted":
        cmd += ["--thinking-display", args.thinking_display]
    if not args.fallbacks:
        cmd += ["--no-fallbacks"]
    cmd.append(prompt)
    print("  $ docker exec ... main.py --print ... <prompt>", flush=True)
    if args.dry_run:
        return 0.0, False
    started = time.monotonic()
    timed_out = False
    with out_path.open("w") as out, err_path.open("w") as err:
        try:
            subprocess.run(cmd, stdout=out, stderr=err, text=True, timeout=args.timeout_min * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["docker", "exec", name, "pkill", "-f", "main.py"], capture_output=True)
    return time.monotonic() - started, timed_out


def collect_transcript(name: str, dest: Path, dry: bool) -> None:
    """Full message history + cost ledger; missing if the agent crashed before exit."""
    if dry:
        return
    subprocess.run(["docker", "cp", f"{name}:{CONTAINER_TRANSCRIPT}", str(dest)], capture_output=True)


def collect_patch(name: str, dry: bool) -> str:
    if dry:
        return ""
    # stage everything (new files included), then diff the index against HEAD
    subprocess.run(["docker", "exec", name, "git", "-C", "/testbed", "add", "-A"], check=True, capture_output=True)
    res = subprocess.run(["docker", "exec", name, "git", "-C", "/testbed", "diff", "--cached"], check=True, capture_output=True, text=True)
    return res.stdout


def summarize(err_text: str, out_text: str, timed_out: bool) -> tuple[int, str, str]:
    calls, cost = 0, "?"
    if m := COST_RE.search(err_text):
        calls, cost = int(m.group(1)), m.group(2)
    reason = "timeout" if timed_out else "completed"
    if m := ENDED_RE.findall(out_text):
        reason = m[-1]
    return calls, reason, cost


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[], help="instance id (repeatable)")
    ap.add_argument("--smoke", action="store_true", help="first task only, no API key, expect the no-credentials message")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--no-fallbacks", dest="fallbacks", action="store_false")
    ap.add_argument("--keep", action="store_true", help="leave containers running for inspection")
    ap.add_argument("--tasks", default=str(HERE / "tasks.json"), help="task file (default bench/tasks.json)")
    ap.add_argument("--timeout-min", type=int, default=DEFAULT_TIMEOUT_MIN, help="per-task wall-clock cap")
    ap.add_argument("--thinking-display", default="omitted", choices=["omitted", "summarized"])
    ap.add_argument("--trace", action="store_true", help="record every SSE event to transcripts/<id>.trace.jsonl")
    args = ap.parse_args()

    api_key = None if args.smoke else os.environ.get("ANTHROPIC_API_KEY")
    if not args.smoke and not args.dry_run and not api_key:
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    tasks = json.loads(Path(args.tasks).read_text())
    if args.only:
        tasks = [t for t in tasks if t["instance_id"] in args.only]
    if args.smoke:
        tasks = tasks[:1]

    for d in ("transcripts", "patches"):
        (HERE / d).mkdir(exist_ok=True)
    stage_source()
    results_path = HERE / "results.csv"
    new_csv = not results_path.exists()

    for task in tasks:
        iid = task["instance_id"]
        name = f"tinyorbit-{iid.lower()}"
        print(f"\n=== {iid}  ({task['difficulty']})  {task['image']}")
        out_path = HERE / "transcripts" / f"{iid}.out"
        err_path = HERE / "transcripts" / f"{iid}.err"
        try:
            start_container(name, task["image"], api_key, args.dry_run)
            install_tinyorbit(name, args.dry_run)
            seconds, timed_out = run_agent(name, render_prompt(task), args, out_path, err_path)
            collect_transcript(name, HERE / "transcripts" / f"{iid}.messages.json", args.dry_run)
            if args.trace and not args.dry_run:
                subprocess.run(["docker", "cp", f"{name}:{CONTAINER_TRACE}",
                                str(HERE / "transcripts" / f"{iid}.trace.jsonl")], capture_output=True)
            patch = collect_patch(name, args.dry_run)
        finally:
            if not args.keep:
                sh(["docker", "rm", "-f", name], dry=args.dry_run, check=False, capture=True)

        if args.dry_run:
            continue
        (HERE / "patches" / f"{iid}.diff").write_text(patch)
        out_text, err_text = out_path.read_text(), err_path.read_text()
        calls, reason, cost = summarize(err_text, out_text, timed_out)
        print(f"  -> {reason}; {calls} calls; ${cost}; {seconds/60:.1f} min; diff {len(patch)} chars")
        if args.smoke:
            ok = "no credentials found" in out_text
            print("  smoke:", "OK, plumbing works" if ok else "FAILED, see transcripts/")
            return 0 if ok else 1

        with (HERE / "predictions.jsonl").open("a") as fh:
            fh.write(json.dumps({"instance_id": iid, "model_name_or_path": "tinyorbit", "model_patch": patch}) + "\n")
        with results_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new_csv:
                w.writerow(["instance_id", "difficulty", "exit_reason", "api_calls", "cost_usd", "minutes", "diff_chars"])
                new_csv = False
            w.writerow([iid, task["difficulty"], reason, calls, cost, f"{seconds/60:.1f}", len(patch)])

    return 0


if __name__ == "__main__":
    sys.exit(main())
