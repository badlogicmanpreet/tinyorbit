# Reports

Two self-contained HTML pages and the data they embed. Both open straight from disk;
neither needs a server, a build step, or network access beyond web fonts.

| File | What it is |
|---|---|
| `swe-bench-report.html` | Results of two live SWE-bench Verified runs: ten sampled instances (10/10 resolved) and ten from the hard tail (7/10), with per-task verdicts, harness behaviour, and a task-level comparison against 19 published leaderboard submissions. |
| `agent-loop-trace.html` | One task traced at the wire. Animated replay of all 19 model calls: SSE event timing, adaptive-thinking pauses, prompt-cache behaviour, and the history resent each turn. |

Published copies: `swe-bench-report.html` is artifact `45be02c9`, `agent-loop-trace.html` is `295f5a6c`.

## Data

| File | Feeds | Notes |
|---|---|---|
| `report-data.json` | `swe-bench-report.html` | Per-task results, token ledgers, tool counts and scoring verdicts for both runs. |
| `loop-data.json` | `agent-loop-trace.html` | The trace below, joined with the message history and collapsed into per-call timelines. |
| `django-14672.trace.jsonl` | `loop-data.json` | 1,411 records from one instrumented run: 19 requests, 19 responses, 1,373 stream events. Request records carry the shape of the history about to be resent; event records carry a millisecond offset, event type, delta type and delta text; response records carry the usage ledger and stop reason. |
| `harness-comparison.json` | `swe-bench-report.html` | 19 published SWE-bench submissions scored on the same 20 instances this agent ran. |

The raw per-instance results those 19 submissions came from are not vendored. Re-fetch with:

    gh api --paginate repos/SWE-bench/experiments/contents/evaluation/verified --jq '.[].name'

## Regenerating

Run records, transcripts, patches and scoring output stay out of git (see `bench/.gitignore`)
and live under `bench/logs/<run-id>/` on the machine that ran them. To reproduce the trace:

    export TINYORBIT_TRACE=/tmp/trace.jsonl
    bench/.venv/bin/python bench/run.py --tasks bench/tasks-easy10.json \
      --only django__django-14672 --trace --thinking-display summarized
