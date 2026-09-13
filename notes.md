# tinyorbit notes

Working notes from walking through the code. Start at the top if new; each section builds on the one before.

---

## 1. What tinyorbit is

A small terminal coding agent in Python. You type a request, it calls a model over HTTPS, the model asks for tools, tinyorbit runs them locally, sends the results back, and repeats until the model stops asking. About 1,100 lines, one third-party dependency (the Anthropic SDK), 24 tests that run without network.

It is built alongside the book in `content/book/`. Each module maps to a chapter and demonstrates one pattern from it.

### Harness vs agent

"Harness" means two things depending on context.

| Layer | What it is |
|---|---|
| Benchmark harness | the script that runs a benchmark: starts containers, launches the agent per task, scores patches |
| Agent | the thing being scored: harness + model |
| Agent harness | the agent minus the model: loop, tools, prompt, permissions |
| Model | the thing that reasons |

tinyorbit is an **agent harness**. tinyorbit plus a model is an **agent**. When the subject is running an evaluation, "harness" means the outer script and tinyorbit is the agent under test.

### Why the Anthropic SDK

Strictly optional. The model is one HTTPS endpoint, `POST /v1/messages`. The SDK is used as a transport only, for four tedious things: parsing server-sent events and reassembling them into one message, finding credentials, typed errors (429 becomes `RateLimitError`), and validating request shapes client-side. tinyorbit does **not** use the SDK's built-in agent loop; owning the loop is the point. SDK retries are set to zero so there is exactly one retry path, ours.

---

## 2. The REPL

Read, Eval, Print, Loop. A 1960s Lisp idea. The program reads a line, evaluates it, prints, and reads again. A shell is one; the Python `>>>` prompt is one. tinyorbit's `run_interactive` in `repl.py` is one with Eval replaced by the agent loop.

Two defining properties: it is synchronous (you cannot type while it works, only interrupt), and it is line-oriented (a scrolling transcript, not a screen).

### Alternatives

- **Headless, one-shot.** No prompt. Task on the command line, run to completion, exit. tinyorbit's `--print`. The shape for scripts, CI, and benchmarks.
- **Full-screen TUI.** Program owns the screen: panes, persistent input box, dialogs. What most people mean by "terminal agent" today.
- **Client and server.** Agent runs as a local server; TUI, desktop app, IDE extension are separate clients. opencode is built this way. Payoff: sessions outlive windows, multiple front ends. Cost: a protocol to maintain.
- **Embedded in an editor.** Output lands in the editor's panels and diff views.
- **Remote and asynchronous.** Agent runs elsewhere; you assign work and get a pull request back.
- **Agent protocols.** A standard protocol between editors and agents (Agent Client Protocol from Zed), so an agent plugs into any editor without its own extension.

All of these are different **renderers** for the same event stream. tinyorbit's loop yields events; the `Renderer` class in `repl.py` is the entire "Print" step. Everything below it is interface-agnostic.

### Two loops, nested

```
you type ──► REPL reads                          (outer loop: once per user message)
               │
        ┌── agent loop ──────────────────────────┐ (inner loop: once per model call)
        │  HTTPS ► model ► response               │
        │  tools? ── yes ─► run locally ─► back ──┘
        │          └─ no ─► Done
        └─────────────────────────────────────────┘
               │
        REPL shows the prompt again
```

- Printing is continuous, not "in between": text streams as generated, tool starts and finishes print as they happen.
- Only the model is remote. Files, edits, shell commands, permissions all stay local. The model asks; the local loop does.
- The model has no memory between calls. Every call sends the full history. That is why the loop's whole job is appending to a message list, and why context size eventually has to be managed.

### Does a client/server design change that?

No. opencode's server is a second process on your own laptop, not the model's server. It stores history in a local database and lets several front ends attach. When it talks to the model it still sends the full history every call, and it has a compaction module for the same reason tinyorbit does.

| | Loop runs in | History lives in | Model receives per call |
|---|---|---|---|
| tinyorbit | terminal process | memory, gone on exit | full history |
| opencode | local server process | local database | full history |
| the agent in the book | terminal process | JSONL transcript on disk | full history |

History genuinely lives server-side only at the model provider: server-side compaction (the API summarises and hands back a block you echo), or Managed Agents (the provider runs the loop, sandbox and history). That is renting an agent rather than running one.

---

## 3. Code flow

### Startup, once

1. `main.py` `main()`. Handles `--version` / `--help` before any heavy import, then calls the three bootstrap phases. There is no Phase 1 (module-level side effects); Python discourages it.
2. `bootstrap.py` `init(argv)`. Parses flags into `Config`. Then the **trust boundary**: above it only argv has been read; below it, with trust granted, project files may be read. Memoised.
3. `setup(config)`. Fills `config.capabilities`: `tools` (six instances, fixed order), `memory` (instruction files from root down to cwd), `system_prompt` (a stable cached block and a small dynamic one with the date), plus empty `hooks` and `agents` slots.
4. `launch(config)`. Picks `run_print_mode` or `run_interactive`. Both end in `run_turn`.

### Per user message

5. `run_interactive()`. Builds once: a `PermissionPolicy` with a terminal `ask` function, a `SessionState` (messages, cost tracker), one API client. Then: `input()`, slash commands, else clear the abort flag, make Ctrl+C set it, call `run_turn`.
6. `run_turn()`. Appends the user message, builds `QueryParams`, then `async for event in query(params): renderer.handle(event)`. On `Done`, copies the final history back into the session.

### Per model call, the inner loop

7. `query.py` `query()`. `while True`: maybe compact; build `ModelRequest`; iterate `call_model`; inspect `stop_reason`; finish or run tools and rebuild `LoopState`.
8. `api.py` `query_model()`. Retry loop around `_stream_once`, which opens the streaming request, yields `TextDelta` per chunk, checks abort between chunks, then yields `ModelResponse`. Retryable errors yield a `StatusEvent` and sleep. Unrecoverable ones raise `ModelCallError` with a reason.
9. `execute.py` `partition_for_concurrency()` then `run_tool()`. Groups consecutive concurrency-safe calls; runs each group under `asyncio.gather`. `run_tool` is the pipeline: lookup, abort check, schema check, tool `validate`, permission, `tool.call`, output cap. Always returns a `tool_result`.
10. `permissions.py` `can_use_tool()`. The chain: bypass mode, explicit rules, read-only, acceptEdits for writes inside cwd, then ask.
11. The tool's `call()` with validated input and a `ToolUseContext` (cwd, abort event, staleness ledger).

```
main ─► init ─► setup ─► launch
                            │
                     run_interactive ◄──────────────┐  outer loop
                            │                       │
                         run_turn ──────────────────┘
                            │
                          query ◄───────────────┐      inner loop
                       ┌────┴────┐              │
                 query_model   run_tool ────────┘
                  (HTTPS)     ├─ can_use_tool
                              └─ Tool.call
```

Data flows down as `Config` and `QueryParams`, back up as events. Nothing below `run_turn` knows a terminal exists. Nothing below `query` knows a conversation exists.

### One request traced

`> add a docstring to fast_path in main.py`

- **Iteration 1.** Model replies with text plus `tool_use Read main.py`, `stop_reason=tool_use`. Assistant message appended as-is. Read is read-only, allowed without asking. Result wrapped in one user message, appended. `turn_count=1`.
- **Iteration 2.** Model sees the file, replies with `tool_use Edit`. Not read-only; the permission chain reaches `ask`, terminal prompts. Edit checks the staleness ledger, finds `old_string` once, writes. Result appended.
- **Iteration 3.** Text only, `stop_reason=end_turn`. `Done("completed", messages)`.

Three API calls, two tool runs, one prompt.

### The other branches

Each way the happy path breaks has a named exit or retry:

- Ctrl+C during streaming: `Done("aborted_streaming")`.
- Ctrl+C during tools: manufacture error `tool_result` blocks for every unanswered `tool_use` (the API rejects a history with one missing), then `Done("aborted_tools")`.
- Output cut off at the token cap: retry once with a larger cap, then append the partial answer plus a "continue" nudge, up to three times.
- Prompt too long from the server: summarise once and retry; second failure is terminal.
- Rate limit or overload: up to four retries with backoff, each yielding a status event.
- `--max-turns`: stop after N tool rounds, tool results still recorded.
- Context over 150k tokens at the start of an iteration: summarise the history first, max three failures.

Every `continue` site rebuilds the whole `LoopState` by hand. Verbose on purpose: any branch can be read in isolation, and the `transition` field records why the loop went around.

---

## 4. Tools

Six, in `src/tinyorbit/tools/`, registered in fixed order (order is part of the cached prompt prefix).

| Tool | Input | Does | Read-only | Parallel |
|---|---|---|---|---|
| Read | `file_path`, `offset`, `limit` | numbered lines, 2000 per page; records mtime in the staleness ledger | yes | yes |
| Write | `file_path`, `content` | create or overwrite; refuses if the file exists and was not Read, or changed since | no | no |
| Edit | `file_path`, `old_string`, `new_string`, `replace_all` | exact-string replace, must match once unless `replace_all`; same staleness rule | no | no |
| Bash | `command`, `timeout_ms` | shell in cwd, stdout+stderr+exit code, killed on timeout or Ctrl+C | depends | depends |
| Glob | `pattern`, `path` | files by pattern, newest first, skips `.git`, `node_modules`, `.venv` | yes | yes |
| Grep | `pattern`, `path`, `glob`, `max_results` | regex over contents, `path:line:text`, skips binaries | yes | yes |

- **Bash's "depends".** A classifier splits the command on pipes and `&&` and calls it read-only only if every segment starts with a known reader and there is no redirect or `sudo`. That answer decides both whether permission is needed and whether it may run in parallel.
- **Output is capped everywhere.** Read pages, Glob/Grep stop at 200 hits, any result over 30,000 characters is saved to disk and replaced with a preview plus the path.
- **Missing** versus fuller agents: sub-agents, web fetch, todo list, LSP diagnostics, patch tool, MCP. Reserved for later chapters.

---

## 5. Testing and benchmarks

Three levels:

1. **Machinery.** The 24 tests in `tests/`, scripted model, no network. Prove loop, tools, permissions behave. Say nothing about coding ability.
2. **Real model, small tasks.** Ten hand-written tasks in a throwaway repo via `--print --permission-mode bypassPermissions`, check the diffs. Catches integration problems the fakes cannot. A few dollars. Do this first; the live path has not been exercised yet.
3. **Public benchmarks.** SWE-bench Verified (real GitHub issues), Terminal-Bench (shell-heavy tasks), Aider Polyglot (edit quality across languages).

Benchmark scores mostly measure the model. Two harnesses with the same model land within a few points. The useful number is the **gap** between tinyorbit and a reference harness (mini-swe-agent) on the same model and the same tasks. Fifteen points behind means the loop is dropping work; three points means the harness is fine.

### SWE-bench family

- **Original** (2023): ~2,300 issues from 12 Python repos. Many tasks unsolvable from the issue text; rarely run in full now.
- **Lite**: 300 cheaper tasks. Superseded.
- **Verified**: 500 tasks from the original, each human-checked for a clear issue and fair tests. Released mid-2024. The headline number in nearly every model launch. **This is what we use.**
- **Pro** (late 2025): harder, larger repos, multi-file, some private tasks. Starting to appear next to Verified.
- **Multilingual / Multimodal**: other languages, screenshots.

Labs report Verified using their own minimal scaffold (bash + edit tool + fixed prompt), roughly tinyorbit-sized, not the product agent. Scores climbed from low teens at release to a large majority within about 18 months, which is why Pro exists.

### What an instance is

One task: a real GitHub issue frozen before its fix, plus what is needed to check a fix.

- `instance_id` like `django__django-11099` (repo, then the fixing PR number)
- `repo`, `base_commit`: the code before the fix
- `problem_statement`: the issue text, the only thing the agent sees
- `patch`, `test_patch`: the real fix and its tests, hidden
- `FAIL_TO_PASS`: tests that must go from failing to passing; decides resolved
- `PASS_TO_PASS`: tests that must keep passing; catches collateral damage

The **instance image** is a Docker image per instance with the repo at `base_commit` and dependencies installed, repo at `/testbed`. The agent works inside it; evaluation applies the diff to a clean copy and runs both test lists.

Dataset name: `SWE-bench/SWE-bench_Verified` (the new organisation; the older `princeton-nlp/...` copy has the same 500 tasks but no `image` column, and the 5.x evaluator needs that column). Each row names its Docker image, e.g. `swebench/sweb.eval.x86_64.django_1776_django-11734:latest`. The images are x86_64 only; on Apple Silicon they run under emulation, which the project calls experimental.

### Five-task plan

**Prerequisites.** Docker with 30 GB free (images are 1 to 3 GB each). API key with a 40 dollar cap. `pip install swebench` in a scratch venv on the host. Containers can reach the API.

1. **Choose five instances** (10 min). Seed 42 from Verified, save IDs to `five_ids.txt`. Prefer two or three repos; if one repo, seed 43 and note it.
2. **Pull the five images** (20 min). Confirm each starts with the repo at `/testbed` and that `import anthropic` fails inside, so tinyorbit needs its own install.
3. **Dry run on one** (15 min). Copy tinyorbit in, install uv, run `--print` with no key: expect the clean "no credentials" message. Then set the key and run for real, watching the transcript. Fix plumbing before spending more.
4. **Run the five** (1 to 2 h), sequentially. Per task: fresh container, copy in, install, run with `--print --permission-mode bypassPermissions --max-turns 60` and a fixed prompt template (issue text verbatim plus: fix it minimally, run relevant tests, do not commit), save stdout as transcript and stderr for cost, `git -C /testbed diff > patches/<id>.diff`, append `{"instance_id", "model_name_or_path": "tinyorbit", "model_patch"}` to `predictions.jsonl`, kill the container. Empty diffs count as failures; keep them. Table per task: turns, exit reason, cost, wall-clock, diff size.
5. **Score** (20 min). `swebench eval verified -p predictions.jsonl --run-id tinyorbit-5 -j 1`. (swebench 5.x ships a `swebench` command; the old `python -m swebench.harness.run_evaluation` form still exists underneath.)
6. **Baseline** (1 h). mini-swe-agent, same model, same five, same scoring.
7. **Read transcripts** (1 h). Bin each failure: wrong file (search or prompt), edit failed (Edit tool), broke tests (did not run them), ran out (compaction, budget, cap). One sentence per task. This is the deliverable, not the count.

**Outcomes.** Same tasks resolved by both: harness is fine, move to model and prompt. Baseline resolves one tinyorbit does not: read that transcript first. tinyorbit crashes, loops, or blows the cap: a bug, and the most valuable result. Any single task past 8 dollars: stop it, treat as a finding.

**Budget.** Around 25 dollars with the baseline, under 5 hours mostly waiting.

### What is prepared (2026-09-12)

In `bench/`:

- `.venv/` with `swebench` 5.0.2 and `datasets`
- `five_ids.txt`, seed 42 from Verified: pylint-7080, django-11734, astropy-14508, scikit-learn-26323, django-14034 (four repos)
- `tasks.json`, the five rows with issue text, image name, difficulty, test lists
- `prompt.txt`, the fixed template
- `run.py`, the host script: `--smoke` (one container, no key, expects the no-credentials message), `--dry-run`, `--only ID`, full run. Writes transcripts, patches, `predictions.jsonl`, `results.csv`.

Still needed from the machine: Docker Desktop (not installed) and an `ANTHROPIC_API_KEY`. Then: `python run.py --smoke`, then `python run.py`, then `swebench eval verified -p predictions.jsonl --run-id tinyorbit-5 -j 1`.
