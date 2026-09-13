# tinyorbit — A Small Terminal Coding Agent in Python

A from-scratch Python coding agent, built alongside the book in [content/book/](content/book/). It is real and runnable: you type a request, it calls the model, runs tools (read/write/edit/bash/glob/grep), and loops until the model stops asking for tools.

Roughly 1,100 lines. Every module maps to a chapter; every non-obvious design choice has a comment pointing at the pattern it demonstrates. [notes.md](notes.md) is a guided walkthrough.

## Running

```bash
uv sync --group dev                    # Python 3.11+, anthropic SDK
export ANTHROPIC_API_KEY=...           # or `ant auth login`

uv run python main.py                              # interactive REPL
uv run python main.py "explain the bootstrap"      # REPL with a first prompt
uv run python main.py --print "list the python files here"   # headless, streams to stdout
uv run pytest                                      # 24 tests, no network
```

Useful flags: `--model`, `--permission-mode {default,acceptEdits,bypassPermissions}`, `--allow 'Bash(git *)'`, `--effort xhigh`, `--max-turns 20`, `--no-fallbacks`. Inside the REPL: `/cost`, `/clear`, `/mode`, `/exit`. Ctrl+C mid-turn interrupts the turn; at the prompt it exits.

Defaults: model `claude-opus-5`, adaptive thinking, server-side refusal fallback on, prompt caching on, permission mode `default` (read-only tools run freely, writes and shell commands prompt).

## The abstractions (mapped to modules)

| Book concept        | Module                        | Chapter | What it demonstrates |
|---------------------|-------------------------------|---------|----------------------|
| Bootstrap           | `src/tinyorbit/bootstrap.py`      | 2       | init → setup → launch; the trust boundary gates reading project files |
| CLI entry           | `main.py`                     | 2       | fast path for `--version`/`--help` before heavy imports |
| State               | `src/tinyorbit/state.py`          | 3       | session ledger: messages, cost tracker, context size |
| API layer           | `src/tinyorbit/api.py`            | 4       | streaming, yield-based retry, cache breakpoints, recoverable-error classification |
| System prompt       | `src/tinyorbit/prompt.py`         | 4       | static cached prefix / dynamic boundary |
| Query loop          | `src/tinyorbit/query.py`          | 5       | async generator; explicit state reconstruction; terminal vs continue reasons |
| Tool system         | `src/tinyorbit/tools/`            | 6       | self-describing tools, fail-closed defaults, input-dependent safety, staleness detection, result budgeting |
| Permissions         | `src/tinyorbit/permissions.py`    | 6       | the resolution chain: bypass → rules → read-only → mode → ask |
| Concurrency         | `src/tinyorbit/tools/execute.py`  | 7       | consecutive concurrency-safe calls run under `asyncio.gather` |
| Memory              | `src/tinyorbit/memory.py`         | 11      | TINYORBIT.md / AGENTS.md from home and every ancestor of cwd |
| REPL / print mode   | `src/tinyorbit/repl.py`           | 13      | event rendering, Ctrl+C as abort signal, permission prompts |

Not built yet (slots exist in `Config.capabilities`): sub-agents (Ch 8), hooks (Ch 12), MCP (Ch 15), the lighter context-management layers (Ch 5: snip/microcompact/collapse). Only auto-compact and reactive compact exist.

## The loop in one screen

```
while True:
    if last request was over the compact threshold:   compact (max 3 failures)
    stream the model
        prompt_too_long  → compact once, retry          (reactive_compact_retry)
        aborted          → Done(aborted_streaming)
        other error      → Done(model_error)
    max_tokens with no tools → retry at 64K once, then nudge "continue" up to 3×
    refusal              → Done(refusal)
    no tool calls        → Done(completed)
    run tools: parallel where every call in the group is concurrency-safe, serial otherwise
        abort mid-batch  → synthesize tool_results for the orphans, Done(aborted_tools)
    max_turns reached    → Done(max_turns)
    state = LoopState(...)   # full reconstruction, transition=next_turn
```

Every `Done` carries the message history, so the caller (REPL or --print) never has to reconstruct it. Every `tool_use` the model emits gets a `tool_result`, even on interruption; that is the invariant the API protocol needs.

## Testing strategy

`QueryDeps` is the injection seam from Chapter 5. Tests pass a scripted `call_model` (a list of canned responses or exceptions) and a fake `compact`, then assert on the event sequence and on the exact request messages the loop would have sent. Tool tests run against a temp directory with real subprocesses.

`bench/` runs tinyorbit against a five-instance sample of SWE-bench Verified. See the last section of [notes.md](notes.md).

## Layout

```
.
  main.py                  # CLI entry (phase 0 fast path → bootstrap)
  pyproject.toml           # uv project; `anthropic>=1.0`
  src/tinyorbit/
    bootstrap.py  api.py  query.py  state.py  prompt.py  memory.py  permissions.py  repl.py
    tools/
      base.py              # Tool, ToolResult, ToolUseContext, schema check
      file_tools.py        # Read, Write, Edit
      bash_tool.py         # Bash + read-only classifier
      search_tools.py      # Glob, Grep
      execute.py           # the pipeline, concurrency partition, orphan safety net
      registry.py
  tests/
    test_query.py          # loop behaviour with a scripted model
    test_tools.py          # tools, permissions, budgeting
  bench/                   # SWE-bench runner: run.py, prompt.txt, tasks.json
  content/                 # the book, its prompts, and the web site
```
