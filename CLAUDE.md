# CLAUDE.md

## What Is This Repository?

**tinyorbit**: a small terminal coding agent in Python. Runnable and tested. It demonstrates the core patterns of a production AI coding agent: bootstrap, an explicit agent loop, a streaming API layer, a tool system with permissions, memory files, and a REPL.

## Layout

```
.
  main.py                 # tinyorbit CLI entry
  src/tinyorbit/          # the agent: bootstrap, query loop, api, tools, permissions, memory, repl
  tests/                  # pytest, scripted model, no network
  bench/                  # SWE-bench Verified runner (5-instance sample), own venv
  pyproject.toml          # uv project, Python 3.11+, anthropic SDK
  notes.md                # walkthrough notes on how tinyorbit works
  README.md               # tinyorbit readme
```

## Rules

- Python 3.11+, `uv` for everything: `uv sync --group dev`, `uv run pytest`, `uv run python main.py`.
- The only third-party dependency is the `anthropic` SDK, used as a transport. tinyorbit owns its own agent loop; do not switch to the SDK's tool runner.
- Do not describe tinyorbit as a clone or copy of any product in code, docstrings, or README.
- Every non-obvious design choice gets a short comment naming the pattern it demonstrates.
- Tests use the `QueryDeps` injection seam with a scripted model. Never add tests that need network.
- Do not commit `bench/` outputs (transcripts, patches, predictions, results).

## Git

- Remote: badlogicmanpreet/tinyorbit, branch `main`.
- Squash before pushing if commits contain sensitive content in history.
- Commit messages: what changed and why, not what files were touched.
