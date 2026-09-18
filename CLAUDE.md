# CLAUDE.md

## What Is This Repository?

**tinyorbit**: a small terminal coding agent in Python. Runnable and tested. It demonstrates the core patterns of a production AI coding agent: bootstrap, an explicit agent loop, a streaming API layer, a tool system with permissions, memory files, and a REPL.

## Layout

```
.
  main.py                 # tinyorbit CLI entry
  src/tinyorbit/          # the agent: bootstrap, query loop, api, tools, permissions, memory, repl
  tests/                  # pytest, scripted model, no network
  bench/                  # SWE-bench Verified runner + frozen task sets, own venv
  reports/                # committed deliverables: HTML reports and the data they embed
  pyproject.toml          # uv project, Python 3.11+, optional per-dialect extras
  notes.md                # walkthrough notes on how tinyorbit works
  README.md               # tinyorbit readme
```

## Rules

- Python 3.11+, `uv` for everything: `uv sync --group dev`, `uv run pytest`, `uv run python main.py`.
- The open-source core has no hard third-party dependency; each model dialect is an optional extra (`tinyorbit[anthropic]` is the reference transport, `[openai]` the other), imported lazily on first use so a build works with any model without dragging in the others. tinyorbit owns its own agent loop; do not switch to a vendor SDK's tool runner.
- Do not describe tinyorbit as a clone or copy of any product in code, docstrings, or README.
- Every non-obvious design choice gets a short comment naming the pattern it demonstrates.
- Tests use the `QueryDeps` injection seam with a scripted model. Never add tests that need network.
- Do not commit `bench/` outputs (transcripts, patches, predictions, results). Run records live in
  `bench/logs/<run-id>/` and stay local. Finished write-ups go in `reports/`, which is committed.
- Observability is opt-in and costs nothing when off: `TINYORBIT_TRACE=<path>` appends one JSON line
  per stream event, and `--thinking-display summarized` returns the model's reasoning instead of the
  API's default empty placeholder blocks. Leave both defaults alone so benchmark runs stay comparable.

## Git

- Remote: badlogicmanpreet/tinyorbit, branch `main`.
- Squash before pushing if commits contain sensitive content in history.
- Commit messages: what changed and why, not what files were touched.
