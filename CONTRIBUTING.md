# Contributing to tinyorbit

Thanks for taking a look. tinyorbit is a small, deliberately readable coding agent, and
contributions are welcome as long as they keep it that way.

## Setup

```bash
git clone https://github.com/badlogicmanpreet/tinyorbit
cd tinyorbit
uv sync --group dev          # Python 3.11+
uv run pytest                # 24 tests, no network, no API key needed
uv run ruff check .
```

You only need an `ANTHROPIC_API_KEY` to actually run the agent, never to develop or test it.

## What this project optimizes for

Read these before a larger change; a patch that conflicts with one of them will be hard to accept.

- **One third-party dependency.** The `anthropic` SDK is used as a transport and nothing else.
  tinyorbit owns its own agent loop; please do not replace it with the SDK's tool runner.
- **Readability over cleverness.** Every non-obvious design choice carries a short comment naming
  the pattern it demonstrates. If your change needs a paragraph to explain, it needs a comment.
- **Explicit state.** The loop reconstructs its whole state at each continue site rather than
  mutating fields, so every transition is legible in one place. Keep that property.
- **Deliberate formatting.** Lint is scoped to correctness and import hygiene (`ruff check`).
  There is no auto-formatter, on purpose: the aligned comments and tables are hand-set. Do not
  run `ruff format` across the tree.

## Tests

Tests drive the loop through the `QueryDeps` injection seam with a scripted model: a list of
canned responses or exceptions, plus a fake compactor. They assert on the event sequence and on
the exact request messages the loop would have sent.

**Never add a test that requires network access or an API key.** If you cannot express the
behaviour through the seam, that usually means the behaviour belongs behind the seam.

Tool tests run against a temporary directory with real subprocesses. That is fine; it is not network.

## Benchmarks

`bench/` runs the agent against SWE-bench Verified in Docker. It costs real money and takes hours,
so it is not part of CI and not expected of contributors. If you do run it, note that run outputs
(transcripts, patches, predictions, scoring) are intentionally untracked. Finished write-ups go in
`reports/`. See the README for commands.

## Pull requests

- Branch from `main`, keep the change focused, and make sure `uv run pytest` and `uv run ruff check .`
  both pass.
- Commit messages say what changed and why, not which files were touched.
- Update `CHANGELOG.md` under `Unreleased` if the change is user-visible.
- If you changed agent behaviour, say in the PR how you verified it, and whether any benchmark
  numbers in the README or `reports/` are now stale.

## Reporting bugs

Open an issue using the bug report template. For anything with a security dimension, read
[SECURITY.md](SECURITY.md) first and report privately instead.
