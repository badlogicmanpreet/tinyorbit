## What changed and why

<!-- Describe the change and the reason for it. Link an issue if there is one. -->

## How it was verified

<!-- Which tests, which commands, what you observed. -->

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check .` passes

## Checklist

- [ ] No new third-party dependency (the `anthropic` SDK is the only one)
- [ ] No test requires network access or an API key
- [ ] Non-obvious design choices carry a short comment naming the pattern
- [ ] `CHANGELOG.md` updated under `Unreleased` if the change is user-visible
- [ ] If agent behaviour changed, benchmark numbers in `README.md` and `reports/` are still accurate,
      or are called out as stale
