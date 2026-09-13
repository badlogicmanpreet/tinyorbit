# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `--thinking-display {omitted,summarized}`. The API omits thinking text by default on current
  models, so without this flag every thinking block arrives carrying an empty string and the
  terminal's reasoning renderer never fires. The default is unchanged.
- `TINYORBIT_TRACE=<path>`. When set, the transport appends one JSON record per server-sent event:
  a millisecond offset, the event type, the delta type and its text, plus the shape of the history
  about to be resent and the usage ledger that comes back. Unset, it costs one environment lookup
  per call.
- Benchmark harness: `--tasks` to select a task set, `--timeout-min` for the per-task wall-clock cap,
  and `--trace` to copy a stream trace out of the container alongside the transcript.
- Three frozen SWE-bench Verified task sets so runs are reproducible: `tasks-easy10.json` (seed 42),
  `tasks-hard10.json` (the 1-4 h and >4 h tail, stratified by repository), and `tasks-five.json`.
- `reports/` with two self-contained HTML write-ups and the data behind them: benchmark results
  across twenty instances, and one task traced at the wire and replayed.
- Project infrastructure: CI on Python 3.11 through 3.13, ruff lint configuration, issue and pull
  request templates, contributing guide, security policy, and code of conduct.

### Changed

- `zip()` over tool calls and their results is now `strict=True`, asserting the invariant that
  every `tool_use` receives exactly one `tool_result`.
- README leads with benchmark results rather than description alone.
- Packaging metadata: real description, project URLs, classifiers and keywords.

### Fixed

- Removed unused imports and sorted imports across the tree.

### Security

- `key.txt` is ignored, and the benchmark harness redacts the API key from its own logs and excludes
  local secret files from the copy it sends into a container.

## [0.1.0]

Initial import: the agent loop, streaming API layer with yield-based retry, the tool system with
permissions and concurrency partitioning, memory files, the REPL and headless print mode, a pytest
suite that runs against a scripted model with no network, and a SWE-bench Verified runner.
