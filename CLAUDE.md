# CLAUDE.md

## What Is This Repository?

Two things live here:

1. **tinyorbit** (repo root): a small terminal coding agent in Python. Runnable, tested, built one chapter at a time as the companion implementation to the book.
2. **content/**: the book and everything around it. `content/book/` is an 18-chapter technical book on the architecture of a production AI coding agent, written in the style of an O'Reilly technical book. `content/web/` is its Astro site.

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
  content/
    README.md             # book cover, TOC, key patterns
    book/                 # ch01 .. ch18 as markdown
    prompts/              # reusable prompts for book generation
    web/                  # Astro static site (GitHub Pages)
    diagrams/ mybook/     # drafts
    .reference/           # local-only source material (gitignored)
```

## Rules for tinyorbit (root code)

- Python 3.11+, `uv` for everything: `uv sync --group dev`, `uv run pytest`, `uv run python main.py`.
- The only third-party dependency is the `anthropic` SDK, used as a transport. tinyorbit owns its own agent loop; do not switch to the SDK's tool runner.
- Do not describe tinyorbit as a clone or copy of any product in code, docstrings, or README. Chapter references (`Ch 5`) are fine.
- Every non-obvious design choice gets a short comment naming the pattern it demonstrates.
- Tests use the `QueryDeps` injection seam with a scripted model. Never add tests that need network.
- Do not commit `bench/` outputs (transcripts, patches, predictions, results).

## Rules for the book (content/book)

- **No verbatim source code.** All code blocks must be pseudocode with different variable names. The book teaches patterns, not implementations.
- **Mermaid for diagrams.** Use ```mermaid fenced code blocks.
- **Each chapter: opening → body → Apply This.** The Apply This section has exactly 5 transferable patterns.
- **One concept, one home.** Cross-reference instead of repeating.
- Voice: expert peer explaining to another expert. Direct, opinionated, no filler.

## Git

- `content/.reference/` is gitignored; never commit source files or raw analysis.
- Squash before pushing if commits contain sensitive content in history.
- Commit messages: what changed and why, not what files were touched.
- Remote: alejandrobalderas/claude-code-from-source. The web site deploys from `content/web` via `.github/workflows/deploy.yml`.
