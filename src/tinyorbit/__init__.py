"""tinyorbit — a small terminal coding agent in Python.

This package is built up chapter-by-chapter alongside the book.
Each module corresponds to one of the six core abstractions from Ch 1:

    query.py    -> the agent loop (async generator)
    tools/      -> the tool system (self-describing tools)
    tasks.py    -> sub-agents (recursive query loops)
    state.py    -> two-tier state (mutable singleton + reactive store)
    memory.py   -> persistent AGENTS.md / TINYORBIT.md context
    hooks.py    -> lifecycle interceptors

Plus plumbing:

    api.py        -> Anthropic API wrapper
    bootstrap.py  -> startup / config loading
"""

__version__ = "0.0.1"
