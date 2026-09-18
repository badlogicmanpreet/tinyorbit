"""allow/deny tool-pool filtering: select_tools + the CLI/env glob parser."""

from __future__ import annotations

from types import SimpleNamespace

from tinyorbit.bootstrap import _tool_globs
from tinyorbit.tools import select_tools


def pool(*names):
    return [SimpleNamespace(name=n) for n in names]


def names(tools):
    return [t.name for t in tools]


# ── select_tools ────────────────────────────────────────────────────────────

def test_empty_allow_keeps_all():
    p = pool("Read", "Write", "Bash")
    assert names(select_tools(p)) == ["Read", "Write", "Bash"]
    assert names(select_tools(p, [], [])) == ["Read", "Write", "Bash"]


def test_allow_is_a_whitelist():
    assert names(select_tools(pool("Read", "Write", "Bash"), ["Read", "Grep"])) == ["Read"]


def test_deny_removes():
    assert names(select_tools(pool("Read", "Bash"), None, ["Bash"])) == ["Read"]


def test_deny_wins_over_allow():
    assert names(select_tools(pool("Read", "Bash"), ["Read", "Bash"], ["Bash"])) == ["Read"]


def test_globs_match_namespaced_names():
    p = pool("Read", "mcp__gw__query", "mcp__gw__delete", "mcp__other__x")
    assert names(select_tools(p, ["mcp__gw__*"])) == ["mcp__gw__query", "mcp__gw__delete"]
    assert names(select_tools(p, None, ["mcp__gw__delete"])) == \
        ["Read", "mcp__gw__query", "mcp__other__x"]


# ── _tool_globs (CLI wins, env fallback, comma-joinable) ─────────────────────

def test_globs_cli_flattens_commas_and_repeats():
    assert _tool_globs(["Read,Grep", "Edit"], "X") == ["Read", "Grep", "Edit"]


def test_globs_env_fallback(monkeypatch):
    monkeypatch.setenv("X_TOOLS", "Glob, Bash ,")
    assert _tool_globs([], "X_TOOLS") == ["Glob", "Bash"]


def test_globs_cli_beats_env(monkeypatch):
    monkeypatch.setenv("X_TOOLS", "Bash")
    assert _tool_globs(["Read"], "X_TOOLS") == ["Read"]


def test_globs_empty_when_neither_set(monkeypatch):
    monkeypatch.delenv("X_TOOLS", raising=False)
    assert _tool_globs([], "X_TOOLS") == []
