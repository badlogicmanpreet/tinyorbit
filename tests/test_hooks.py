"""PreToolUse guard-hook tests: the runner, integration through run_tool, discovery."""

from __future__ import annotations

import asyncio

from tinyorbit.hooks import (
    HookInput,
    HookOutcome,
    HookSpec,
    discover_hooks,
    run_pre_tool_use,
)
from tinyorbit.permissions import PermissionMode, PermissionPolicy
from tinyorbit.tools import ToolCall, ToolUseContext, run_tool
from tinyorbit.tools.base import Tool, ToolResult


def ctx(tmp_path, mode=PermissionMode.BYPASS, hooks=None):
    return ToolUseContext(
        cwd=tmp_path, abort=asyncio.Event(),
        permissions=PermissionPolicy(mode=mode), data_dir=tmp_path / ".d",
        hooks=hooks or [],
    )


class SpyTool(Tool):
    name = "Spy"
    description = "records whether it ran"
    input_schema = {"type": "object", "properties": {}}

    def __init__(self):
        self.ran = False

    def is_read_only(self, inp):        # a "write" so DEFAULT mode would prompt
        return False

    async def call(self, inp, c):
        self.ran = True
        return ToolResult("ran")


def deny_hook(reason="nope", match="*"):
    return HookSpec(lambda i: HookOutcome("deny", reason), match=match)


def allow_hook(match="*"):
    return HookSpec(lambda i: HookOutcome("allow"), match=match)


# ── the runner ────────────────────────────────────────────────────────────────

async def test_empty_hooks_pass(tmp_path):
    out = await run_pre_tool_use([], "Read", {}, ctx(tmp_path))
    assert out.decision == "pass"


async def test_deny_wins_over_allow(tmp_path):
    hooks = [allow_hook(), deny_hook("blocked")]
    out = await run_pre_tool_use(hooks, "Bash", {}, ctx(tmp_path))
    assert out.decision == "deny" and out.reason == "blocked"


async def test_deny_wins_even_when_registered_first(tmp_path):
    hooks = [deny_hook("blocked"), allow_hook()]
    out = await run_pre_tool_use(hooks, "Bash", {}, ctx(tmp_path))
    assert out.decision == "deny"


async def test_matcher_scopes_the_hook(tmp_path):
    hooks = [deny_hook("only bash", match="Bash")]
    assert (await run_pre_tool_use(hooks, "Read", {}, ctx(tmp_path))).decision == "pass"
    assert (await run_pre_tool_use(hooks, "Bash", {}, ctx(tmp_path))).decision == "deny"


async def test_raising_hook_denies_fail_closed(tmp_path):
    def boom(i):
        raise RuntimeError("kaboom")

    out = await run_pre_tool_use([HookSpec(boom)], "Bash", {}, ctx(tmp_path))
    assert out.decision == "deny" and "kaboom" in out.reason


async def test_async_hook_supported(tmp_path):
    async def guard(i: HookInput):
        return HookOutcome("deny", f"no {i.tool_name}")

    out = await run_pre_tool_use([HookSpec(guard)], "Write", {}, ctx(tmp_path))
    assert out.decision == "deny" and out.reason == "no Write"


async def test_none_return_is_pass(tmp_path):
    out = await run_pre_tool_use([HookSpec(lambda i: None)], "Read", {}, ctx(tmp_path))
    assert out.decision == "pass"


# ── integration through run_tool ──────────────────────────────────────────────

async def test_deny_blocks_execution(tmp_path):
    spy = SpyTool()
    res = await run_tool(ToolCall("i", "Spy", {}), [spy],
                         ctx(tmp_path, hooks=[deny_hook("policy says no")]))
    assert res["is_error"] and "Blocked by policy" in res["content"]
    assert not spy.ran


async def test_allow_bypasses_permission(tmp_path):
    # DEFAULT mode with no asker would deny a write; an allow hook overrides that.
    spy = SpyTool()
    res = await run_tool(ToolCall("i", "Spy", {}), [spy],
                         ctx(tmp_path, mode=PermissionMode.DEFAULT, hooks=[allow_hook()]))
    assert not res.get("is_error") and spy.ran


async def test_pass_falls_through_to_permission(tmp_path):
    # No hook opinion + DEFAULT + no asker -> the permission chain denies the write.
    spy = SpyTool()
    res = await run_tool(ToolCall("i", "Spy", {}), [spy],
                         ctx(tmp_path, mode=PermissionMode.DEFAULT,
                             hooks=[HookSpec(lambda i: None)]))
    assert res["is_error"] and "Permission denied" in res["content"]
    assert not spy.ran


async def test_hook_sees_tool_input(tmp_path):
    seen = {}

    def capture(i: HookInput):
        seen["name"] = i.tool_name
        seen["input"] = dict(i.tool_input)
        return None

    spy = SpyTool()
    await run_tool(ToolCall("i", "Spy", {"path": "x"}), [spy],
                   ctx(tmp_path, hooks=[HookSpec(capture)]))
    assert seen == {"name": "Spy", "input": {"path": "x"}}


# ── discovery ─────────────────────────────────────────────────────────────────

def test_discover_empty_without_env(monkeypatch):
    monkeypatch.delenv("TINYORBIT_HOOKS", raising=False)
    assert discover_hooks() == []


def test_discover_imports_named_module(tmp_path, monkeypatch):
    mod = tmp_path / "myhooks.py"
    mod.write_text(
        "from tinyorbit.hooks import HookSpec, HookOutcome\n"
        "def pre_tool_use_hooks():\n"
        "    return [HookSpec(lambda i: HookOutcome('deny', 'x'), match='Bash')]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("TINYORBIT_HOOKS", "myhooks")
    hooks = discover_hooks()
    assert len(hooks) == 1 and hooks[0].match == "Bash"


def test_discover_skips_bad_module(monkeypatch):
    monkeypatch.setenv("TINYORBIT_HOOKS", "no_such_module_xyz")
    assert discover_hooks() == []
