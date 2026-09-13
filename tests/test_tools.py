from __future__ import annotations

import asyncio
from pathlib import Path

from tinyorbit.permissions import PermissionMode, PermissionPolicy, rule_matches
from tinyorbit.tools import ToolCall, ToolUseContext, check_schema, get_all_base_tools, run_tool
from tinyorbit.tools.bash_tool import BashTool, classify_read_only
from tinyorbit.tools.execute import MAX_RESULT_CHARS
from tinyorbit.tools.file_tools import EditTool, ReadTool, WriteTool


def ctx(tmp_path: Path, mode=PermissionMode.BYPASS) -> ToolUseContext:
    return ToolUseContext(cwd=tmp_path, abort=asyncio.Event(), permissions=PermissionPolicy(mode=mode), data_dir=tmp_path / ".d")


async def test_edit_requires_read_first(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a = 1\n")
    c = ctx(tmp_path)
    res = await EditTool().call({"file_path": "x.py", "old_string": "1", "new_string": "2"}, c)
    assert res.is_error and "not been read" in res.data

    await ReadTool().call({"file_path": "x.py"}, c)
    res = await EditTool().call({"file_path": "x.py", "old_string": "1", "new_string": "2"}, c)
    assert not res.is_error and f.read_text() == "a = 2\n"


async def test_edit_detects_external_change(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a = 1\n")
    c = ctx(tmp_path)
    await ReadTool().call({"file_path": "x.py"}, c)
    c.file_state[f.resolve()] -= 10        # pretend the read happened earlier than the write
    res = await EditTool().call({"file_path": "x.py", "old_string": "1", "new_string": "2"}, c)
    assert res.is_error and "changed on disk" in res.data


async def test_edit_rejects_ambiguous_match(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("x\nx\n")
    c = ctx(tmp_path)
    await ReadTool().call({"file_path": "x.py"}, c)
    res = await EditTool().call({"file_path": "x.py", "old_string": "x", "new_string": "y"}, c)
    assert res.is_error and "2 places" in res.data
    res = await EditTool().call({"file_path": "x.py", "old_string": "x", "new_string": "y", "replace_all": True}, c)
    assert not res.is_error and f.read_text() == "y\ny\n"


async def test_write_creates_dirs_and_read_numbers_lines(tmp_path):
    c = ctx(tmp_path)
    await WriteTool().call({"file_path": "sub/dir/new.txt", "content": "one\ntwo\n"}, c)
    res = await ReadTool().call({"file_path": "sub/dir/new.txt", "offset": 2}, c)
    assert res.data.strip() == "2\ttwo"


def test_bash_read_only_classification():
    assert classify_read_only("git status && ls -la | head")
    assert classify_read_only("grep -rn foo src")
    assert not classify_read_only("git push origin main")
    assert not classify_read_only("echo hi > out.txt")
    assert not classify_read_only("rm -rf build")
    assert not classify_read_only("npm test")
    assert not classify_read_only("ls 'unterminated")


async def test_bash_runs_and_reports_exit_code(tmp_path):
    c = ctx(tmp_path)
    ok = await BashTool().call({"command": "printf hello"}, c)
    assert ok.data == "hello" and not ok.is_error
    bad = await BashTool().call({"command": "echo oops >&2; exit 3"}, c)
    assert bad.is_error and "[exit code 3]" in bad.data and "oops" in bad.data


async def test_bash_timeout_kills_process(tmp_path):
    res = await BashTool().call({"command": "sleep 5", "timeout_ms": 100}, ctx(tmp_path))
    assert res.is_error and "timed out" in res.data


async def test_run_tool_validates_schema(tmp_path):
    res = await run_tool(ToolCall("id", "Read", {}), get_all_base_tools(), ctx(tmp_path))
    assert res["is_error"] and "missing required field 'file_path'" in res["content"]
    res = await run_tool(ToolCall("id", "Read", {"file_path": 3}), get_all_base_tools(), ctx(tmp_path))
    assert res["is_error"] and "must be a string" in res["content"]
    res = await run_tool(ToolCall("id", "Nope", {}), get_all_base_tools(), ctx(tmp_path))
    assert res["is_error"] and "Unknown tool" in res["content"]


async def test_run_tool_budgets_large_output(tmp_path):
    (tmp_path / "big.txt").write_text("x" * (MAX_RESULT_CHARS + 5000))
    res = await run_tool(ToolCall("id", "Bash", {"command": "cat big.txt"}), get_all_base_tools(), ctx(tmp_path))
    assert "output truncated" in res["content"]
    saved = list((tmp_path / ".d" / "tool-results").glob("*.txt"))
    assert len(saved) == 1 and len(saved[0].read_text()) == MAX_RESULT_CHARS + 5000


def test_check_schema_rejects_bool_as_int():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    assert check_schema(schema, {"n": True}) is not None
    assert check_schema(schema, {"n": 3}) is None


async def test_permission_chain(tmp_path):
    tools = {t.name: t for t in get_all_base_tools()}
    c = ctx(tmp_path, PermissionMode.DEFAULT)
    policy = c.permissions

    # read-only tools are always allowed
    assert (await policy.can_use_tool(tools["Read"], {"file_path": "x"}, c)).behavior == "allow"
    assert (await policy.can_use_tool(tools["Bash"], {"command": "git status"}, c)).behavior == "allow"
    # writes need a human; none available → deny
    assert (await policy.can_use_tool(tools["Bash"], {"command": "git push"}, c)).behavior == "deny"
    # explicit rule wins
    policy.allow_rules.append("Bash(git push*)")
    assert (await policy.can_use_tool(tools["Bash"], {"command": "git push origin"}, c)).behavior == "allow"
    # acceptEdits allows file writes inside cwd only
    policy.mode = PermissionMode.ACCEPT_EDITS
    assert (await policy.can_use_tool(tools["Write"], {"file_path": "in.txt", "content": ""}, c)).behavior == "allow"
    assert (await policy.can_use_tool(tools["Write"], {"file_path": "/tmp/out.txt", "content": ""}, c)).behavior == "deny"
    # the ask function is the last resort
    asked = []

    async def ask(prompt):
        asked.append(prompt)
        return True

    policy.ask = ask
    assert (await policy.can_use_tool(tools["Bash"], {"command": "npm test"}, c)).behavior == "allow"
    assert asked == ["Bash: npm test"]


def test_rule_matching():
    bash = BashTool()
    assert rule_matches("Bash", bash, {"command": "anything"})
    assert rule_matches("Bash(git *)", bash, {"command": "git log"})
    assert not rule_matches("Bash(git *)", bash, {"command": "gh pr"})
    assert not rule_matches("Read", bash, {"command": "ls"})
