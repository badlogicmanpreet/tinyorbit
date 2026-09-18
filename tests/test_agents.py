"""Subagent dispatch tests — discovery, the Task tool, and a nested loop.

All network-free: the nested query() runs against the same scripted-model seam
(QueryDeps) the top-level loop tests use.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from tinyorbit.agents import (
    AgentDefinition,
    SubagentContext,
    discover_agents,
    parse_agent_file,
)
from tinyorbit.permissions import PermissionMode, PermissionPolicy
from tinyorbit.query import QueryDeps
from tinyorbit.state import CostTracker
from tinyorbit.tools import ToolUseContext, get_all_base_tools
from tinyorbit.tools.task_tool import TaskTool

# ── fakes (mirror test_query.py's scripted model) ────────────────────────────

def text(t: str):
    return SimpleNamespace(type="text", text=t)


def tool_use(id: str, name: str, **inp):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=inp)


def message(*content, stop_reason="end_turn"):
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )


def scripted(*responses):
    from tinyorbit.providers.base import ModelResponse

    queue = list(responses)
    requests = []

    async def call_model(client, req, abort):
        requests.append(req)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        yield ModelResponse(item)

    return call_model, requests


def make_ctx(tmp_path: Path, agents, deps, *, tools=None):
    config = SimpleNamespace(
        model="test-model",
        capabilities={"tools": tools if tools is not None else get_all_base_tools()},
        max_turns=None, effort=None, fallbacks=True, thinking_display="omitted",
    )
    sc = SubagentContext(
        config=config, provider=None, cost=CostTracker(), agents=agents, deps=deps,
    )
    return ToolUseContext(
        cwd=tmp_path, abort=asyncio.Event(),
        permissions=PermissionPolicy(mode=PermissionMode.BYPASS),
        data_dir=tmp_path / ".data", subagent=sc,
    )


REVIEWER = AgentDefinition("reviewer", "reviews code", "You are a reviewer.")


# ── discovery ─────────────────────────────────────────────────────────────────

def test_discover_empty_without_dir(tmp_path):
    assert discover_agents(tmp_path) == []


def test_discover_parses_frontmatter(tmp_path):
    d = tmp_path / ".tinyorbit" / "agents"
    d.mkdir(parents=True)
    (d / "researcher.md").write_text(
        "---\nname: researcher\ndescription: digs through docs\n"
        "tools: Read, Grep\nmodel: some-model\neffort: medium\nmax_turns: 40\n---\n"
        "You research things. Be thorough.\n"
    )
    agents = discover_agents(tmp_path)
    assert len(agents) == 1
    a = agents[0]
    assert a.name == "researcher"
    assert a.description == "digs through docs"
    assert a.tools == ["Read", "Grep"]
    assert a.model == "some-model"
    assert a.effort == "medium"
    assert a.max_turns == 40
    assert a.prompt == "You research things. Be thorough."


def test_parse_no_frontmatter_is_all_prompt(tmp_path):
    f = tmp_path / "plain.md"
    f.write_text("Just be helpful.")
    a = parse_agent_file(f)
    assert a.name == "plain"
    assert a.prompt == "Just be helpful."
    assert a.tools is None          # omitted -> inherit everything
    assert a.model is None
    assert a.effort is None         # omitted -> inherit the parent loop's budget
    assert a.max_turns is None


def test_parse_empty_body_is_skipped(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("---\nname: x\n---\n")
    assert parse_agent_file(f) is None


# ── Task tool schema ────────────────────────────────────────────────────────

def test_task_tool_schema_lists_agents():
    t = TaskTool([REVIEWER])
    assert t.name == "Task"
    assert t.input_schema["properties"]["subagent_type"]["enum"] == ["reviewer"]
    assert "reviewer: reviews code" in t.description


def test_task_tool_validate_requires_prompt():
    t = TaskTool([REVIEWER])
    assert t.validate({"subagent_type": "reviewer", "prompt": ""}) is not None
    assert t.validate({"subagent_type": "reviewer", "prompt": "go"}) is None


# ── dispatch ──────────────────────────────────────────────────────────────────

async def test_dispatch_returns_final_text(tmp_path):
    call_model, requests = scripted(message(text("the answer is 42")))
    ctx = make_ctx(tmp_path, [REVIEWER], QueryDeps(call_model=call_model))
    res = await TaskTool([REVIEWER]).call(
        {"subagent_type": "reviewer", "prompt": "what is the answer"}, ctx
    )
    assert not res.is_error
    assert res.data == "the answer is 42"
    # The child saw the agent's system prompt and the task, not the parent's history.
    assert requests[0].system == [{"type": "text", "text": "You are a reviewer."}]
    assert requests[0].messages[0]["content"] == "what is the answer"


async def test_dispatch_runs_child_tool(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n")
    call_model, _ = scripted(
        message(tool_use("t1", "Read", file_path="a.txt"), stop_reason="tool_use"),
        message(text("file read")),
    )
    ctx = make_ctx(tmp_path, [REVIEWER], QueryDeps(call_model=call_model))
    res = await TaskTool([REVIEWER]).call(
        {"subagent_type": "reviewer", "prompt": "read a.txt"}, ctx
    )
    assert not res.is_error and res.data == "file read"


async def test_per_subagent_effort_overrides_parent(tmp_path):
    # The subagent declares effort=medium; the parent config runs effort=high.
    # The child's model request must carry the subagent's effort, not the parent's.
    bounded = AgentDefinition("bounded", "runs bounded", "You are bounded.", effort="medium")
    call_model, requests = scripted(message(text("done")))
    config = SimpleNamespace(
        model="test-model", capabilities={"tools": get_all_base_tools()},
        max_turns=None, effort="high", fallbacks=True, thinking_display="omitted",
    )
    sc = SubagentContext(config=config, provider=None, cost=CostTracker(),
                         agents=[bounded], deps=QueryDeps(call_model=call_model))
    ctx = ToolUseContext(
        cwd=tmp_path, abort=asyncio.Event(),
        permissions=PermissionPolicy(mode=PermissionMode.BYPASS),
        data_dir=tmp_path / ".data", subagent=sc,
    )
    res = await TaskTool([bounded]).call({"subagent_type": "bounded", "prompt": "go"}, ctx)
    assert not res.is_error
    assert requests[0].effort == "medium"


async def test_subagent_inherits_parent_budget_when_unset(tmp_path):
    call_model, requests = scripted(message(text("done")))
    config = SimpleNamespace(
        model="test-model", capabilities={"tools": get_all_base_tools()},
        max_turns=None, effort="high", fallbacks=True, thinking_display="omitted",
    )
    sc = SubagentContext(config=config, provider=None, cost=CostTracker(),
                         agents=[REVIEWER], deps=QueryDeps(call_model=call_model))
    ctx = ToolUseContext(
        cwd=tmp_path, abort=asyncio.Event(),
        permissions=PermissionPolicy(mode=PermissionMode.BYPASS),
        data_dir=tmp_path / ".data", subagent=sc,
    )
    res = await TaskTool([REVIEWER]).call({"subagent_type": "reviewer", "prompt": "go"}, ctx)
    assert not res.is_error
    assert requests[0].effort == "high"   # REVIEWER sets no effort -> parent's


async def test_unknown_agent_is_error(tmp_path):
    call_model, _ = scripted(message(text("unused")))
    ctx = make_ctx(tmp_path, [REVIEWER], QueryDeps(call_model=call_model))
    res = await TaskTool([REVIEWER]).call(
        {"subagent_type": "ghost", "prompt": "hi"}, ctx
    )
    assert res.is_error and "unknown subagent 'ghost'" in res.data


async def test_no_subagent_context_fails_closed(tmp_path):
    ctx = ToolUseContext(
        cwd=tmp_path, abort=asyncio.Event(),
        permissions=PermissionPolicy(mode=PermissionMode.BYPASS),
        data_dir=tmp_path / ".data",
    )  # subagent left as None
    res = await TaskTool([REVIEWER]).call(
        {"subagent_type": "reviewer", "prompt": "hi"}, ctx
    )
    assert res.is_error and "not available" in res.data


async def test_early_exit_surfaces_reason(tmp_path):
    # A max_turns exit is reported but not flagged as an error (partial work is real).
    call_model, _ = scripted(
        message(tool_use("t1", "Read", file_path="a.txt"), stop_reason="tool_use"),
    )
    (tmp_path / "a.txt").write_text("x\n")
    config = SimpleNamespace(
        model="test-model",
        capabilities={"tools": get_all_base_tools()},
        max_turns=1, effort=None, fallbacks=True, thinking_display="omitted",
    )
    sc = SubagentContext(config=config, provider=None, cost=CostTracker(),
                         agents=[REVIEWER], deps=QueryDeps(call_model=call_model))
    ctx = ToolUseContext(
        cwd=tmp_path, abort=asyncio.Event(),
        permissions=PermissionPolicy(mode=PermissionMode.BYPASS),
        data_dir=tmp_path / ".data", subagent=sc,
    )
    res = await TaskTool([REVIEWER]).call(
        {"subagent_type": "reviewer", "prompt": "loop"}, ctx
    )
    assert not res.is_error and "max_turns" in res.data


# ── recursion bound & tool filtering ─────────────────────────────────────────

def test_child_pool_excludes_task():
    task = TaskTool([REVIEWER])
    sc = SimpleNamespace(config=SimpleNamespace(
        capabilities={"tools": [*get_all_base_tools(), task]}))
    child = TaskTool._child_tools(sc, REVIEWER)
    assert "Task" not in {t.name for t in child}
    assert "Read" in {t.name for t in child}


def test_child_pool_respects_agent_allowlist():
    narrow = AgentDefinition("narrow", "", "p", tools=["Read", "Grep"])
    sc = SimpleNamespace(config=SimpleNamespace(
        capabilities={"tools": get_all_base_tools()}))
    child = TaskTool._child_tools(sc, narrow)
    assert {t.name for t in child} == {"Read", "Grep"}


async def test_subagent_cannot_dispatch_subagents(tmp_path):
    # Give the parent pool a Task tool, then have the subagent try to call Task.
    # The child context carries no subagent handle, so the nested Task fails closed.
    parent_pool = [*get_all_base_tools(), TaskTool([REVIEWER])]
    call_model, _ = scripted(
        message(tool_use("t1", "Task", subagent_type="reviewer", prompt="again"),
                stop_reason="tool_use"),
        message(text("gave up on nesting")),
    )
    ctx = make_ctx(tmp_path, [REVIEWER], QueryDeps(call_model=call_model),
                   tools=parent_pool)
    res = await TaskTool([REVIEWER]).call(
        {"subagent_type": "reviewer", "prompt": "try to nest"}, ctx
    )
    # The child never had Task in its pool (filtered out), so the model's Task
    # call comes back as an unknown-tool error, and the child recovers with text.
    assert not res.is_error and res.data == "gave up on nesting"
