"""Loop tests with a scripted model — the injection seam from Ch 5."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinyorbit.api import ModelCallError, ModelResponse, StatusEvent
from tinyorbit.permissions import PermissionMode, PermissionPolicy
from tinyorbit.query import Done, QueryDeps, QueryParams, ToolFinished, ToolStarted, query
from tinyorbit.tools import ToolUseContext, get_all_base_tools


# ── fakes ───────────────────────────────────────────────────────────────────

def text(t: str):
    return SimpleNamespace(type="text", text=t)


def tool_use(id: str, name: str, **inp):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=inp)


def message(*content, stop_reason="end_turn", input_tokens=10):
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=input_tokens, output_tokens=5,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )


def scripted(*responses):
    """A call_model that replays canned responses (or raises the given error)."""
    queue = list(responses)
    requests = []

    async def call_model(client, req, abort):
        requests.append(req)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        yield ModelResponse(item)

    return call_model, requests


def make_params(tmp_path: Path, tools=None, mode=PermissionMode.BYPASS, **kw) -> QueryParams:
    ctx = ToolUseContext(
        cwd=tmp_path, abort=asyncio.Event(),
        permissions=PermissionPolicy(mode=mode), data_dir=tmp_path / ".data",
    )
    return QueryParams(
        messages=[{"role": "user", "content": "hi"}],
        system=[{"type": "text", "text": "sys"}],
        tools=tools if tools is not None else get_all_base_tools(),
        ctx=ctx, model="test-model", **kw,
    )


async def drain(params, deps):
    return [e async for e in query(params, deps)]


# ── tests ───────────────────────────────────────────────────────────────────

async def test_text_only_completes(tmp_path):
    call_model, requests = scripted(message(text("hello")))
    events = await drain(make_params(tmp_path), QueryDeps(call_model=call_model))
    done = events[-1]
    assert isinstance(done, Done) and done.reason == "completed"
    assert len(requests) == 1
    assert done.messages[-1]["role"] == "assistant"


async def test_tool_round_trip(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    call_model, requests = scripted(
        message(tool_use("t1", "Read", file_path="a.txt"), stop_reason="tool_use"),
        message(text("done")),
    )
    events = await drain(make_params(tmp_path), QueryDeps(call_model=call_model))

    assert [type(e).__name__ for e in events if isinstance(e, (ToolStarted, ToolFinished))] == ["ToolStarted", "ToolFinished"]
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert "alpha" in finished.result["content"] and not finished.result.get("is_error")

    # second request carried assistant turn + tool_result in a single user message
    second = requests[1].messages
    assert second[-2]["role"] == "assistant"
    assert second[-1]["role"] == "user"
    assert second[-1]["content"][0]["tool_use_id"] == "t1"
    assert events[-1].reason == "completed"


async def test_parallel_safe_tools_run_together(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    call_model, _ = scripted(
        message(
            tool_use("t1", "Read", file_path="a.txt"),
            tool_use("t2", "Read", file_path="b.txt"),
            tool_use("t3", "Write", file_path="c.txt", content="c"),
            stop_reason="tool_use",
        ),
        message(text("ok")),
    )
    events = await drain(make_params(tmp_path), QueryDeps(call_model=call_model))
    names = [type(e).__name__ for e in events if isinstance(e, (ToolStarted, ToolFinished))]
    # two safe reads start together, then finish; the write is its own group
    assert names == ["ToolStarted", "ToolStarted", "ToolFinished", "ToolFinished", "ToolStarted", "ToolFinished"]
    assert (tmp_path / "c.txt").read_text() == "c"


async def test_max_turns_stops_loop(tmp_path):
    call_model, requests = scripted(
        message(tool_use("t1", "Glob", pattern="*"), stop_reason="tool_use"),
        message(tool_use("t2", "Glob", pattern="*"), stop_reason="tool_use"),
        message(text("never")),
    )
    events = await drain(make_params(tmp_path, max_turns=1), QueryDeps(call_model=call_model))
    assert events[-1].reason == "max_turns"
    assert len(requests) == 1
    # the tool result was still recorded, so history is protocol-valid
    assert events[-1].messages[-1]["content"][0]["tool_use_id"] == "t1"


async def test_abort_during_tools_fills_orphans(tmp_path):
    params = make_params(tmp_path)
    call_model, _ = scripted(
        message(
            tool_use("t1", "Bash", command="echo one"),
            tool_use("t2", "Bash", command="echo two"),
            stop_reason="tool_use",
        ),
    )
    params.ctx.abort.set()
    events = await drain(params, QueryDeps(call_model=call_model))
    done = events[-1]
    assert done.reason == "aborted_tools"
    results = done.messages[-1]["content"]
    assert {r["tool_use_id"] for r in results} == {"t1", "t2"}
    assert all(r.get("is_error") for r in results)


async def test_prompt_too_long_triggers_reactive_compact(tmp_path):
    call_model, requests = scripted(
        ModelCallError("prompt_too_long", "prompt is too long"),
        message(text("after compaction")),
    )
    compacted = []

    async def fake_compact(params, messages):
        compacted.append(messages)
        return [{"role": "user", "content": "summary"}]

    events = await drain(make_params(tmp_path), QueryDeps(call_model=call_model, compact=fake_compact))
    assert len(compacted) == 1
    assert requests[1].messages[0]["content"] == "summary"
    assert events[-1].reason == "completed"


async def test_prompt_too_long_twice_is_terminal(tmp_path):
    call_model, _ = scripted(
        ModelCallError("prompt_too_long", "x"),
        ModelCallError("prompt_too_long", "x"),
    )

    async def fake_compact(params, messages):
        return messages

    events = await drain(make_params(tmp_path), QueryDeps(call_model=call_model, compact=fake_compact))
    assert events[-1].reason == "prompt_too_long"


async def test_max_tokens_escalates_then_recovers(tmp_path):
    call_model, requests = scripted(
        message(text("part 1"), stop_reason="max_tokens"),
        message(text("part 2"), stop_reason="max_tokens"),
        message(text("part 3")),
    )
    events = await drain(make_params(tmp_path), QueryDeps(call_model=call_model))
    assert requests[0].max_tokens < requests[1].max_tokens          # escalation retried same prompt
    assert requests[1].messages == requests[0].messages
    assert requests[2].messages[-1]["role"] == "user"               # recovery appended a nudge
    assert "cut off" in requests[2].messages[-1]["content"]
    assert events[-1].reason == "completed"


async def test_auto_compact_when_context_large(tmp_path):
    call_model, requests = scripted(
        message(tool_use("t1", "Glob", pattern="*"), stop_reason="tool_use", input_tokens=200_000),
        message(text("done")),
    )
    compacted = []

    async def fake_compact(params, messages):
        compacted.append(len(messages))
        return [{"role": "user", "content": "summary"}]

    events = await drain(make_params(tmp_path), QueryDeps(call_model=call_model, compact=fake_compact))
    assert compacted == [3]                                          # user + assistant + tool results
    assert requests[1].messages == [{"role": "user", "content": "summary"}]
    assert any(isinstance(e, StatusEvent) and "compacting" in e.message for e in events)
    assert events[-1].reason == "completed"


async def test_permission_denied_becomes_error_result(tmp_path):
    call_model, requests = scripted(
        message(tool_use("t1", "Bash", command="rm -rf build"), stop_reason="tool_use"),
        message(text("ok, skipped")),
    )
    params = make_params(tmp_path, mode=PermissionMode.DEFAULT)      # no ask fn → deny
    events = await drain(params, QueryDeps(call_model=call_model))
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.result["is_error"] and "Permission denied" in finished.result["content"]
    assert events[-1].reason == "completed"


async def test_model_error_is_terminal(tmp_path):
    call_model, _ = scripted(ModelCallError("model_error", "boom"))
    events = await drain(make_params(tmp_path), QueryDeps(call_model=call_model))
    assert events[-1].reason == "model_error"


async def test_refusal_is_terminal(tmp_path):
    call_model, _ = scripted(message(stop_reason="refusal"))
    events = await drain(make_params(tmp_path), QueryDeps(call_model=call_model))
    assert events[-1].reason == "refusal"
