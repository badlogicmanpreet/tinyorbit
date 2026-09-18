"""Provider-layer tests — registry resolution and OpenAI dialect translation.

No network: the translation functions are pure, and resolution is exercised
through the registry. This is the SDK seam that lets tinyorbit talk to any
model without the loop knowing which dialect answered.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tinyorbit.providers import (
    ModelRequest,
    Provider,
    ProviderCapabilities,
    available_providers,
    register_provider,
    resolve_provider,
)
from tinyorbit.providers.base import ModelCallError, _infer_provider
from tinyorbit.providers.openai import (
    OpenAIProvider,
    _finalize,
    build_request,
    encode_messages,
    encode_system,
    encode_tools,
)

# ── registry / resolution ─────────────────────────────────────────────────────

def test_builtin_dialects_registered():
    names = available_providers()
    assert "anthropic" in names and "openai" in names


def test_infer_provider_by_model_id():
    assert _infer_provider("claude-opus-5") == "anthropic"
    assert _infer_provider("gpt-4o") == "openai"
    assert _infer_provider("o3") == "openai"
    assert _infer_provider(None) == "anthropic"          # safe default


def test_explicit_provider_beats_inference():
    p = resolve_provider(provider="openai", model="claude-opus-5")
    assert isinstance(p, OpenAIProvider)


def test_env_provider_used(monkeypatch):
    monkeypatch.setenv("TINYORBIT_PROVIDER", "openai")
    p = resolve_provider(model="claude-opus-5")           # id says anthropic, env wins
    assert isinstance(p, OpenAIProvider)


def test_unknown_provider_raises():
    with pytest.raises(ModelCallError) as e:
        resolve_provider(provider="does-not-exist")
    assert "unknown provider" in str(e.value)


def test_opt_in_provider_registers_via_lazy_import():
    # A provider absent from the registry is looked up as tinyorbit.providers.<name>.
    # We simulate that discovery having happened by registering, then resolving.
    class DummyProvider(Provider):
        name = "dummy"

        def __init__(self, **_):
            pass

    register_provider("dummy", DummyProvider)
    assert isinstance(resolve_provider(provider="dummy"), DummyProvider)


# ── OpenAI outbound translation ───────────────────────────────────────────────

def test_encode_system_flattens_and_drops_cache_control():
    system = [
        {"type": "text", "text": "identity", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "today is friday"},
    ]
    assert encode_system(system) == "identity\n\ntoday is friday"


def test_encode_tools_to_function_shape():
    tools = [{"name": "Read", "description": "read a file",
              "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]
    out = encode_tools(tools)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "Read"
    assert out[0]["function"]["parameters"]["properties"]["path"]["type"] == "string"


def test_encode_messages_tool_use_becomes_tool_calls():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "reading"},
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a.txt"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "alpha"},
        ]},
    ]
    out = encode_messages(messages)
    assert out[0] == {"role": "user", "content": "hi"}
    assistant = out[1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "t1"
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    # tool result becomes its own role:"tool" message
    tool_msg = out[-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "t1"
    assert tool_msg["content"] == "alpha"


def test_build_request_prepends_system_and_sets_tool_choice():
    req = ModelRequest(
        model="gpt-4o",
        system=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "Read", "description": "", "input_schema": {"type": "object", "properties": {}}}],
        effort="high",
    )
    kw = build_request(req)
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["tool_choice"] == "auto"
    assert kw["reasoning_effort"] == "high"
    assert kw["stream"] is True


# ── OpenAI inbound translation ────────────────────────────────────────────────

def test_finalize_maps_finish_reason_and_builds_blocks():
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
    msg = _finalize("hello", {}, "stop", usage)
    assert msg.stop_reason == "end_turn"
    assert msg.content[0].type == "text" and msg.content[0].text == "hello"
    assert msg.usage.input_tokens == 100 and msg.usage.output_tokens == 20


def test_finalize_tool_calls_force_tool_use_stop():
    tool_calls = {0: {"id": "c0", "name": "Read", "arguments": '{"path": "a.txt"}'}}
    # gateway reported "stop" but there are tool calls → normalize to tool_use
    msg = _finalize("", tool_calls, "stop", SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    assert msg.stop_reason == "tool_use"
    block = msg.content[0]
    assert block.type == "tool_use" and block.name == "Read" and block.input == {"path": "a.txt"}


def test_finalize_length_maps_to_max_tokens():
    msg = _finalize("x", {}, "length", SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    assert msg.stop_reason == "max_tokens"


def test_finalize_bad_json_arguments_degrade_to_empty():
    tool_calls = {0: {"id": "c0", "name": "Read", "arguments": "{not valid"}}
    msg = _finalize("", tool_calls, "tool_calls", SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    assert msg.content[0].input == {}


# ── round-trip through the loop with a fake OpenAI client ─────────────────────

class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c
        return gen()


def _chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


async def test_openai_provider_streams_text_and_response(monkeypatch):
    # a fake AsyncOpenAI whose completions.create yields two content chunks + usage
    async def fake_create(**kwargs):
        return _FakeStream([
            _chunk(content="hel"),
            _chunk(content="lo"),
            _chunk(finish_reason="stop", usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2)),
        ])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    provider = OpenAIProvider(client=client)

    req = ModelRequest(model="gpt-4o", system=[{"type": "text", "text": "s"}],
                       messages=[{"role": "user", "content": "hi"}])
    events = [e async for e in provider.stream(req, asyncio.Event())]

    texts = [e.text for e in events if e.__class__.__name__ == "TextDelta"]
    assert "".join(texts) == "hello"
    final = events[-1].message
    assert final.stop_reason == "end_turn"
    assert final.usage.input_tokens == 5


def test_openai_capabilities_are_lowest_common_denominator():
    caps = OpenAIProvider.capabilities
    assert isinstance(caps, ProviderCapabilities)
    assert caps.prompt_caching is False and caps.adaptive_thinking is False
