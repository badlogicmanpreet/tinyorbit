"""Runtime embedding seam: startup wiring, sink pass-through, resume, cleanup.

run_turn drives the real query loop (network); here we stub it and the provider
so the tests exercise the *adapter* — how Runtime assembles context, forwards a
caller's event sink, tracks the session, and tears down — with no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tinyorbit import runtime as rt
from tinyorbit.bootstrap import Config
from tinyorbit.sessions import save_session
from tinyorbit.state import SessionState


class FakeProvider:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeMcp:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the network-touching startup pieces and the loop driver."""
    provider = FakeProvider()
    mcp = FakeMcp()
    monkeypatch.setattr(rt, "resolve_provider", lambda **_: provider)

    async def fake_start_mcp(config):
        return mcp

    monkeypatch.setattr(rt, "_start_mcp", fake_start_mcp)

    calls = []

    async def fake_run_turn(config, session, ctx, provider, prompt, sink=None):
        calls.append({"prompt": prompt, "ctx": ctx, "provider": provider})
        # Emit a couple of events so the sink wiring is observable, then advance
        # the session the way the real run_turn would on Done.
        if sink is not None:
            sink({"type": "text", "text": "hello "})
            sink({"type": "text", "text": prompt})
        session.turns += 1
        session.messages.append({"role": "user", "content": prompt})
        return f"echo:{prompt}"

    monkeypatch.setattr(rt, "run_turn", fake_run_turn)
    return {"provider": provider, "mcp": mcp, "calls": calls}


def _config(tmp_path: Path, **kw) -> Config:
    return Config(cwd=tmp_path, model="claude-opus-5", data_dir=tmp_path, **kw)


async def test_submit_returns_result_and_advances_session(tmp_path, stubbed):
    async with rt.Runtime(_config(tmp_path)) as run:
        result = await run.submit("do a thing")
    assert result.text == "echo:do a thing"
    assert result.turns == 1
    assert result.session_id == run.session_id


async def test_submit_forwards_events_to_the_sink(tmp_path, stubbed):
    seen = []
    async with rt.Runtime(_config(tmp_path)) as run:
        await run.submit("ping", on_event=seen.append)
    assert seen == [{"type": "text", "text": "hello "}, {"type": "text", "text": "ping"}]


async def test_provider_and_mcp_closed_on_exit(tmp_path, stubbed):
    async with rt.Runtime(_config(tmp_path)) as run:
        await run.submit("x")
    assert stubbed["provider"].closed is True
    assert stubbed["mcp"].closed is True


async def test_submit_before_start_is_an_error(tmp_path, stubbed):
    run = rt.Runtime(_config(tmp_path))
    with pytest.raises(RuntimeError):
        await run.submit("too early")


async def test_context_carries_the_resolved_provider(tmp_path, stubbed):
    async with rt.Runtime(_config(tmp_path)) as run:
        await run.submit("x")
    # The context handed to run_turn is the one Runtime built at start().
    assert stubbed["calls"][0]["provider"] is stubbed["provider"]
    assert stubbed["calls"][0]["ctx"] is not None


async def test_runtime_resumes_a_saved_session(tmp_path, stubbed):
    s = SessionState(session_id="prior")
    s.messages = [{"role": "user", "content": "earlier"}]
    s.turns = 3
    save_session(tmp_path, s)

    async with rt.Runtime(_config(tmp_path, resume_session="prior")) as run:
        assert run.session_id == "prior"
        assert run.session.messages == [{"role": "user", "content": "earlier"}]
        result = await run.submit("more")
    assert result.turns == 4  # picked up from the restored count


async def test_explicit_session_overrides_resume(tmp_path, stubbed):
    given = SessionState(session_id="injected")
    run = rt.Runtime(_config(tmp_path), session=given)
    assert run.session is given
    assert run.session_id == "injected"
