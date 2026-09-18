"""Session persistence: save/load round-trip, resume vs fork, and the opener."""

from __future__ import annotations

from pathlib import Path

from tinyorbit.bootstrap import Config
from tinyorbit.repl import _open_session
from tinyorbit.sessions import (
    load_session,
    restore_into,
    save_session,
    session_path,
)
from tinyorbit.state import CostTracker, SessionState


class _Block:
    """Stands in for an SDK content block: model_dump() is what gets persisted."""

    def __init__(self, **d):
        self._d = d

    def model_dump(self):
        return dict(self._d)


def _session(**kw):
    s = SessionState(**{k: v for k, v in kw.items() if k in ("session_id", "turns")})
    s.messages = kw.get("messages", [])
    s.cost = kw.get("cost", CostTracker())
    return s


# ── round-trip ────────────────────────────────────────────────────────────

def test_save_then_load_round_trips_messages_and_cost(tmp_path: Path):
    cost = CostTracker(api_calls=3, input_tokens=100, output_tokens=42, cache_read_tokens=7)
    s = _session(session_id="abc123", turns=2,
                 messages=[{"role": "user", "content": "hi"}], cost=cost)
    save_session(tmp_path, s, model="claude-opus-5")

    loaded = load_session(tmp_path, "abc123")
    assert loaded is not None
    assert loaded.session_id == "abc123"
    assert loaded.turns == 2
    assert loaded.model == "claude-opus-5"
    assert loaded.messages == [{"role": "user", "content": "hi"}]
    assert loaded.cost.api_calls == 3
    assert loaded.cost.output_tokens == 42
    assert loaded.cost.cache_read_tokens == 7


def test_sdk_blocks_are_dumped_to_plain_dicts(tmp_path: Path):
    s = _session(session_id="blk", messages=[
        {"role": "assistant", "content": [_Block(type="text", text="ok")]},
    ])
    save_session(tmp_path, s)
    loaded = load_session(tmp_path, "blk")
    # The live object became the wire-shape dict the Messages API accepts on resend.
    assert loaded.messages == [
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]


def test_load_missing_session_is_none(tmp_path: Path):
    assert load_session(tmp_path, "nope") is None


def test_save_writes_under_the_id(tmp_path: Path):
    save_session(tmp_path, _session(session_id="xyz"))
    assert session_path(tmp_path, "xyz").exists()


# ── resume vs fork ──────────────────────────────────────────────────────────

def test_resume_keeps_the_id(tmp_path: Path):
    save_session(tmp_path, _session(session_id="keepme", turns=1,
                                    messages=[{"role": "user", "content": "x"}]))
    saved = load_session(tmp_path, "keepme")
    session = restore_into(saved, fork=False)
    assert session.session_id == "keepme"
    assert session.turns == 1
    assert session.messages == [{"role": "user", "content": "x"}]


def test_fork_takes_a_new_id_but_copies_history(tmp_path: Path):
    save_session(tmp_path, _session(session_id="origin", turns=4,
                                    messages=[{"role": "user", "content": "y"}]))
    saved = load_session(tmp_path, "origin")
    session = restore_into(saved, fork=True)
    assert session.session_id != "origin"
    assert session.messages == [{"role": "user", "content": "y"}]
    assert session.turns == 4


def test_fork_does_not_touch_the_original_file(tmp_path: Path):
    save_session(tmp_path, _session(session_id="origin",
                                    messages=[{"role": "user", "content": "keep"}]))
    session = restore_into(load_session(tmp_path, "origin"), fork=True)
    session.messages.append({"role": "user", "content": "diverged"})
    save_session(tmp_path, session)
    # Original still holds only its message; the fork lives under a new id.
    assert load_session(tmp_path, "origin").messages == [{"role": "user", "content": "keep"}]
    assert session_path(tmp_path, session.session_id).exists()


def test_forked_history_is_a_copy_not_shared(tmp_path: Path):
    save_session(tmp_path, _session(session_id="src", messages=[{"role": "user", "content": "a"}]))
    saved = load_session(tmp_path, "src")
    session = restore_into(saved, fork=True)
    session.messages.append({"role": "user", "content": "b"})
    assert saved.messages == [{"role": "user", "content": "a"}]  # untouched


# ── _open_session (the repl wiring) ──────────────────────────────────────────

def _config(tmp_path: Path, **kw) -> Config:
    return Config(cwd=tmp_path, model="claude-opus-5", data_dir=tmp_path, **kw)


def test_open_session_fresh_when_no_resume(tmp_path: Path):
    session = _open_session(_config(tmp_path))
    assert session.messages == []
    assert session.turns == 0


def test_open_session_resumes(tmp_path: Path, capsys):
    save_session(tmp_path, _session(session_id="live", turns=2,
                                    messages=[{"role": "user", "content": "prev"}]))
    session = _open_session(_config(tmp_path, resume_session="live"))
    assert session.session_id == "live"
    assert session.messages == [{"role": "user", "content": "prev"}]
    assert "resumed live" in capsys.readouterr().out


def test_open_session_forks(tmp_path: Path, capsys):
    save_session(tmp_path, _session(session_id="base",
                                    messages=[{"role": "user", "content": "prev"}]))
    session = _open_session(_config(tmp_path, resume_session="base", fork_session=True))
    assert session.session_id != "base"
    assert session.messages == [{"role": "user", "content": "prev"}]
    assert "forked base" in capsys.readouterr().out


def test_open_session_missing_id_starts_fresh(tmp_path: Path, capsys):
    session = _open_session(_config(tmp_path, resume_session="ghost"))
    assert session.messages == []
    assert "no saved session" in capsys.readouterr().out


def test_open_session_survives_a_corrupt_file(tmp_path: Path):
    (tmp_path / "sessions").mkdir()
    session_path(tmp_path, "broken").write_text("{ not json")
    session = _open_session(_config(tmp_path, resume_session="broken"))
    assert session.messages == []  # soft miss, not a crash
