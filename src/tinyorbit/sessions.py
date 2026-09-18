"""Session persistence — resume and fork (Ch 3).

A session is just its message history plus the cost ledger (see state.py), so
persisting it is a JSON round-trip. That is enough to *resume* (reopen an id and
keep appending) or *fork* (load an id's history but write under a fresh id, so
the original is untouched — the basis for exploring alternatives from a point).

Serialization detail worth naming: assistant turns are stored as
`{"role": "assistant", "content": <blocks>}` where the blocks are the
provider's SDK objects. `model_dump()` turns them into plain `{"type": ...}`
dicts, which is exactly the shape the Messages API accepts on resend — so a
reloaded, all-dict history replays without a provider-specific decoder.

Sessions live under `<data_dir>/sessions/<id>.json`. Saving is best-effort: a
write failure must never take a turn down, so callers ignore the return.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tinyorbit.state import CostTracker, SessionState

SESSIONS_SUBDIR = "sessions"


@dataclass
class SavedSession:
    session_id: str
    messages: list[Any] = field(default_factory=list)
    cost: CostTracker = field(default_factory=CostTracker)
    turns: int = 0
    model: str | None = None
    saved_at: str | None = None


def sessions_dir(data_dir: Path) -> Path:
    return Path(data_dir) / SESSIONS_SUBDIR


def session_path(data_dir: Path, session_id: str) -> Path:
    return sessions_dir(data_dir) / f"{session_id}.json"


def _json_default(o: object) -> object:
    # SDK content blocks (TextBlock, ToolUseBlock, …) expose model_dump();
    # anything else degrades to its string form rather than failing the write.
    return o.model_dump() if hasattr(o, "model_dump") else str(o)


def save_session(data_dir: Path, session: SessionState, *, model: str | None = None) -> Path | None:
    """Write the session to disk. Returns the path, or None if the write failed."""
    try:
        store = sessions_dir(data_dir)
        store.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "session_id": session.session_id,
            "model": model,
            "turns": session.turns,
            "cost": asdict(session.cost),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "messages": session.messages,
        }
        path = session_path(data_dir, session.session_id)
        path.write_text(json.dumps(payload, indent=1, default=_json_default))
        return path
    except OSError:
        return None


def load_session(data_dir: Path, session_id: str) -> SavedSession | None:
    """Load a saved session by id, or None if it does not exist / is unreadable."""
    path = session_path(data_dir, session_id)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    cost = CostTracker(**{k: v for k, v in (data.get("cost") or {}).items()
                          if k in CostTracker.__dataclass_fields__})
    return SavedSession(
        session_id=data.get("session_id", session_id),
        messages=data.get("messages", []),
        cost=cost,
        turns=data.get("turns", 0),
        model=data.get("model"),
        saved_at=data.get("saved_at"),
    )


def restore_into(saved: SavedSession, *, fork: bool) -> SessionState:
    """Build a SessionState from a saved one.

    resume -> keep the id, so saving overwrites the same file and the history
    grows in place. fork -> a fresh id, so the original session file is left
    intact and the two diverge from this point on.
    """
    session = SessionState()  # fresh id from the default factory
    if not fork:
        session.session_id = saved.session_id
    session.messages = list(saved.messages)
    session.cost = saved.cost
    session.turns = saved.turns
    return session
