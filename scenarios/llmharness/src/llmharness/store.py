"""File-based store for the harness pipeline.

Layout (rooted at ``HarnessStore.root``)::

    inbox/<sid>.jsonl              # transcript deltas, append-only
    events/<sid>.jsonl             # extracted events, append-only
    cursor/<sid>.json              # {"last_turn_index": N, "next_event_id": M, ...}
    cursor/<sid>.lock              # flock target for serializing tick() per session
    pending_reminders/<sid>.json   # at most one pending reminder; deleted on inject
    verdicts/<sid>.jsonl           # cognitive-audit verdicts, append-only

All readers tolerate missing files (treat as empty). All writers create parent
directories as needed. Inbox/event JSONL writes rely on POSIX ``O_APPEND``
atomicity for small lines; ``HarnessStore.session_lock`` serializes the
worker's read-modify-write tick across processes.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import Event, Reminder, Turn, Verdict, dumps_jsonl, loads_jsonl


@dataclass(frozen=True)
class _Cursor:
    last_turn_index: int = -1
    next_event_id: int = 0
    last_reminder_at_index: int = -1  # turn index when last reminder was written


class HarnessStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    # --- paths -----------------------------------------------------------------

    def _path(self, kind: str, sid: str, suffix: str) -> Path:
        return self.root / kind / f"{sid}{suffix}"

    def inbox_path(self, sid: str) -> Path:
        return self._path("inbox", sid, ".jsonl")

    def events_path(self, sid: str) -> Path:
        return self._path("events", sid, ".jsonl")

    def cursor_path(self, sid: str) -> Path:
        return self._path("cursor", sid, ".json")

    def lock_path(self, sid: str) -> Path:
        return self._path("cursor", sid, ".lock")

    def reminder_path(self, sid: str) -> Path:
        return self._path("pending_reminders", sid, ".json")

    def verdicts_path(self, sid: str) -> Path:
        return self._path("verdicts", sid, ".jsonl")

    @contextlib.contextmanager
    def session_lock(self, sid: str) -> Iterator[None]:
        """Per-session advisory lock. Use to serialize tick() across workers.

        Always yields exactly once and releases on exit, even on error.
        """

        path = self.lock_path(sid)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = path.open("a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()

    # --- inbox -----------------------------------------------------------------

    def append_inbox(self, sid: str, turns: list[Turn]) -> None:
        if not turns:
            return
        path = self.inbox_path(sid)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(dumps_jsonl([t.to_dict() for t in turns]))

    def read_inbox(self, sid: str) -> list[Turn]:
        path = self.inbox_path(sid)
        if not path.exists():
            return []
        return [Turn.from_dict(r) for r in loads_jsonl(path.read_text(encoding="utf-8"))]

    def list_sessions(self) -> list[str]:
        inbox_dir = self.root / "inbox"
        if not inbox_dir.exists():
            return []
        return sorted(p.stem for p in inbox_dir.glob("*.jsonl"))

    # --- events ----------------------------------------------------------------

    def append_events(self, sid: str, events: list[Event]) -> None:
        if not events:
            return
        path = self.events_path(sid)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(dumps_jsonl([e.to_dict() for e in events]))

    def read_events(self, sid: str) -> list[Event]:
        path = self.events_path(sid)
        if not path.exists():
            return []
        return [Event.from_dict(r) for r in loads_jsonl(path.read_text(encoding="utf-8"))]

    # --- cursor ----------------------------------------------------------------

    def read_cursor(self, sid: str) -> _Cursor:
        path = self.cursor_path(sid)
        if not path.exists():
            return _Cursor()
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return _Cursor(
            last_turn_index=int(data.get("last_turn_index", -1)),
            next_event_id=int(data.get("next_event_id", 0)),
            last_reminder_at_index=int(data.get("last_reminder_at_index", -1)),
        )

    def write_cursor(self, sid: str, cursor: _Cursor) -> None:
        path = self.cursor_path(sid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "last_turn_index": cursor.last_turn_index,
                    "next_event_id": cursor.next_event_id,
                    "last_reminder_at_index": cursor.last_reminder_at_index,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # --- pending reminder ------------------------------------------------------

    def write_reminder(self, reminder: Reminder) -> None:
        path = self.reminder_path(reminder.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reminder.to_dict(), ensure_ascii=False), encoding="utf-8")

    def pop_reminder(self, sid: str) -> Reminder | None:
        path = self.reminder_path(sid)
        if not path.exists():
            return None
        reminder = Reminder.from_dict(json.loads(path.read_text(encoding="utf-8")))
        path.unlink()
        return reminder

    # --- verdicts (cognitive audit) -------------------------------------------

    def append_verdict(self, sid: str, verdict: Verdict) -> None:
        path = self.verdicts_path(sid)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(dumps_jsonl([verdict.to_dict()]))

    def read_verdicts(self, sid: str) -> list[Verdict]:
        path = self.verdicts_path(sid)
        if not path.exists():
            return []
        return [
            Verdict.from_dict(r)
            for r in loads_jsonl(path.read_text(encoding="utf-8"))
        ]

    def recent_verdicts(self, sid: str, n: int = 5) -> list[Verdict]:
        """Return the last ``n`` verdicts (oldest first within the slice).

        Used by the cognitive-audit adapter to feed ``recent_alerts`` into
        the diagnostic child session per design §5.2.
        """

        if n <= 0:
            return []
        verdicts = self.read_verdicts(sid)
        return verdicts[-n:]
