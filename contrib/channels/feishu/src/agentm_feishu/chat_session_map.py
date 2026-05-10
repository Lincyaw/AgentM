"""Maps chat (or chat+thread) routes to AgentM session ids.

Persisted as a flat JSON file under ``<state_dir>/chat_sessions.json``
so that gateway restarts can resume the conversation in the same
``AgentSession`` rather than spawning a fresh agent for every chat
restart. The file is rewritten atomically (write-then-rename) — small
enough that we don't bother with a real database.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


def _route_key(chat_id: str, thread_id: str | None) -> str:
    return f"{chat_id}::{thread_id}" if thread_id else chat_id


class ChatSessionMap:
    """In-memory map with disk-backed persistence."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._routes: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            self._routes = {str(k): str(v) for k, v in data.items() if isinstance(v, str)}

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic rewrite: write to a temp file in the same directory, then rename.
        # ``os.replace`` is atomic on POSIX and Windows for files on the same FS.
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._path.parent,
            delete=False,
            suffix=".tmp",
        ) as fh:
            json.dump(self._routes, fh, indent=2, sort_keys=True)
            tmp_name = fh.name
        os.replace(tmp_name, self._path)

    def get(self, chat_id: str, thread_id: str | None = None) -> str | None:
        with self._lock:
            return self._routes.get(_route_key(chat_id, thread_id))

    def set(self, chat_id: str, session_id: str, thread_id: str | None = None) -> None:
        with self._lock:
            self._routes[_route_key(chat_id, thread_id)] = session_id
            self._persist()

    def drop(self, chat_id: str, thread_id: str | None = None) -> None:
        with self._lock:
            self._routes.pop(_route_key(chat_id, thread_id), None)
            self._persist()

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._routes)

    @classmethod
    def from_dict(cls, path: Path, data: dict[str, Any]) -> ChatSessionMap:
        instance = cls(path)
        instance._routes = {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
        return instance
