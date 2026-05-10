"""Maps a session_key (``channel:chat_id`` or thread-scoped override)
to an AgentSession id, persisted to JSON.

Lets gateway restarts resume the same agent for the same conversation
instead of spawning a fresh session every time the daemon bounces. The
file is small enough that we don't bother with a database — atomic
write-then-rename is sufficient.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path


class ChatSessionMap:
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
            self._routes = {
                str(k): str(v) for k, v in data.items() if isinstance(v, str)
            }

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self._path.parent, delete=False, suffix=".tmp"
        ) as fh:
            json.dump(self._routes, fh, indent=2, sort_keys=True)
            tmp = fh.name
        os.replace(tmp, self._path)

    def get(self, session_key: str) -> str | None:
        with self._lock:
            return self._routes.get(session_key)

    def set(self, session_key: str, session_id: str) -> None:
        with self._lock:
            self._routes[session_key] = session_id
            self._persist()

    def drop(self, session_key: str) -> None:
        with self._lock:
            self._routes.pop(session_key, None)
            self._persist()

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._routes)
