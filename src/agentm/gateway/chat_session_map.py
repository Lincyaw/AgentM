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

from loguru import logger


class ChatSessionMap:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._routes: dict[str, str] = {}
        self._metadata: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("chat_session_map: failed to load {}: {}", self._path, exc)
            return
        if not isinstance(data, dict):
            return
        routes: dict[str, str] = {}
        metadata: dict[str, dict[str, str]] = {}
        for raw_key, raw_value in data.items():
            key = str(raw_key)
            if isinstance(raw_value, str):
                routes[key] = raw_value
                continue
            if not isinstance(raw_value, dict):
                continue
            session_id = raw_value.get("session_id")
            if isinstance(session_id, str) and session_id:
                routes[key] = session_id
            meta = {
                name: value
                for name in ("model", "scenario")
                if isinstance((value := raw_value.get(name)), str) and value
            }
            if meta:
                metadata[key] = meta
        self._routes = routes
        self._metadata = metadata

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, str | dict[str, str]] = {}
        for key in sorted(set(self._routes) | set(self._metadata)):
            route = self._routes.get(key)
            meta = self._metadata.get(key, {})
            if meta:
                item = dict(meta)
                if route:
                    item["session_id"] = route
                payload[key] = item
            elif route:
                payload[key] = route
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self._path.parent, delete=False, suffix=".tmp"
        ) as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
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

    def snapshot_metadata(self) -> dict[str, dict[str, str]]:
        with self._lock:
            return {key: dict(value) for key, value in self._metadata.items()}

    def metadata(self, session_key: str) -> dict[str, str]:
        with self._lock:
            return dict(self._metadata.get(session_key, {}))

    def set_metadata(
        self,
        session_key: str,
        *,
        model: str | None = None,
        scenario: str | None = None,
    ) -> None:
        with self._lock:
            meta = dict(self._metadata.get(session_key, {}))
            for key, value in (("model", model), ("scenario", scenario)):
                if value is None:
                    continue
                if value:
                    meta[key] = value
                else:
                    meta.pop(key, None)
            if meta:
                self._metadata[session_key] = meta
            else:
                self._metadata.pop(session_key, None)
            self._persist()
