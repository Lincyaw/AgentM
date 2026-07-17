"""Read-state tracking for read-before-edit coordination.

Records what the agent has read so that write/edit can enforce safety
guards (read-before-write, partial-read rejection, content-hash
staleness detection).

State is keyed by normalized path.  In local mode the instance lives
in-process for the session lifetime.  In sandbox mode the CLI runner
serializes/deserializes state to a JSON file between exec invocations.

Zero external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class FileReadState:
    total_lines: int
    is_partial: bool
    mtime_ns: int = 0
    content_hash: str = ""


def content_hash_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ReadStateStore:
    """In-memory read-state store, optionally backed by a JSON file."""

    def __init__(self) -> None:
        self._states: dict[str, FileReadState] = {}

    def record(
        self,
        path: str,
        *,
        total_lines: int,
        is_partial: bool,
        mtime_ns: int = 0,
        content_hash: str = "",
    ) -> None:
        normalized = os.path.normpath(path)
        self._states[normalized] = FileReadState(
            total_lines=total_lines,
            is_partial=is_partial,
            mtime_ns=mtime_ns,
            content_hash=content_hash,
        )

    def get(self, path: str) -> FileReadState | None:
        return self._states.get(os.path.normpath(path))

    def file_modified_since_read(self, path: str) -> bool:
        normalized = os.path.normpath(path)
        state = self._states.get(normalized)
        if state is None or state.mtime_ns == 0:
            return False
        try:
            current_mtime_ns = os.stat(normalized).st_mtime_ns
        except OSError:
            return False
        return current_mtime_ns > state.mtime_ns

    # -- serialization (for sandbox CLI mode) --------------------------------

    def dump(self) -> str:
        return json.dumps(
            {k: asdict(v) for k, v in self._states.items()},
            separators=(",", ":"),
        )

    @classmethod
    def load(cls, data: str) -> "ReadStateStore":
        store = cls()
        for path, fields in json.loads(data).items():
            store._states[path] = FileReadState(**fields)
        return store

    def save_to(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.dump())

    @classmethod
    def load_from(cls, path: str) -> "ReadStateStore":
        try:
            with open(path) as f:
                return cls.load(f.read())
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
