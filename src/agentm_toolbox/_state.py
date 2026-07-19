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
import re
from dataclasses import asdict, dataclass
from tempfile import NamedTemporaryFile


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

    # -- serialization (for sandbox CLI mode) --------------------------------

    def dump(self) -> str:
        return json.dumps(
            {k: asdict(v) for k, v in self._states.items()},
            separators=(",", ":"),
        )

    @classmethod
    def load(cls, data: str) -> "ReadStateStore":
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("toolbox state must be a JSON object")
        store = cls()
        expected_fields = {
            "total_lines",
            "is_partial",
            "mtime_ns",
            "content_hash",
        }
        for path, fields in payload.items():
            if not isinstance(path, str) or not path or "\0" in path:
                raise ValueError("toolbox state contains an invalid path")
            if not isinstance(fields, dict) or set(fields) != expected_fields:
                raise ValueError(f"toolbox state for {path!r} has invalid fields")
            total_lines = fields["total_lines"]
            is_partial = fields["is_partial"]
            mtime_ns = fields["mtime_ns"]
            content_hash = fields["content_hash"]
            if (
                not isinstance(total_lines, int)
                or isinstance(total_lines, bool)
                or total_lines < 0
                or not isinstance(is_partial, bool)
                or not isinstance(mtime_ns, int)
                or isinstance(mtime_ns, bool)
                or mtime_ns < 0
                or not isinstance(content_hash, str)
                or (
                    content_hash
                    and re.fullmatch(r"[0-9a-f]{64}", content_hash) is None
                )
            ):
                raise ValueError(f"toolbox state for {path!r} is invalid")
            store._states[path] = FileReadState(
                total_lines=total_lines,
                is_partial=is_partial,
                mtime_ns=mtime_ns,
                content_hash=content_hash,
            )
        return store

    def save_to(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        directory = os.path.dirname(path) or "."
        with NamedTemporaryFile(
            "w",
            dir=directory,
            delete=False,
            encoding="utf-8",
        ) as handle:
            tmp_path = handle.name
            try:
                handle.write(self.dump())
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                os.unlink(tmp_path)
                raise
        try:
            os.replace(tmp_path, path)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @classmethod
    def load_from(cls, path: str) -> "ReadStateStore":
        try:
            with open(path, encoding="utf-8") as f:
                return cls.load(f.read())
        except FileNotFoundError:
            return cls()
