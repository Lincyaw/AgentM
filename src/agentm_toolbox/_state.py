"""Session-local read state for read-before-mutation coordination."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


@dataclass(slots=True)
class FileReadState:
    total_lines: int
    is_partial: bool
    mtime_ns: int = 0
    content_hash: str = ""


def content_hash_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ReadStateStore:
    """In-memory read-state store owned by one file-tool session."""

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
