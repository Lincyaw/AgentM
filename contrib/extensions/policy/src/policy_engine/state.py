"""Policy engine state tables — reactions that record raw facts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, deque

from dataclasses import dataclass

from .types import EffectRecord, FileStateEntry, ToolArgs, ToolLogEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMBERS_RE = re.compile(r"\b\d+\b")


def fingerprint(text: str | None) -> str | None:
    """Deterministic error fingerprint: strip noise, SHA-256[:16]."""
    if not text:
        return None
    cleaned = _ANSI_RE.sub("", text)
    cleaned = _NUMBERS_RE.sub("N", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return hashlib.sha256(cleaned.encode()).hexdigest()[:16]


def args_hash(args: ToolArgs) -> str:
    """Deterministic hash of tool call arguments."""
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _classify_error(error_text: str | None) -> str | None:
    """Heuristic error category from error text."""
    if not error_text:
        return None
    low = error_text.lower()
    if "not found" in low or "no such file" in low or "does not exist" in low:
        return "not-found"
    if "permission" in low or "access denied" in low or "forbidden" in low:
        return "permission"
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if "invalid" in low or "validation" in low or "must be" in low:
        return "validation"
    return "runtime"


# ---------------------------------------------------------------------------
# RollingLog — generic bounded append-only log
# ---------------------------------------------------------------------------


class RollingLog[T]:
    """Bounded append-only log backed by a deque."""

    def __init__(self, max_size: int = 500) -> None:
        self._entries: deque[T] = deque(maxlen=max_size)

    def append(self, entry: T) -> None:
        self._entries.append(entry)

    def entries(self) -> deque[T]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# IndexedTable — turn-indexed or key-indexed table
# ---------------------------------------------------------------------------


class IndexedTable[K, V]:
    """Key-indexed state table (one row per key)."""

    def __init__(self) -> None:
        self._entries: dict[K, V] = {}

    def get(self, key: K) -> V | None:
        return self._entries.get(key)

    def record(self, key: K, value: V) -> None:
        self._entries[key] = value


# ---------------------------------------------------------------------------
# FileState — entity map keyed by path
# ---------------------------------------------------------------------------


class FileState:
    """Per-file state tracking. One row per path, upserted on file ops."""

    def __init__(self) -> None:
        self._entries: dict[str, FileStateEntry] = {}

    def get(self, path: str) -> FileStateEntry | None:
        return self._entries.get(path)

    def record_file_op(
        self,
        path: str,
        tool_name: str,
        turn: int,
        content_hash: str | None = None,
    ) -> None:
        existing = self._entries.get(path)
        if existing is None:
            existing = FileStateEntry(path=path)
            self._entries[path] = existing

        if tool_name in ("read", "glob"):
            if existing.first_read_turn is None:
                existing.first_read_turn = turn
            existing.last_read_turn = turn
            existing.read_count += 1
        elif tool_name in ("edit", "write"):
            existing.last_write_turn = turn
            existing.write_count += 1

        if content_hash:
            old_hash = existing.content_hash
            existing.content_hash = content_hash
            existing.reverts_to_prior_hash = content_hash in existing._content_hashes
            if old_hash:
                existing._content_hashes.add(old_hash)


# ---------------------------------------------------------------------------
# SessionTree
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SessionEntry:
    session_id: str = ""
    parent_id: str | None = None
    purpose: str | None = None
    concurrent_siblings: int = 0
    exit_reason: str | None = None


class SessionTree:
    def __init__(self) -> None:
        self._entries: dict[str, SessionEntry] = {}

    def get(self, session_id: str) -> SessionEntry | None:
        return self._entries.get(session_id)

    def record_spawn(self, session_id: str, entry: SessionEntry) -> None:
        self._entries[session_id] = entry

    def concurrent_children(self, parent_id: str) -> int:
        return sum(
            1 for e in self._entries.values()
            if e.parent_id == parent_id and e.exit_reason is None
        )


# ---------------------------------------------------------------------------
# State aggregate — all tables in one object
# ---------------------------------------------------------------------------


class PolicyState:
    """Holds all state tables for one session."""

    def __init__(self) -> None:
        self.tool_log: RollingLog[ToolLogEntry] = RollingLog(500)
        self.file_state = FileState()
        self.turn_summary: IndexedTable[int, dict[str, object]] = IndexedTable()
        self.session_tree = SessionTree()
        self.error_log: RollingLog[ToolLogEntry] = RollingLog(200)
        self.context_state: IndexedTable[int, dict[str, object]] = IndexedTable()
        self.effect_log: RollingLog[EffectRecord] = RollingLog(500)
        self._turn_count: int = 0
        self._repeat_counter: Counter[tuple[str, str]] = Counter()

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def advance_turn(self) -> None:
        self._turn_count += 1

    def record_tool_call(
        self,
        tool_name: str,
        args: ToolArgs,
        result: dict[str, object] | None = None,
    ) -> None:
        """Record a completed tool call into tool_log + file_state."""
        ah = args_hash(args)
        path = str(args.get("path") or args.get("file_path") or "")
        cmd = str(args.get("cmd") or args.get("command") or "")
        error: str | None = None
        exit_code: int | None = None
        if result:
            error = str(result["error"]) if "error" in result else None
            raw_code = result.get("exit_code")
            exit_code = int(str(raw_code)) if raw_code is not None else None

        key = (tool_name, ah)
        repeat_count = self._repeat_counter[key]
        self._repeat_counter[key] += 1

        entry = ToolLogEntry(
            turn=self._turn_count,
            tool=tool_name,
            args_hash=ah,
            path=path or None,
            cmd=cmd or None,
            exit_code=exit_code,
            error=error,
            error_fingerprint=fingerprint(error),
            error_category=_classify_error(error),
            duration_ms=0,
            result_length=len(str(result)) if result else 0,
            is_repeat=repeat_count > 0,
            repeat_count=repeat_count,
        )
        self.tool_log.append(entry)

        if path and tool_name in ("read", "edit", "write", "glob"):
            self.file_state.record_file_op(path, tool_name, self._turn_count)

        if error:
            err_entry = ToolLogEntry(
                turn=self._turn_count,
                tool=tool_name,
                error=error,
                error_fingerprint=fingerprint(error),
                error_category=_classify_error(error),
            )
            self.error_log.append(err_entry)
