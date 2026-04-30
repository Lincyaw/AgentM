"""Session entry tree + InMemory and JSONL implementations.

Implements §5 (SessionManager) of ``.claude/designs/extension-as-scenario.md``
and §3.3 (Session Persistence) of ``.claude/designs/pluggable-architecture.md``.

Storage model: a single append-only stream of immutable :class:`SessionEntry`
records. Each entry has a ``parent_id`` so the stream is a tree (forks,
branches, compaction-replacement). One pointer (``_active_leaf``) marks the
"current" leaf; ``get_active_branch`` walks parent links from that leaf back
to the root.

Persistence: ``JsonlSessionManager`` writes one JSON line per entry. Payloads
that contain dataclasses are serialized via ``asdict``. On reload the payload
field is left as a plain dict — this is a Phase 1 limitation: we do not yet
have a typed serializer registry that can reconstruct ``AgentMessage``
variants. Tests assert metadata round-trip only; payload-shape tests for
JSONL deliberately accept dicts.

Hard rule (see ``.claude/designs/pluggable-architecture.md``): this module
imports only stdlib + ``agentm.core.kernel``.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agentm.core.kernel import AgentMessage


# --- Entry ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionEntry:
    """One node in the session entry tree. Immutable by contract."""

    type: str
    id: str
    parent_id: str | None
    timestamp: float
    payload: Any


def _new_id() -> str:
    """Generate a fresh entry id. Uses ``uuid4`` because stdlib lacks uuid7
    on Python <3.14; identifiers are still globally unique."""

    return uuid.uuid4().hex


def message_entry(msg: AgentMessage, parent_id: str | None) -> SessionEntry:
    """Build a ``type='message'`` entry whose payload is the kernel message."""

    return SessionEntry(
        type="message",
        id=_new_id(),
        parent_id=parent_id,
        timestamp=time.time(),
        payload=msg,
    )


def branch_summary_entry(summary: str, parent_id: str | None) -> SessionEntry:
    """Build a ``type='branch_summary'`` entry recording a fork narrative."""

    return SessionEntry(
        type="branch_summary",
        id=_new_id(),
        parent_id=parent_id,
        timestamp=time.time(),
        payload={"summary": summary},
    )


def compaction_entry(payload: Any, parent_id: str | None) -> SessionEntry:
    """Build a ``type='compaction'`` entry. ``payload`` is opaque to the
    harness; compaction extensions own the schema."""

    return SessionEntry(
        type="compaction",
        id=_new_id(),
        parent_id=parent_id,
        timestamp=time.time(),
        payload=payload,
    )


# --- Protocol ---------------------------------------------------------------


@runtime_checkable
class SessionManager(Protocol):
    """Append-only entry tree with a movable active-leaf pointer."""

    def append(self, entry: SessionEntry) -> None: ...
    def get_active_branch(self) -> list[SessionEntry]: ...
    def get_messages(self) -> list[AgentMessage]: ...
    def fork_at(self, entry_id: str) -> "SessionManager": ...
    def navigate_to(self, leaf_id: str) -> None: ...
    def find(self, entry_id: str) -> SessionEntry | None: ...


# --- In-memory impl ---------------------------------------------------------


class InMemorySessionManager:
    """Reference implementation: a dict of entries + an active-leaf pointer.

    Entries are immutable, so ``fork_at`` may share the underlying entry dict
    between the parent and forked managers without copying. The fork only
    differs in which leaf it considers active and where future appends land.
    """

    def __init__(
        self,
        *,
        entries: dict[str, SessionEntry] | None = None,
        active_leaf: str | None = None,
    ) -> None:
        self._entries: dict[str, SessionEntry] = entries if entries is not None else {}
        self._active_leaf: str | None = active_leaf

    # --- Mutation ---------------------------------------------------------

    def append(self, entry: SessionEntry) -> None:
        if entry.id in self._entries:
            raise ValueError(f"duplicate entry id: {entry.id}")
        # Validate parent reference if any.
        if entry.parent_id is not None and entry.parent_id not in self._entries:
            raise ValueError(
                f"parent_id {entry.parent_id!r} for entry {entry.id!r} not found"
            )
        self._entries[entry.id] = entry
        self._active_leaf = entry.id

    def navigate_to(self, leaf_id: str) -> None:
        if leaf_id not in self._entries:
            raise KeyError(f"unknown entry id: {leaf_id}")
        self._active_leaf = leaf_id

    def fork_at(self, entry_id: str) -> "InMemorySessionManager":
        if entry_id not in self._entries:
            raise KeyError(f"unknown entry id: {entry_id}")
        # Share the same entry store (entries are frozen) so a fork sees the
        # full history; future appends on each manager append to the shared
        # store but carve out separate branches via parent_id.
        return InMemorySessionManager(
            entries=self._entries,
            active_leaf=entry_id,
        )

    # --- Query ------------------------------------------------------------

    def find(self, entry_id: str) -> SessionEntry | None:
        return self._entries.get(entry_id)

    def get_active_branch(self) -> list[SessionEntry]:
        if self._active_leaf is None:
            return []
        chain: list[SessionEntry] = []
        cursor: str | None = self._active_leaf
        while cursor is not None:
            entry = self._entries.get(cursor)
            if entry is None:
                break
            chain.append(entry)
            cursor = entry.parent_id
        chain.reverse()  # root → leaf
        return chain

    def get_messages(self) -> list[AgentMessage]:
        return [
            entry.payload
            for entry in self.get_active_branch()
            if entry.type == "message"
        ]


# --- JSONL impl -------------------------------------------------------------


def _serialize_payload(payload: Any) -> Any:
    """Recursively convert dataclasses to dicts for JSON serialization.

    Lists, tuples, and dicts are walked. Anything else is returned as-is
    (``json.dumps`` with ``default=str`` will stringify the rest)."""

    if is_dataclass(payload) and not isinstance(payload, type):
        return {f.name: _serialize_payload(getattr(payload, f.name)) for f in fields(payload)}
    if isinstance(payload, (list, tuple)):
        return [_serialize_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {str(k): _serialize_payload(v) for k, v in payload.items()}
    return payload


def _entry_to_jsonl(entry: SessionEntry) -> str:
    record = {
        "type": entry.type,
        "id": entry.id,
        "parent_id": entry.parent_id,
        "timestamp": entry.timestamp,
        "payload": _serialize_payload(entry.payload),
    }
    return json.dumps(record, default=str)


def _entry_from_record(record: dict[str, Any]) -> SessionEntry:
    return SessionEntry(
        type=str(record["type"]),
        id=str(record["id"]),
        parent_id=record.get("parent_id"),
        timestamp=float(record.get("timestamp", 0.0)),
        # Payload stays as a plain dict/list/str on reload — see module
        # docstring for the Phase 1 limitation note.
        payload=record.get("payload"),
    )


class JsonlSessionManager:
    """JSONL-backed session manager.

    Writes are append-only; one JSON line per entry. On construction, any
    pre-existing file at ``path`` is read and entries are reconstructed. The
    active leaf defaults to the last entry in file order; callers may change
    it with ``navigate_to``.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, SessionEntry] = {}
        self._order: list[str] = []
        self._active_leaf: str | None = None

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._load()

    # --- Mutation ---------------------------------------------------------

    def append(self, entry: SessionEntry) -> None:
        if entry.id in self._entries:
            raise ValueError(f"duplicate entry id: {entry.id}")
        if entry.parent_id is not None and entry.parent_id not in self._entries:
            raise ValueError(
                f"parent_id {entry.parent_id!r} for entry {entry.id!r} not found"
            )
        self._entries[entry.id] = entry
        self._order.append(entry.id)
        self._active_leaf = entry.id
        with self._path.open("a", encoding="utf-8") as f:
            f.write(_entry_to_jsonl(entry))
            f.write("\n")

    def navigate_to(self, leaf_id: str) -> None:
        if leaf_id not in self._entries:
            raise KeyError(f"unknown entry id: {leaf_id}")
        self._active_leaf = leaf_id

    def fork_at(self, entry_id: str) -> "InMemorySessionManager":
        """Return an in-memory manager forked at ``entry_id``.

        Forking a JSONL manager intentionally returns an in-memory branch:
        forks are exploratory and shouldn't write to the canonical session
        file. If callers need persistence for a fork, they construct a new
        ``JsonlSessionManager`` against a fresh path.
        """

        if entry_id not in self._entries:
            raise KeyError(f"unknown entry id: {entry_id}")
        return InMemorySessionManager(
            entries=dict(self._entries),
            active_leaf=entry_id,
        )

    # --- Query ------------------------------------------------------------

    def find(self, entry_id: str) -> SessionEntry | None:
        return self._entries.get(entry_id)

    def get_active_branch(self) -> list[SessionEntry]:
        if self._active_leaf is None:
            return []
        chain: list[SessionEntry] = []
        cursor: str | None = self._active_leaf
        while cursor is not None:
            entry = self._entries.get(cursor)
            if entry is None:
                break
            chain.append(entry)
            cursor = entry.parent_id
        chain.reverse()
        return chain

    def get_messages(self) -> list[AgentMessage]:
        return [
            entry.payload
            for entry in self.get_active_branch()
            if entry.type == "message"
        ]

    # --- Internal ---------------------------------------------------------

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as f:
            lines: Iterable[str] = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            entry = _entry_from_record(record)
            self._entries[entry.id] = entry
            self._order.append(entry.id)
        if self._order:
            self._active_leaf = self._order[-1]


__all__ = [
    "InMemorySessionManager",
    "JsonlSessionManager",
    "SessionEntry",
    "SessionManager",
    "branch_summary_entry",
    "compaction_entry",
    "message_entry",
]
