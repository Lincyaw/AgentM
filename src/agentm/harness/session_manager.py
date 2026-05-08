"""Tree-shaped session storage, context reconstruction, and persistence.

Mirrors pi-mono's session model closely enough for AgentM's v2 harness:
append-only entries with parent pointers, a movable active leaf, branch-aware
context reconstruction, and optional JSONL persistence.

Layer purity: stdlib + ``agentm.core.abi`` only.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Self

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from agentm.core.abi.session import (
    CURRENT_SESSION_VERSION,
    SessionContext,
    SessionEntry,
    SessionHeader,
    SessionTreeNode,
    branch_summary_entry,
    compaction_entry,
    message_entry,
)


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> float:
    return time.time()


def _serialize_payload(payload: Any) -> Any:
    if is_dataclass(payload) and not isinstance(payload, type):
        return {f.name: _serialize_payload(getattr(payload, f.name)) for f in fields(payload)}
    if isinstance(payload, list):
        return [_serialize_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return [_serialize_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {str(k): _serialize_payload(v) for k, v in payload.items()}
    if isinstance(payload, bytes):
        return {"__bytes__": list(payload)}
    return payload


def _deserialize_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        if isinstance(payload, list):
            return [_deserialize_payload(item) for item in payload]
        return payload

    if "__bytes__" in payload:
        raw = payload["__bytes__"]
        if isinstance(raw, list):
            return bytes(int(item) for item in raw)
        return b""

    role = payload.get("role")
    if role == "user":
        return UserMessage(
            role="user",
            content=_deserialize_user_blocks(payload.get("content", [])),
            timestamp=float(payload.get("timestamp", 0.0)),
        )
    if role == "assistant":
        return AssistantMessage(
            role="assistant",
            content=_deserialize_assistant_blocks(payload.get("content", [])),
            timestamp=float(payload.get("timestamp", 0.0)),
            stop_reason=payload.get("stop_reason"),
            usage=_deserialize_usage(payload.get("usage")),
        )
    if role == "tool_result":
        return ToolResultMessage(
            role="tool_result",
            content=_deserialize_tool_result_blocks(payload.get("content", [])),
            timestamp=float(payload.get("timestamp", 0.0)),
        )

    return {str(k): _deserialize_payload(v) for k, v in payload.items()}


def _deserialize_usage(payload: Any) -> Usage | None:
    if not isinstance(payload, dict):
        return None
    return Usage(
        input_tokens=int(payload.get("input_tokens", 0)),
        output_tokens=int(payload.get("output_tokens", 0)),
        cache_read=int(payload.get("cache_read", 0)),
        cache_write=int(payload.get("cache_write", 0)),
    )


def _deserialize_user_blocks(payload: Any) -> list[TextContent | ImageContent]:
    if not isinstance(payload, list):
        return []
    blocks: list[TextContent | ImageContent] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") == "text":
            blocks.append(TextContent(type="text", text=str(raw.get("text", ""))))
        elif raw.get("type") == "image":
            blocks.append(
                ImageContent(
                    type="image",
                    data=_deserialize_payload(raw.get("data", {"__bytes__": []})),
                    mime_type=str(raw.get("mime_type", "application/octet-stream")),
                )
            )
    return blocks


def _deserialize_assistant_blocks(payload: Any) -> list[TextContent | ToolCallBlock | ThinkingBlock]:
    if not isinstance(payload, list):
        return []
    blocks: list[TextContent | ToolCallBlock | ThinkingBlock] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("type")
        if kind == "text":
            blocks.append(TextContent(type="text", text=str(raw.get("text", ""))))
        elif kind == "tool_call":
            args = raw.get("arguments", {})
            blocks.append(
                ToolCallBlock(
                    type="tool_call",
                    id=str(raw.get("id", "")),
                    name=str(raw.get("name", "")),
                    arguments=args if isinstance(args, dict) else {},
                )
            )
        elif kind == "thinking":
            signature = raw.get("signature")
            blocks.append(
                ThinkingBlock(
                    type="thinking",
                    text=str(raw.get("text", "")),
                    signature=str(signature) if isinstance(signature, str) else None,
                )
            )
    return blocks


def _deserialize_tool_result_blocks(payload: Any) -> list[ToolResultBlock]:
    if not isinstance(payload, list):
        return []
    blocks: list[ToolResultBlock] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        content = _deserialize_user_blocks(raw.get("content", []))
        blocks.append(
            ToolResultBlock(
                type="tool_result",
                tool_call_id=str(raw.get("tool_call_id", "")),
                content=content,
                is_error=bool(raw.get("is_error", False)),
            )
        )
    return blocks


def _entry_to_record(entry: SessionEntry) -> dict[str, Any]:
    return {
        "type": entry.type,
        "id": entry.id,
        "parent_id": entry.parent_id,
        "timestamp": entry.timestamp,
        "payload": _serialize_payload(entry.payload),
    }


def _entry_from_record(record: dict[str, Any]) -> SessionEntry:
    return SessionEntry(
        type=str(record["type"]),
        id=str(record["id"]),
        parent_id=record.get("parent_id"),
        timestamp=float(record.get("timestamp", 0.0)),
        payload=_deserialize_payload(record.get("payload")),
    )


def _header_to_record(header: SessionHeader) -> dict[str, Any]:
    return asdict(header)


def _header_from_record(record: dict[str, Any]) -> SessionHeader:
    return SessionHeader(
        type="session",
        version=int(record.get("version", CURRENT_SESSION_VERSION)),
        id=str(record["id"]),
        timestamp=float(record.get("timestamp", 0.0)),
        cwd=str(record.get("cwd", "")),
        parent_session=(
            str(record["parent_session"]) if record.get("parent_session") is not None else None
        ),
    )


def _entry_text(message: AgentMessage) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []):
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
        elif getattr(block, "type", None) == "tool_call":
            parts.append(f"tool:{getattr(block, 'name', '?')}")
        elif getattr(block, "type", None) == "tool_result":
            parts.append("tool_result")
    return " ".join(parts)


def _branch_summary_message(summary: str, timestamp: float) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=f"Branch summary: {summary}")],
        timestamp=timestamp,
        stop_reason="end_turn",
    )


def _compaction_summary_message(summary: str, timestamp: float) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=summary)],
        timestamp=timestamp,
        stop_reason="end_turn",
    )


class SessionManager:
    """Pi-style append-only session tree with an active leaf pointer."""

    def __init__(
        self,
        *,
        cwd: str = "",
        session_dir: Path | None = None,
        session_file: Path | None = None,
        persist: bool = False,
        parent_session: str | None = None,
    ) -> None:
        self._cwd = cwd
        self._session_dir = session_dir
        self._session_file = session_file
        self._persist = persist
        self._header: SessionHeader | None = None
        self._entries: dict[str, SessionEntry] = {}
        self._order: list[str] = []
        self._leaf_id: str | None = None

        if self._persist:
            if self._session_dir is None and self._session_file is not None:
                self._session_dir = self._session_file.parent
            if self._session_dir is not None:
                self._session_dir.mkdir(parents=True, exist_ok=True)

        if self._persist and self._session_file is not None and self._session_file.exists():
            self._load()
        else:
            self.new_session(parent_session=parent_session)

    @property
    def session_file(self) -> Path | None:
        """Path to the on-disk JSONL log when ``persist=True``; ``None`` otherwise."""
        return self._session_file

    # ------------------------------------------------------------------
    # Lifecycle / persistence
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, cwd: str, session_dir: Path | None = None) -> Self:
        directory = session_dir or cls.default_session_dir(cwd)
        return cls(cwd=cwd, session_dir=directory, persist=True)

    @classmethod
    def open(
        cls,
        path: str | Path,
        session_dir: Path | None = None,
        cwd_override: str | None = None,
    ) -> Self:
        file_path = Path(path)
        inferred_cwd = cwd_override
        if inferred_cwd is None and file_path.exists():
            with file_path.open("r", encoding="utf-8") as handle:
                first_line = handle.readline().strip()
            if first_line:
                try:
                    record = json.loads(first_line)
                    inferred_cwd = str(record.get("cwd", ""))
                except json.JSONDecodeError:
                    inferred_cwd = ""
        return cls(
            cwd=inferred_cwd or "",
            session_dir=session_dir or file_path.parent,
            session_file=file_path,
            persist=True,
        )

    @classmethod
    def continue_recent(cls, cwd: str, session_dir: Path | None = None) -> Self:
        directory = session_dir or cls.default_session_dir(cwd)
        latest = cls._find_most_recent(directory)
        if latest is None:
            return cls.create(cwd, directory)
        return cls.open(latest, session_dir=directory, cwd_override=cwd)

    @classmethod
    def in_memory(cls, cwd: str = "") -> Self:
        return SessionManager(cwd=cwd, persist=False)  # type: ignore[return-value]

    @staticmethod
    def default_session_dir(cwd: str) -> Path:
        safe = cwd.strip(os.sep).replace(os.sep, "-") or "root"
        return Path.home() / ".agentm" / "sessions" / f"--{safe}--"

    @staticmethod
    def _find_most_recent(session_dir: Path) -> Path | None:
        if not session_dir.exists():
            return None
        files = sorted(
            (path for path in session_dir.glob("*.jsonl") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return files[0] if files else None

    def new_session(
        self,
        *,
        id: str | None = None,
        parent_session: str | None = None,
    ) -> str | None:
        self._entries = {}
        self._order = []
        self._leaf_id = None
        self._header = SessionHeader(
            type="session",
            version=CURRENT_SESSION_VERSION,
            id=id or _new_id(),
            timestamp=_now(),
            cwd=self._cwd,
            parent_session=parent_session,
        )
        if self._persist and self._session_file is None:
            assert self._session_dir is not None
            stamp = f"{self._header.timestamp:.6f}".replace(".", "-")
            self._session_file = self._session_dir / f"{stamp}_{self._header.id}.jsonl"
        self._rewrite_file()
        return str(self._session_file) if self._session_file is not None else None

    def _load(self) -> None:
        assert self._session_file is not None
        self._entries = {}
        self._order = []
        self._leaf_id = None
        self._header = None
        with self._session_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") == "session":
                    self._header = _header_from_record(record)
                    self._cwd = self._header.cwd
                    continue
                entry = _entry_from_record(record)
                self._entries[entry.id] = entry
                self._order.append(entry.id)
                self._leaf_id = entry.id
        if self._header is None:
            self.new_session()

    def _rewrite_file(self) -> None:
        if not self._persist or self._session_file is None or self._header is None:
            return
        records = [_header_to_record(self._header)]
        records.extend(_entry_to_record(self._entries[entry_id]) for entry_id in self._order)
        with self._session_file.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, default=str))
                handle.write("\n")

    def _append_record(self, entry: SessionEntry) -> None:
        self._entries[entry.id] = entry
        self._order.append(entry.id)
        self._leaf_id = entry.id
        if not self._persist or self._session_file is None:
            return
        with self._session_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_entry_to_record(entry), default=str))
            handle.write("\n")

    # ------------------------------------------------------------------
    # Append / mutation
    # ------------------------------------------------------------------

    def append(self, entry: SessionEntry) -> None:
        if entry.id in self._entries:
            raise ValueError(f"duplicate entry id: {entry.id}")
        if entry.parent_id is not None and entry.parent_id not in self._entries:
            raise ValueError(
                f"parent_id {entry.parent_id!r} for entry {entry.id!r} not found"
            )
        self._append_record(entry)

    def append_message(self, message: AgentMessage) -> SessionEntry:
        entry = message_entry(message, self._leaf_id)
        self.append(entry)
        return entry

    def append_custom_entry(self, type: str, payload: Any) -> SessionEntry:
        entry = SessionEntry(
            type=type,
            id=_new_id(),
            parent_id=self._leaf_id,
            timestamp=_now(),
            payload=payload,
        )
        self.append(entry)
        return entry

    def branch(self, entry_id: str) -> None:
        if entry_id not in self._entries:
            raise KeyError(f"unknown entry id: {entry_id}")
        self._leaf_id = entry_id

    def navigate_to(self, leaf_id: str) -> None:
        self.branch(leaf_id)

    def reset_leaf(self) -> None:
        self._leaf_id = None

    def branch_with_summary(
        self,
        branch_from_id: str | None,
        summary: str,
        *,
        details: Any = None,
    ) -> str:
        if branch_from_id is None:
            self.reset_leaf()
        else:
            self.branch(branch_from_id)
        entry = branch_summary_entry(
            summary,
            self._leaf_id,
            from_id=branch_from_id,
            details=details,
        )
        self.append(entry)
        return entry.id

    def delete_session_file(self) -> None:
        if self._session_file is not None and self._session_file.exists():
            self._session_file.unlink()

    def fork_at(self, entry_id: str) -> Self:
        if entry_id not in self._entries:
            raise KeyError(f"unknown entry id: {entry_id}")
        fork = self.in_memory(self._cwd)
        fork._header = self._header
        fork._entries = dict(self._entries)
        fork._order = list(self._order)
        fork._leaf_id = entry_id
        return fork

    def create_branched_session(self, leaf_id: str) -> str | None:
        branch = self.get_branch(leaf_id)
        if not branch:
            raise KeyError(f"unknown entry id: {leaf_id}")

        fork = self.create(self._cwd, self._session_dir or self.default_session_dir(self._cwd))
        if self._session_file is not None and fork._header is not None:
            fork._header = SessionHeader(
                type="session",
                version=CURRENT_SESSION_VERSION,
                id=fork._header.id,
                timestamp=fork._header.timestamp,
                cwd=self._cwd,
                parent_session=str(self._session_file),
            )
            fork._rewrite_file()
        prev_new_id: str | None = None
        for original in branch:
            copied = SessionEntry(
                type=original.type,
                id=_new_id(),
                parent_id=prev_new_id,
                timestamp=original.timestamp,
                payload=original.payload,
            )
            fork.append(copied)
            prev_new_id = copied.id
        return str(fork._session_file) if fork._session_file is not None else None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_cwd(self) -> str:
        return self._cwd

    def get_session_dir(self) -> Path | None:
        return self._session_dir

    def get_session_id(self) -> str:
        return self._header.id if self._header is not None else ""

    def get_session_file(self) -> str | None:
        return str(self._session_file) if self._session_file is not None else None

    def get_header(self) -> SessionHeader | None:
        return self._header

    def is_persisted(self) -> bool:
        return self._persist

    def get_leaf_id(self) -> str | None:
        return self._leaf_id

    def get_leaf_entry(self) -> SessionEntry | None:
        if self._leaf_id is None:
            return None
        return self._entries.get(self._leaf_id)

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        return self._entries.get(entry_id)

    def find(self, entry_id: str) -> SessionEntry | None:
        return self.get_entry(entry_id)

    def get_children(self, parent_id: str) -> list[SessionEntry]:
        children = [entry for entry in self.get_entries() if entry.parent_id == parent_id]
        return sorted(children, key=lambda item: item.timestamp)

    def get_entries(self) -> list[SessionEntry]:
        return [self._entries[entry_id] for entry_id in self._order]

    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]:
        path: list[SessionEntry] = []
        cursor_id = self._leaf_id if from_id is None else from_id
        while cursor_id is not None:
            entry = self._entries.get(cursor_id)
            if entry is None:
                break
            path.append(entry)
            cursor_id = entry.parent_id
        path.reverse()
        return path

    def get_active_branch(self) -> list[SessionEntry]:
        return self.get_branch()

    def get_tree(self) -> list[SessionTreeNode]:
        nodes = {entry.id: SessionTreeNode(entry=entry, children=[]) for entry in self.get_entries()}
        roots: list[SessionTreeNode] = []
        for entry in self.get_entries():
            node = nodes[entry.id]
            if entry.parent_id is None or entry.parent_id not in nodes:
                roots.append(node)
                continue
            nodes[entry.parent_id].children.append(node)
        stack = list(roots)
        while stack:
            current = stack.pop()
            current.children.sort(key=lambda child: child.entry.timestamp)
            for child in current.children:
                child.has_compacted_ancestor = (
                    current.has_compacted_ancestor or current.entry.type == "compaction"
                )
            stack.extend(current.children)
        roots.sort(key=lambda node: node.entry.timestamp)
        return roots

    def build_session_context(self, leaf_id: str | None = None) -> SessionContext:
        path = self.get_branch(leaf_id)
        if not path:
            return SessionContext(messages=[])

        latest_compaction: SessionEntry | None = None
        for entry in path:
            if entry.type == "compaction":
                latest_compaction = entry

        messages: list[AgentMessage] = []

        def append_materialized(entry: SessionEntry) -> None:
            if entry.type == "message" and isinstance(entry.payload, (UserMessage, AssistantMessage, ToolResultMessage)):
                messages.append(entry.payload)
            elif entry.type == "branch_summary":
                payload = entry.payload if isinstance(entry.payload, dict) else {}
                summary = payload.get("summary")
                if isinstance(summary, str) and summary:
                    messages.append(_branch_summary_message(summary, entry.timestamp))

        if latest_compaction is None:
            for entry in path:
                append_materialized(entry)
            return SessionContext(messages=messages)

        details = latest_compaction.payload if isinstance(latest_compaction.payload, dict) else {}
        summary = details.get("summary")
        if isinstance(summary, str) and summary:
            messages.append(_compaction_summary_message(summary, latest_compaction.timestamp))

        first_kept_id = details.get("first_kept_entry_id") or details.get("firstKeptEntryId")
        compaction_index = path.index(latest_compaction)
        first_kept_index: int | None = None
        if isinstance(first_kept_id, str):
            first_kept_index = next(
                (index for index, entry in enumerate(path[:compaction_index]) if entry.id == first_kept_id),
                None,
            )

        if first_kept_index is None:
            first_kept_index = compaction_index

        for entry in path[first_kept_index:compaction_index]:
            append_materialized(entry)

        for entry in path[compaction_index + 1 :]:
            append_materialized(entry)
        return SessionContext(messages=messages)

    def get_messages(self) -> list[AgentMessage]:
        return self.build_session_context().messages


class InMemorySessionManager(SessionManager):
    def __init__(
        self,
        *,
        cwd: str = "",
        entries: dict[str, SessionEntry] | None = None,
        active_leaf: str | None = None,
    ) -> None:
        super().__init__(cwd=cwd, persist=False)
        if entries is not None:
            self._entries = dict(entries)
            self._order = list(entries.keys())
            self._leaf_id = active_leaf


class JsonlSessionManager(SessionManager):
    def __init__(self, path: Path, *, cwd: str = "") -> None:
        super().__init__(cwd=cwd, session_dir=path.parent, session_file=path, persist=True)


__all__ = [
    "CURRENT_SESSION_VERSION",
    "InMemorySessionManager",
    "JsonlSessionManager",
    "SessionContext",
    "SessionEntry",
    "SessionHeader",
    "SessionManager",
    "SessionTreeNode",
    "branch_summary_entry",
    "compaction_entry",
    "message_entry",
]
