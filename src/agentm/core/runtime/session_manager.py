"""Tree-shaped session storage, context reconstruction, and persistence.

Append-only entries with parent pointers, a movable active leaf,
branch-aware context reconstruction, and OTLP/JSON persistence via
the session telemetry SDK.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from agentm.core.abi import AgentMessage
from agentm.core.abi.events import (
    MessageAppendedEvent,
    SessionHeaderEmittedEvent,
)
from agentm.core.abi.session import (
    CURRENT_SESSION_VERSION,
    ENTRY_MATERIALIZERS,
    ENTRY_TYPE_COMPACTION,
    ENTRY_TYPE_MESSAGE,
    ENTRY_TYPE_TURN_COMMITTED,
    SessionContext,
    SessionEntry,
    SessionHeader,
    SessionTreeNode,
    branch_summary_entry,
    compaction_entry,
    message_entry,
)
from agentm.core.lib.message_codec import deserialize_payload, serialize_payload
from agentm.core.observability.otel_export import setup_session_telemetry
from agentm.core.lib.trace_reader import TraceReader
from opentelemetry._logs import SeverityNumber

if TYPE_CHECKING:
    from agentm.core.abi.bus import EventBus


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> float:
    return time.time()


def _entry_to_record(entry: SessionEntry) -> dict[str, Any]:
    return {
        "type": entry.type,
        "id": entry.id,
        "parent_id": entry.parent_id,
        "timestamp": entry.timestamp,
        "payload": serialize_payload(entry.payload),
    }


def _entry_from_record(record: dict[str, Any]) -> SessionEntry:
    return SessionEntry(
        type=str(record["type"]),
        id=str(record["id"]),
        parent_id=record.get("parent_id"),
        timestamp=float(record.get("timestamp", 0.0)),
        payload=deserialize_payload(record.get("payload")),
    )


def _header_to_record(header: SessionHeader) -> dict[str, Any]:
    return asdict(header)


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """Reduce *value* to JSON-serializable primitives, replacing live
    objects with their repr.

    The session header is persisted (and deep-copied by ``asdict``), so
    everything stored in it must be a plain value. Resolved configs can
    carry live objects — e.g. a child session inherits the parent's
    instantiated provider, whose stream fn holds an httpx client with
    asyncio futures that neither deepcopy nor JSON can handle.
    """
    if _depth > 8:
        return repr(value)[:200]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, _depth + 1) for v in value]
    return repr(value)[:200]


def _infer_cwd_from_log(file_path: Path) -> str | None:
    """Best-effort scan for ``cwd`` on the first ``agentm.session.header``
    record in an OTLP/JSON log. Used by :meth:`JsonlSessionManager.open`
    when the caller does not supply ``cwd_override``.
    """
    for record in TraceReader(file_path).iter_log_records(
        name="agentm.session.header"
    ):
        if isinstance(record.body, dict):
            cwd_value = record.body.get("cwd")
            if isinstance(cwd_value, str):
                return cwd_value
    return None


def _header_from_record(record: dict[str, Any]) -> SessionHeader:
    raw_config = record.get("config")
    return SessionHeader(
        type="session",
        version=int(record.get("version", CURRENT_SESSION_VERSION)),
        id=str(record["id"]),
        timestamp=float(record.get("timestamp", 0.0)),
        cwd=str(record.get("cwd", "")),
        parent_session=(
            str(record["parent_session"])
            if record.get("parent_session") is not None
            else None
        ),
        config=raw_config if isinstance(raw_config, dict) else None,
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
        self._bus: "EventBus | None" = None
        self._pending_emits: list[tuple[str, Any]] = []

        if self._persist:
            if self._session_dir is None and self._session_file is not None:
                self._session_dir = self._session_file.parent
            if self._session_dir is not None:
                self._session_dir.mkdir(parents=True, exist_ok=True)

        if (
            self._persist
            and self._session_file is not None
            and self._session_file.exists()
        ):
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
    def from_records(
        cls,
        header: SessionHeader,
        entries: list[dict[str, Any]],
        cwd: str = "",
    ) -> Self:
        """Reconstruct session state from pre-loaded records (e.g. ClickHouse)."""
        mgr = cls(cwd=cwd or header.cwd, persist=False)
        mgr._header = header
        mgr._entries = {}
        mgr._order = []
        mgr._leaf_id = None
        for record in entries:
            entry = _entry_from_record(record)
            mgr._entries[entry.id] = entry
            mgr._order.append(entry.id)
            mgr._leaf_id = entry.id
        mgr._truncate_to_last_boundary()
        return mgr

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
            inferred_cwd = _infer_cwd_from_log(file_path) or ""
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
        """Per-cwd session log directory (``<cwd>/.agentm/observability/``)."""

        from agentm.core.observability.otel_export import resolve_observability_dir

        return resolve_observability_dir(cwd)

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

    # ------------------------------------------------------------------
    # Bus wiring
    # ------------------------------------------------------------------

    def attach_bus(self, bus: "EventBus") -> None:
        """Wire the SessionManager to an EventBus and flush any buffered emits.

        Sessions are constructed (and ``new_session`` emits its header) before
        the factory builds the bus or installs the observability writer.
        Anything we'd have emitted in the meantime accumulates in
        ``_pending_emits`` and gets replayed in order once the bus is ready,
        so the resulting JSONL contains the header before any messages.

        Until ``attach_bus`` is called we also direct-write each pending emit
        to ``_session_file`` (when persistent), so a standalone
        ``SessionManager.create(...)`` — used by replay tools and unit
        fixtures that never construct a full AgentSession — still leaves a
        real merged log on disk. Once a bus arrives, that direct-write path
        is retired and the file is owned by the observability sink.
        """

        self._bus = bus
        pending, self._pending_emits = self._pending_emits, []
        # Persistent sessions never buffer (``_emit`` direct-writes), so
        # ``pending`` is non-empty only for in-memory sessions whose state
        # would otherwise be lost. Replay through the bus so any observer
        # (typically the test harness) sees the header + messages in order.
        for channel, event in pending:
            bus.emit_sync(channel, event)

    def _emit(self, channel: str, event: Any) -> None:
        if self._bus is not None:
            self._bus.emit_sync(channel, event)
            return
        # No bus yet. Two paths:
        # * Persistent: write the row straight to disk so bus-less callers
        #   (replay tools, unit fixtures) still see a real merged log on
        #   disk. Don't also buffer for bus replay — when ``attach_bus``
        #   fires, observability will open the same file in append mode
        #   and start writing later events; replaying these would
        #   duplicate rows.
        # * In-memory: buffer so ``attach_bus`` can replay through the
        #   bus once observability is subscribed. Nothing else carries
        #   the trajectory in that mode.
        if self._persist and self._session_file is not None:
            self._direct_append_event(channel, event)
            return
        self._pending_emits.append((channel, event))

    def _direct_append_event(self, channel: str, event: Any) -> None:
        """Append a single SessionManager event to the merged log directly.

        Used when no ``EventBus`` is attached yet (replay tools, unit
        fixtures that build a ``SessionManager`` without going through the
        ``AgentSession`` factory). Writes OTLP/JSON ndjson through the same
        SDK plumbing the observability atom uses, so on-disk shape stays
        identical to the bus-driven path.
        """
        if channel == SessionHeaderEmittedEvent.CHANNEL:
            assert isinstance(event, SessionHeaderEmittedEvent)
            event_name = "agentm.session.header"
            body = event.record
            attributes: dict[str, Any] = {
                "agentm.session.id": str(event.record.get("id", "")),
                "agentm.session.header.id": str(event.record.get("id", "")),
            }
        elif channel == MessageAppendedEvent.CHANNEL:
            assert isinstance(event, MessageAppendedEvent)
            event_name = "agentm.message.appended"
            body = event.record
            attributes = {
                "agentm.session.id": (
                    self._header.id if self._header is not None else ""
                ),
                "agentm.message.id": str(event.record.get("id", "")),
                "agentm.message.parent_id": str(
                    event.record.get("parent_id", "") or ""
                ),
                "agentm.message.type": str(event.record.get("type", "")),
            }
        else:
            return
        assert self._session_file is not None
        # Construct a transient SessionTelemetry, emit one log record, and
        # tear it down. The SDK setup cost is one-time; replay-tool callers
        # aren't on a hot path. ``file_path`` override pins the write to
        # the SessionManager's exact configured file regardless of cwd
        # layout (tests sometimes use bespoke paths).
        telemetry = setup_session_telemetry(
            session_id=(self._header.id if self._header is not None else "session"),
            cwd=Path(self._cwd) if self._cwd else self._session_file.parent,
            scenario_name=None,
            file_path=self._session_file,
        )
        try:
            telemetry.logger.emit(
                body=body,
                severity_number=SeverityNumber.INFO,
                severity_text="INFO",
                event_name=event_name,
                attributes=attributes,
            )
        finally:
            telemetry.shutdown()

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
        if self._persist and self._session_file is None and self._session_dir is not None:
            self._session_file = self._session_dir / f"{self._header.id}.jsonl"
        self._emit(
            SessionHeaderEmittedEvent.CHANNEL,
            SessionHeaderEmittedEvent(record=_header_to_record(self._header)),
        )
        return str(self._session_file) if self._session_file is not None else None

    def set_session_config(self, config: dict[str, Any]) -> None:
        """Persist resolved session config (scenario, provider, extensions, env).

        Replaces the header with an updated copy and re-emits it so the
        JSONL on disk carries the config. Called by the session factory
        after extension resolution — framework-level, no atom involvement.

        The config is scrubbed to JSON-safe primitives first: resolved
        configs may carry live objects (a child session inherits the
        parent's instantiated provider), which must never reach the
        persisted, ``asdict``-copied header.
        """
        from dataclasses import replace as _replace

        if self._header is None:
            return
        self._header = _replace(self._header, config=_json_safe(config))
        self._emit(
            SessionHeaderEmittedEvent.CHANNEL,
            SessionHeaderEmittedEvent(record=_header_to_record(self._header)),
        )

    def _load(self) -> None:
        """Reconstruct in-memory state from the merged OTLP/JSON event log.

        Each line on disk is a self-contained OTLP element — either a
        ``resourceSpans`` wrapper or a ``resourceLogs`` wrapper (PR-A wire
        format). We walk only the log records: SessionManager state lives
        in ``agentm.session.header`` (body = SessionHeader dict) and
        ``agentm.message.appended`` (body = the SessionEntry record dict).
        Every other event name and every span line is ignored on load —
        spans carry trace data, not persisted session state.
        """

        assert self._session_file is not None
        self._entries = {}
        self._order = []
        self._leaf_id = None
        self._header = None
        latest_header: SessionHeader | None = None
        # Skip span lines: SessionManager state only lives in log records.
        # ``iter_log_records`` walks them in file order; the latest header
        # wins, message.appended entries accumulate in their on-disk order.
        for record in TraceReader(self._session_file).iter_log_records():
            event_name = record.event_name
            body = record.body
            if event_name == "agentm.session.header" and isinstance(body, dict):
                latest_header = _header_from_record(body)
            elif event_name == "agentm.message.appended" and isinstance(body, dict):
                entry = _entry_from_record(body)
                self._entries[entry.id] = entry
                self._order.append(entry.id)
                self._leaf_id = entry.id
        if latest_header is not None:
            self._header = latest_header
            self._cwd = latest_header.cwd
        else:
            self.new_session()
        self._truncate_to_last_boundary()

    def _append_record(self, entry: SessionEntry) -> None:
        self._entries[entry.id] = entry
        self._order.append(entry.id)
        self._leaf_id = entry.id
        self._emit(
            MessageAppendedEvent.CHANNEL,
            MessageAppendedEvent(record=_entry_to_record(entry)),
        )

    def _direct_write_merged_log(self) -> None:
        """Rewrite the entire merged log to ``_session_file`` in OTLP/JSON shape.

        Used by :meth:`create_branched_session` — branched copies are
        constructed without an attached EventBus or observability writer, so
        the bus-driven path can't reach them. We open a transient
        :class:`SessionTelemetry` pinned at the exact ``_session_file``
        path, emit one ``agentm.session.header`` log record followed by one
        ``agentm.message.appended`` per entry, and shut down — the result
        loads through :meth:`_load` exactly like a normal session log.

        Truncates first so re-running on an existing file produces a clean
        snapshot rather than appending duplicates.
        """

        if not self._persist or self._session_file is None or self._header is None:
            return
        # Truncate so the transient SDK exporter (append-mode) writes a
        # clean snapshot. Done explicitly here because the SDK's
        # FileSpanExporter / FileLogExporter always open in append.
        self._session_file.parent.mkdir(parents=True, exist_ok=True)
        self._session_file.write_text("", encoding="utf-8")

        telemetry = setup_session_telemetry(
            session_id=self._header.id,
            cwd=Path(self._cwd) if self._cwd else self._session_file.parent,
            scenario_name=None,
            file_path=self._session_file,
        )
        try:
            header_record = _header_to_record(self._header)
            telemetry.logger.emit(
                body=header_record,
                severity_number=SeverityNumber.INFO,
                severity_text="INFO",
                event_name="agentm.session.header",
                attributes={
                    "agentm.session.id": str(header_record.get("id", "")),
                    "agentm.session.header.id": str(header_record.get("id", "")),
                },
            )
            for entry_id in self._order:
                entry_record = _entry_to_record(self._entries[entry_id])
                telemetry.logger.emit(
                    body=entry_record,
                    severity_number=SeverityNumber.INFO,
                    severity_text="INFO",
                    event_name="agentm.message.appended",
                    attributes={
                        "agentm.session.id": self._header.id,
                        "agentm.message.id": str(entry_record.get("id", "")),
                        "agentm.message.parent_id": str(
                            entry_record.get("parent_id", "") or ""
                        ),
                        "agentm.message.type": str(entry_record.get("type", "")),
                    },
                )
        finally:
            telemetry.shutdown()

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
        fork._header = copy.deepcopy(self._header)
        fork._entries = copy.deepcopy(self._entries)
        fork._order = list(self._order)
        fork._leaf_id = entry_id
        return fork

    def create_branched_session(self, leaf_id: str) -> str | None:
        branch = self.get_branch(leaf_id)
        if not branch:
            raise KeyError(f"unknown entry id: {leaf_id}")

        fork = self.create(
            self._cwd, self._session_dir or self.default_session_dir(self._cwd)
        )
        if self._session_file is not None and fork._header is not None:
            fork._header = SessionHeader(
                type="session",
                version=CURRENT_SESSION_VERSION,
                id=fork._header.id,
                timestamp=fork._header.timestamp,
                cwd=self._cwd,
                parent_session=str(self._session_file),
            )
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
        # No bus is attached to ``fork`` — the in-memory tree is now
        # complete, so persist the snapshot directly through the merged-
        # log writer instead of relying on the (absent) observability sink.
        fork._direct_write_merged_log()
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
        children = [
            entry for entry in self.get_entries() if entry.parent_id == parent_id
        ]
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

    def _truncate_to_last_boundary(self) -> None:
        """Drop a trailing incomplete turn after a cold load.

        Entries persist incrementally during a turn; only a clean
        ``agent_end`` appends a ``turn_committed`` marker (see
        ``AgentSession._on_agent_end_commit_boundary``). A process killed
        mid-turn leaves entries with no trailing marker — replaying them would
        rebuild a context ending in a dangling tool_call. Move the active leaf
        back to the last committed boundary so the half-turn is off the active
        branch (it stays in the tree for audit). No-op for logs that carry no
        markers at all (pre-feature sessions), so old traces resume verbatim.
        """
        # Walk leaf -> root; the first marker we hit is the last on the active
        # branch. Stops at the nearest boundary instead of scanning the whole
        # branch, and never leaves the active path (a plain ``_order`` scan
        # would wrongly consider markers on a forked/dead branch).
        cursor = self._leaf_id
        while cursor is not None:
            entry = self._entries.get(cursor)
            if entry is None:
                return
            if entry.type == ENTRY_TYPE_TURN_COMMITTED:
                self._leaf_id = cursor
                return
            cursor = entry.parent_id

    def get_tree(self) -> list[SessionTreeNode]:
        nodes = {
            entry.id: SessionTreeNode(entry=entry, children=[])
            for entry in self.get_entries()
        }
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
                    current.has_compacted_ancestor
                    or current.entry.type == ENTRY_TYPE_COMPACTION
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
            if entry.type == ENTRY_TYPE_COMPACTION:
                latest_compaction = entry

        messages: list[AgentMessage] = []

        def append_materialized(entry: SessionEntry) -> None:
            # Skip compaction entries here — they are handled separately
            # below so the synthesized summary anchors the rebuilt context.
            if entry.type == ENTRY_TYPE_COMPACTION:
                return
            materializer = ENTRY_MATERIALIZERS.get(entry.type)
            if materializer is None:
                return
            message = materializer.to_message(entry)
            if message is not None:
                messages.append(message)

        if latest_compaction is None:
            for entry in path:
                append_materialized(entry)
            return SessionContext(messages=messages)

        materializer = ENTRY_MATERIALIZERS.get(ENTRY_TYPE_COMPACTION)
        if materializer is not None:
            summary_message = materializer.to_message(latest_compaction)
            if summary_message is not None:
                messages.append(summary_message)

        details = (
            latest_compaction.payload
            if isinstance(latest_compaction.payload, dict)
            else {}
        )
        # ``first_kept_entry_id`` is the general kernel seam for compaction
        # strategies that keep a verbatim tail: it names the first entry to
        # replay after the summary. The default ``llm_compaction`` atom is
        # full-compress and omits it, so this is ``None`` and the fallback
        # below keeps nothing before the compaction entry — yielding
        # ``[summary] + post-compaction``. Kept generic so a tail-keeping
        # compaction atom can plug in without editing core.
        first_kept_id = details.get("first_kept_entry_id") or details.get(
            "firstKeptEntryId"
        )
        compaction_index = path.index(latest_compaction)
        first_kept_index: int | None = None
        if isinstance(first_kept_id, str):
            first_kept_index = next(
                (
                    index
                    for index, entry in enumerate(path[:compaction_index])
                    if entry.id == first_kept_id
                ),
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

    def get_raw_messages(self) -> list[AgentMessage]:
        """Return message payloads directly from entries, no materializers.

        Unlike :meth:`get_messages` (which routes through
        ``ENTRY_MATERIALIZERS``), this reads the already-deserialized
        ``AgentMessage`` payloads straight from the entry tree. Use when
        atoms are not loaded (offline tools, replay drivers).
        """
        return [
            e.payload
            for e in self.get_branch()
            if e.type == ENTRY_TYPE_MESSAGE and isinstance(e.payload, AgentMessage)
        ]


class JsonlSessionStore:
    """Default presenter session store backed by JSONL SessionManager files."""

    def __init__(
        self, *, cwd: Path | None = None, session_dir: Path | None = None
    ) -> None:
        self._cwd = cwd
        self._session_dir = session_dir

    def open(self, id: str) -> SessionManager:
        candidate = Path(id)
        if candidate.is_file():
            return SessionManager.open(candidate)
        directory = self._session_dir or SessionManager.default_session_dir(
            str(self._cwd or Path.cwd())
        )
        direct = directory / f"{id}.jsonl"
        if direct.is_file():
            return SessionManager.open(direct)
        matches = sorted(directory.glob(f"*_{id}.jsonl"))
        if not matches:
            raise FileNotFoundError(id)
        return SessionManager.open(matches[0])

    def most_recent(self, cwd: Path) -> SessionManager | None:
        directory = self._session_dir or SessionManager.default_session_dir(str(cwd))
        latest = SessionManager._find_most_recent(directory)
        if latest is None:
            return None
        return SessionManager.open(latest, session_dir=directory, cwd_override=str(cwd))

    def create(self, cwd: Path) -> SessionManager:
        return SessionManager.create(str(cwd), self._session_dir)

    def fork(
        self,
        source_id: str,
        *,
        up_to: int | None = None,
    ) -> SessionManager:
        source = self.open(source_id)
        branch = source.get_branch()
        messages = [
            e.payload
            for e in branch
            if e.type == ENTRY_TYPE_MESSAGE and isinstance(e.payload, AgentMessage)
        ]
        if up_to is not None:
            messages = messages[:up_to]

        source_sid = source.get_session_id()
        cwd = source._cwd or str(self._cwd or Path.cwd())
        directory = self._session_dir or SessionManager.default_session_dir(cwd)

        forked = SessionManager(
            cwd=cwd,
            session_dir=directory,
            persist=True,
            parent_session=source_sid,
        )
        for msg in messages:
            forked.append_message(msg)
        return forked


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
        super().__init__(
            cwd=cwd, session_dir=path.parent, session_file=path, persist=True
        )


__all__ = [
    "CURRENT_SESSION_VERSION",
    "InMemorySessionManager",
    "JsonlSessionManager",
    "JsonlSessionStore",
    "SessionContext",
    "SessionEntry",
    "SessionHeader",
    "SessionManager",
    "SessionTreeNode",
    "branch_summary_entry",
    "compaction_entry",
    "message_entry",
]
