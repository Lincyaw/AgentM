# code-health: ignore-file[AM025] -- persistence serializes untyped JSON/SQLite values
"""Policy engine persistence — SQLite WAL for cross-session state."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from agentm.storage.sql import create_sqlite_engine, execute_script

from .state import args_hash
from .types import EffectRecord, EntityRecord, FileStateEntry, ToolArgs, ToolLogEntry

if TYPE_CHECKING:
    from .ifg import IfgToolEvent
    from .repository_index import RepositoryIndex


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rule_state (
    rule_id TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    fire_count INTEGER DEFAULT 0,
    last_fired_at REAL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (rule_id, scope_key)
);

CREATE TABLE IF NOT EXISTS event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    rule_id TEXT,
    mode TEXT,
    effect TEXT,
    reason TEXT,
    turn INTEGER,
    context_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_event_log_ts ON event_log(ts);
CREATE INDEX IF NOT EXISTS idx_event_log_rule ON event_log(rule_id, ts);
CREATE INDEX IF NOT EXISTS idx_event_log_session ON event_log(session_id, ts);

CREATE TABLE IF NOT EXISTS policy_tool_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    phase TEXT NOT NULL,
    tool_call_id TEXT,
    tool_name TEXT NOT NULL,
    args_hash TEXT,
    args_json TEXT,
    result_json TEXT,
    state_json TEXT,
    processed_json TEXT,
    exit_code INTEGER,
    duration_ms INTEGER,
    result_content_hash TEXT,
    cwd TEXT
);

CREATE INDEX IF NOT EXISTS idx_policy_tool_events_session
    ON policy_tool_events(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_policy_tool_events_tool
    ON policy_tool_events(tool_name, ts);

CREATE TABLE IF NOT EXISTS policy_file_state (
    session_id TEXT NOT NULL,
    path TEXT NOT NULL,
    updated_at REAL NOT NULL,
    first_read_turn INTEGER,
    last_read_turn INTEGER,
    last_write_turn INTEGER,
    read_count INTEGER NOT NULL,
    write_count INTEGER NOT NULL,
    content_hash TEXT,
    reverts_to_prior_hash INTEGER NOT NULL,
    state_json TEXT,
    PRIMARY KEY (session_id, path)
);

CREATE TABLE IF NOT EXISTS policy_entity_state (
    session_id TEXT NOT NULL,
    entity TEXT NOT NULL,
    updated_at REAL NOT NULL,
    entity_type TEXT NOT NULL,
    first_seen_turn INTEGER NOT NULL,
    last_seen_turn INTEGER NOT NULL,
    occurrence_count INTEGER NOT NULL,
    evidence_json TEXT NOT NULL,
    PRIMARY KEY (session_id, entity)
);

CREATE INDEX IF NOT EXISTS idx_policy_entity_state_session
    ON policy_entity_state(session_id, updated_at);

CREATE TABLE IF NOT EXISTS policy_context_state (
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    context_json TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_index)
);

CREATE TABLE IF NOT EXISTS policy_turn_summary (
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    summary_json TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_index)
);

CREATE TABLE IF NOT EXISTS policy_session_summary (
    session_id TEXT PRIMARY KEY,
    updated_at REAL NOT NULL,
    summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_eval_error (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    tool_name TEXT,
    error TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policy_eval_error_session
    ON policy_eval_error(session_id, ts);
"""


_POLICY_TOOL_EVENTS_COLUMNS = {
    "processed_json": "TEXT",
    "exit_code": "INTEGER",
    "duration_ms": "INTEGER",
    "result_content_hash": "TEXT",
    "cwd": "TEXT",
}


class PolicyPersistence:
    """Cross-session state store backed by a SQLAlchemy SQLite engine."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._engine: Engine | None = None
        self._pending_effects: list[tuple[str, EffectRecord]] = []
        self._pending_tool_events: list[dict[str, object]] = []
        self._pending_file_states: list[dict[str, object]] = []
        self._pending_entity_states: list[dict[str, object]] = []
        self._pending_context_states: list[dict[str, object]] = []
        self._pending_turn_summaries: list[dict[str, object]] = []
        self._pending_session_summaries: list[dict[str, object]] = []
        self._pending_eval_errors: list[dict[str, object]] = []

    def open(self) -> None:
        if self._db_path is None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_sqlite_engine(self._db_path)
        with self._engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
            execute_script(conn, _SCHEMA_SQL)
            self._ensure_columns(
                conn,
                "policy_tool_events",
                _POLICY_TOOL_EVENTS_COLUMNS,
            )

    def close(self) -> None:
        self.flush()
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def queue_effect(self, session_id: str, record: EffectRecord) -> None:
        """Queue an effect record for batch write."""
        self._pending_effects.append((session_id, record))

    def queue_tool_event(
        self,
        *,
        session_id: str,
        turn: int,
        phase: str,
        tool_call_id: str,
        tool_name: str,
        args: ToolArgs,
        entry: ToolLogEntry | None = None,
        result: Mapping[str, object | None] | None = None,
        processed: Mapping[str, object | None] | None = None,
        taint_labels: Sequence[str] = (),
        cwd: str | None = None,
    ) -> None:
        """Queue a policy-observed tool event with raw + processed payloads."""
        state: dict[str, object] = {"taint_labels": list(taint_labels)}
        if entry is not None:
            state["tool_log_entry"] = _tool_log_entry_json(entry)
        if processed is not None:
            state["processed"] = dict(processed)
        self._pending_tool_events.append(
            {
                "session_id": session_id,
                "turn": turn,
                "phase": phase,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "args_hash": args_hash(args),
                "args_json": _to_json(args),
                "result_json": _to_json(result or {}),
                "state_json": _to_json(state),
                "processed_json": _to_json(processed or state),
                "exit_code": entry.exit_code if entry is not None else None,
                "duration_ms": entry.duration_ms if entry is not None else None,
                "result_content_hash": (
                    (processed or {}).get("result_content_hash")
                    if processed is not None
                    else None
                ),
                "cwd": cwd,
            }
        )

    def queue_file_state_snapshot(
        self,
        session_id: str,
        entries: Sequence[FileStateEntry],
    ) -> None:
        """Queue the current per-file state table."""
        for entry in entries:
            self._pending_file_states.append(
                {
                    "session_id": session_id,
                    "path": entry.path,
                    "first_read_turn": entry.first_read_turn,
                    "last_read_turn": entry.last_read_turn,
                    "last_write_turn": entry.last_write_turn,
                    "read_count": entry.read_count,
                    "write_count": entry.write_count,
                    "content_hash": entry.content_hash,
                    "reverts_to_prior_hash": int(entry.reverts_to_prior_hash),
                    "state_json": _to_json(_file_state_json(entry)),
                }
            )

    def queue_entity_state_snapshot(
        self,
        session_id: str,
        entries: Sequence[EntityRecord],
    ) -> None:
        """Queue the current entity/evidence registry."""
        for entry in entries:
            self._pending_entity_states.append(
                {
                    "session_id": session_id,
                    "entity": entry.entity,
                    "entity_type": entry.entity_type,
                    "first_seen_turn": entry.first_seen_turn,
                    "last_seen_turn": entry.last_seen_turn,
                    "occurrence_count": entry.occurrence_count,
                    "evidence_json": _to_json(
                        [
                            {
                                "type": ev.type,
                                "turn": ev.turn,
                                "detail": ev.detail,
                            }
                            for ev in entry.evidence.records
                        ]
                    ),
                }
            )

    def queue_context_state(
        self,
        session_id: str,
        turn_index: int,
        state: Mapping[str, object],
    ) -> None:
        self._pending_context_states.append(
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "context_json": _to_json(state),
            }
        )

    def queue_turn_summary(
        self,
        session_id: str,
        turn_index: int,
        summary: Mapping[str, object],
    ) -> None:
        self._pending_turn_summaries.append(
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "summary_json": _to_json(summary),
            }
        )

    def queue_session_summary(
        self,
        session_id: str,
        summary: Mapping[str, object],
    ) -> None:
        self._pending_session_summaries.append(
            {
                "session_id": session_id,
                "summary_json": _to_json(summary),
            }
        )

    def queue_eval_error(
        self,
        *,
        session_id: str,
        turn: int,
        rule_id: str,
        channel: str,
        tool_name: str,
        error: str,
    ) -> None:
        self._pending_eval_errors.append(
            {
                "session_id": session_id,
                "turn": turn,
                "rule_id": rule_id,
                "channel": channel,
                "tool_name": tool_name,
                "error": error,
            }
        )

    def flush(self) -> None:
        """Write all pending policy records in one transaction."""
        if not self._engine or not self._has_pending():
            return

        now = time.time()
        try:
            with self._engine.begin() as conn:
                self._flush_effects(conn, now)
                self._flush_tool_events(conn, now)
                self._flush_file_states(conn, now)
                self._flush_entity_states(conn, now)
                self._flush_context_states(conn, now)
                self._flush_turn_summaries(conn, now)
                self._flush_session_summaries(conn, now)
                self._flush_eval_errors(conn, now)
            self._clear_pending()
        except SQLAlchemyError as e:
            logger.warning("policy persistence flush failed: {}", e)

    def persist_ifg_tool_events(
        self,
        events: Sequence["IfgToolEvent"],
        *,
        repository_index: "RepositoryIndex | None" = None,
        rebuild_projection: bool = True,
    ) -> None:
        """Write IFG facts for runtime-observed tool events."""
        if not self._engine or not events:
            return
        from .ifg import persist_ifg_tool_events

        try:
            with self._engine.begin() as conn:
                persist_ifg_tool_events(
                    conn,
                    events,
                    update_summary=False,
                    repository_index=repository_index,
                    rebuild_projection=rebuild_projection,
                )
        except Exception as e:
            logger.warning("policy IFG realtime extraction failed: {}", e)

    def rebuild_ifg_session(
        self,
        session_id: str,
        *,
        repository_index: "RepositoryIndex | None" = None,
    ) -> None:
        """Materialize one session's derived IFG projection from atomic rows."""

        if not self._engine:
            return
        from .ifg.service import rebuild_ifg_projection

        try:
            with self._engine.begin() as conn:
                rebuild_ifg_projection(
                    conn,
                    session_id,
                    repository_index=repository_index,
                )
        except Exception as e:
            logger.warning("policy IFG projection rebuild failed: {}", e)

    def count_cross_session(
        self,
        rule_id: str | None = None,
        error_fingerprint: str | None = None,
        ttl_days: int = 14,
    ) -> int:
        """Count matching events across sessions within TTL."""
        if not self._engine:
            return 0

        cutoff = time.time() - (ttl_days * 86400)
        conditions = ["ts > ?"]
        params: list[float | str] = [cutoff]

        if rule_id:
            conditions.append("rule_id = ?")
            params.append(rule_id)
        if error_fingerprint:
            conditions.append("context_json LIKE ?")
            params.append(f"%{error_fingerprint}%")

        sql = f"SELECT COUNT(*) FROM event_log WHERE {' AND '.join(conditions)}"
        try:
            with self._engine.connect() as conn:
                row = conn.exec_driver_sql(sql, tuple(params)).fetchone()
            return row[0] if row else 0
        except SQLAlchemyError:
            return 0

    def delete_session(self, session_id: str) -> int:
        """Delete all persisted policy projections for one session."""
        if not self._engine:
            return 0
        tables = (
            "event_log",
            "policy_tool_events",
            "policy_file_state",
            "policy_entity_state",
            "policy_context_state",
            "policy_turn_summary",
            "policy_session_summary",
            "policy_eval_error",
        )
        try:
            from .ifg import delete_ifg_session

            deleted = 0
            with self._engine.begin() as conn:
                for table in tables:
                    cursor = conn.exec_driver_sql(
                        f"DELETE FROM {table} WHERE session_id = ?",  # noqa: S608
                        (session_id,),
                    )
                    deleted += cursor.rowcount
                deleted += delete_ifg_session(conn, session_id)
            return deleted
        except SQLAlchemyError:
            return 0

    def prune(self, ttl_days: int = 30) -> int:
        """Remove old entries from TTL. Returns count deleted."""
        if not self._engine:
            return 0
        cutoff = time.time() - (ttl_days * 86400)
        try:
            deleted = 0
            with self._engine.begin() as conn:
                for table in (
                    "event_log",
                    "policy_tool_events",
                    "policy_eval_error",
                ):
                    cursor = conn.exec_driver_sql(
                        f"DELETE FROM {table} WHERE ts < ?",  # noqa: S608
                        (cutoff,),
                    )
                    deleted += cursor.rowcount
                for table in (
                    "policy_file_state",
                    "policy_entity_state",
                    "policy_context_state",
                    "policy_turn_summary",
                    "policy_session_summary",
                ):
                    cursor = conn.exec_driver_sql(
                        f"DELETE FROM {table} WHERE updated_at < ?",  # noqa: S608
                        (cutoff,),
                    )
                    deleted += cursor.rowcount
            return deleted
        except SQLAlchemyError:
            return 0

    def _has_pending(self) -> bool:
        return any(
            (
                self._pending_effects,
                self._pending_tool_events,
                self._pending_file_states,
                self._pending_entity_states,
                self._pending_context_states,
                self._pending_turn_summaries,
                self._pending_session_summaries,
                self._pending_eval_errors,
            )
        )

    def _clear_pending(self) -> None:
        self._pending_effects.clear()
        self._pending_tool_events.clear()
        self._pending_file_states.clear()
        self._pending_entity_states.clear()
        self._pending_context_states.clear()
        self._pending_turn_summaries.clear()
        self._pending_session_summaries.clear()
        self._pending_eval_errors.clear()

    def _flush_effects(self, conn: Connection, now: float) -> None:
        for session_id, rec in self._pending_effects:
            conn.exec_driver_sql(
                """
                INSERT INTO event_log
                    (ts, session_id, rule_id, mode, effect, reason,
                     turn, context_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    session_id,
                    rec.rule_id,
                    rec.mode,
                    rec.effect,
                    rec.reason,
                    rec.turn,
                    _to_json(
                        {
                            "channel": rec.channel,
                            "evidence": rec.context,
                        }
                    ),
                ),
            )

    def _flush_tool_events(self, conn: Connection, now: float) -> None:
        for row in self._pending_tool_events:
            conn.exec_driver_sql(
                """
                INSERT INTO policy_tool_events
                    (ts, session_id, turn, phase, tool_call_id, tool_name,
                     args_hash, args_json, result_json, state_json,
                     processed_json, exit_code, duration_ms, result_content_hash, cwd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    row["session_id"],
                    row["turn"],
                    row["phase"],
                    row["tool_call_id"],
                    row["tool_name"],
                    row["args_hash"],
                    row["args_json"],
                    row["result_json"],
                    row["state_json"],
                    row["processed_json"],
                    row["exit_code"],
                    row["duration_ms"],
                    row["result_content_hash"],
                    row["cwd"],
                ),
            )

    def _flush_file_states(self, conn: Connection, now: float) -> None:
        for row in self._pending_file_states:
            conn.exec_driver_sql(
                """
                INSERT INTO policy_file_state
                    (session_id, path, updated_at, first_read_turn,
                     last_read_turn, last_write_turn, read_count, write_count,
                     content_hash, reverts_to_prior_hash, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, path) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    first_read_turn = excluded.first_read_turn,
                    last_read_turn = excluded.last_read_turn,
                    last_write_turn = excluded.last_write_turn,
                    read_count = excluded.read_count,
                    write_count = excluded.write_count,
                    content_hash = excluded.content_hash,
                    reverts_to_prior_hash = excluded.reverts_to_prior_hash,
                    state_json = excluded.state_json
                """,
                (
                    row["session_id"],
                    row["path"],
                    now,
                    row["first_read_turn"],
                    row["last_read_turn"],
                    row["last_write_turn"],
                    row["read_count"],
                    row["write_count"],
                    row["content_hash"],
                    row["reverts_to_prior_hash"],
                    row["state_json"],
                ),
            )

    def _flush_entity_states(self, conn: Connection, now: float) -> None:
        for row in self._pending_entity_states:
            conn.exec_driver_sql(
                """
                INSERT INTO policy_entity_state
                    (session_id, entity, updated_at, entity_type,
                     first_seen_turn, last_seen_turn, occurrence_count,
                     evidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, entity) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    entity_type = excluded.entity_type,
                    first_seen_turn = excluded.first_seen_turn,
                    last_seen_turn = excluded.last_seen_turn,
                    occurrence_count = excluded.occurrence_count,
                    evidence_json = excluded.evidence_json
                """,
                (
                    row["session_id"],
                    row["entity"],
                    now,
                    row["entity_type"],
                    row["first_seen_turn"],
                    row["last_seen_turn"],
                    row["occurrence_count"],
                    row["evidence_json"],
                ),
            )

    def _flush_context_states(self, conn: Connection, now: float) -> None:
        for row in self._pending_context_states:
            conn.exec_driver_sql(
                """
                INSERT INTO policy_context_state
                    (session_id, turn_index, updated_at, context_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, turn_index) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    context_json = excluded.context_json
                """,
                (
                    row["session_id"],
                    row["turn_index"],
                    now,
                    row["context_json"],
                ),
            )

    def _flush_turn_summaries(self, conn: Connection, now: float) -> None:
        for row in self._pending_turn_summaries:
            conn.exec_driver_sql(
                """
                INSERT INTO policy_turn_summary
                    (session_id, turn_index, updated_at, summary_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, turn_index) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    summary_json = excluded.summary_json
                """,
                (
                    row["session_id"],
                    row["turn_index"],
                    now,
                    row["summary_json"],
                ),
            )

    def _flush_session_summaries(self, conn: Connection, now: float) -> None:
        for row in self._pending_session_summaries:
            conn.exec_driver_sql(
                """
                INSERT INTO policy_session_summary
                    (session_id, updated_at, summary_json)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    summary_json = excluded.summary_json
                """,
                (
                    row["session_id"],
                    now,
                    row["summary_json"],
                ),
            )

    def _flush_eval_errors(self, conn: Connection, now: float) -> None:
        for row in self._pending_eval_errors:
            conn.exec_driver_sql(
                """
                INSERT INTO policy_eval_error
                    (ts, session_id, turn, rule_id, channel, tool_name, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    row["session_id"],
                    row["turn"],
                    row["rule_id"],
                    row["channel"],
                    row["tool_name"],
                    row["error"],
                ),
            )

    def _ensure_columns(
        self,
        conn: Connection,
        table: str,
        columns: Mapping[str, str],
    ) -> None:
        existing = {
            str(row[1])
            for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")  # noqa: S608
        }
        for name, ddl_type in columns.items():
            if name in existing:
                continue
            conn.exec_driver_sql(
                f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}"  # noqa: S608
            )


def _tool_log_entry_json(entry: ToolLogEntry) -> dict[str, object]:
    return {
        "turn": entry.turn,
        "tool": entry.tool,
        "args_hash": entry.args_hash,
        "path": entry.path,
        "cmd": entry.cmd,
        "exit_code": entry.exit_code,
        "error": entry.error,
        "error_fingerprint": entry.error_fingerprint,
        "error_category": entry.error_category,
        "duration_ms": entry.duration_ms,
        "result_length": entry.result_length,
        "is_repeat": entry.is_repeat,
        "repeat_count": entry.repeat_count,
    }


def _file_state_json(entry: FileStateEntry) -> dict[str, object]:
    return {
        "path": entry.path,
        "first_read_turn": entry.first_read_turn,
        "last_read_turn": entry.last_read_turn,
        "last_write_turn": entry.last_write_turn,
        "read_count": entry.read_count,
        "write_count": entry.write_count,
        "content_hash": entry.content_hash,
        "reverts_to_prior_hash": entry.reverts_to_prior_hash,
    }


def _to_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(raw) for key, raw in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}
    if isinstance(value, bytearray):
        return {"encoding": "hex", "data": bytes(value).hex()}
    return value
