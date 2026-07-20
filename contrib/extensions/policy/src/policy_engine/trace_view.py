# code-health: ignore-file[AM025] -- policy engine normalizes untyped YAML, SQLite, and runtime event payloads
"""Policy-owned trace view provider for the shared AgentM Textual viewer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from sqlalchemy.engine import Connection, RowMapping

from policy_engine.bash_parser import BashRedirect, BashSegment, parse_bash_segments

from agentm.presenter.trajectory import (
    TraceQuery,
    TraceRow,
    TraceSnapshot,
    TraceTableColumn,
    TraceView,
    TraceViewRegistry,
    TraceViewSpec,
    build_trace_view_registry,
    filter_trace_rows,
    run_textual_viewer,
)
from agentm.core.abi.query import TrajectoryQueryStore
from agentm.storage.sql import create_sqlite_engine

_POLICY_TABLES = (
    "event_log",
    "policy_tool_events",
    "policy_file_state",
    "policy_entity_state",
    "policy_context_state",
    "policy_turn_summary",
    "policy_session_summary",
    "policy_eval_error",
)

_SUMMARY_CATEGORIES = frozenset({"summary"})
_EFFECT_CATEGORIES = frozenset({"effects"})
_TOOL_CATEGORIES = frozenset({"tools"})
_FILE_CATEGORIES = frozenset({"files"})
_FILE_STREAM_CATEGORIES = frozenset({"file_stream"})
_FILE_TOOL_STREAM_CATEGORIES = frozenset({"file_tool_stream"})
_BASH_FILE_STREAM_CATEGORIES = frozenset({"bash_file_stream"})
_BASH_COMMAND_CATEGORIES = frozenset({"bash_commands"})
_ENTITY_CATEGORIES = frozenset({"entities"})
_ERROR_CATEGORIES = frozenset({"errors"})

_FILE_COLUMNS = (
    TraceTableColumn("location", "Loc", max_width=8),
    TraceTableColumn("name", "File", max_width=60),
    TraceTableColumn("reads", "Reads", max_width=6),
    TraceTableColumn("writes", "Writes", max_width=6),
    TraceTableColumn("refs", "Refs", max_width=5),
    TraceTableColumn("events", "Events", max_width=6),
    TraceTableColumn("latest", "Latest", max_width=34),
    TraceTableColumn("preview", "Preview", max_width=56),
)
_FILE_STREAM_COLUMNS = (
    TraceTableColumn("location", "Loc", max_width=8),
    TraceTableColumn("name", "File", max_width=58),
    TraceTableColumn("operation", "Operation", max_width=9),
    TraceTableColumn("phase", "Phase", max_width=9),
    TraceTableColumn("file_result", "File Result", max_width=12),
    TraceTableColumn("tool_result", "Tool Result", max_width=18),
    TraceTableColumn("source", "Source", max_width=22),
    TraceTableColumn("tool", "Tool", max_width=16),
    TraceTableColumn("preview", "Preview", max_width=56),
)
_BASH_FILE_STREAM_COLUMNS = (
    TraceTableColumn("location", "Loc", max_width=8),
    TraceTableColumn("name", "File", max_width=58),
    TraceTableColumn("operation", "Operation", max_width=9),
    TraceTableColumn("evidence", "Evidence", max_width=12),
    TraceTableColumn("phase", "Phase", max_width=9),
    TraceTableColumn("tool_result", "Tool Result", max_width=18),
    TraceTableColumn("source", "Source", max_width=22),
    TraceTableColumn("command", "Command", max_width=72),
)
_BASH_COMMAND_COLUMNS = (
    TraceTableColumn("count", "Count", max_width=7),
    TraceTableColumn("family", "Family", max_width=10),
    TraceTableColumn("command", "Command", max_width=14),
    TraceTableColumn("template", "Template", max_width=82),
    TraceTableColumn("errors", "Tool Err", max_width=8),
    TraceTableColumn("first_turn", "First", max_width=7),
    TraceTableColumn("last_turn", "Last", max_width=7),
    TraceTableColumn("example", "Example", max_width=72),
)

_HEREDOC_RE = re.compile(
    r"<<-?\s*(?:(?P<quote>['\"])(?P<quoted>[^'\"]+)(?P=quote)|(?P<bare>[A-Za-z0-9_./-]+))"
)
_SHELL_SEPARATORS = frozenset({";", "&&", "||", "|", "|&", "\n"})
_SHELL_REDIRECTS = frozenset(
    {">", ">>", "<", "<<", "<<-", "<<<", "<>", ">|", "&>", "&>>"}
)
_FILE_READER_COMMANDS = frozenset({"awk", "cat", "head", "nl", "sed", "tail", "wc"})
_FILE_QUERY_COMMANDS = frozenset(
    {"ack", "ag", "fd", "find", "grep", "ls", "locate", "rg", "tree"}
)
_GIT_QUERY_SUBCOMMANDS = frozenset(
    {"check-ignore", "diff", "grep", "log", "ls-files", "show", "status"}
)
_FILE_WRITER_COMMANDS = frozenset(
    {"cp", "mkdir", "mv", "perl", "rm", "sed", "tee", "touch", "truncate"}
)
_STRUCTURED_FILE_TOOL_NAMES = frozenset({"read", "write", "edit"})
_SUBCOMMAND_TOOLS = frozenset({"git", "npm", "npx", "pnpm", "yarn"})
_CONTROL_COMMANDS = frozenset({"cd", "echo", "pwd", "readlink", "which"})
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]{2,}\b")
_HEX_TOKEN_RE = re.compile(r"[0-9a-fA-F]{7,64}")
_NUMERIC_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class _FileEvent:
    event_id: int
    ts: float
    turn: int
    phase: str
    tool_name: str
    operation: str
    path: str
    source: str
    args_hash: str | None
    file_status: str
    tool_status: str
    tool_exit_code: int | None
    duration_ms: int | None
    error_category: str | None
    error_fingerprint: str | None
    result_content_hash: str | None
    content_hash: str | None
    previous_content_hash: str | None
    policy_tool_event: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _MergedFileEvent:
    event: _FileEvent
    phases: tuple[str, ...]
    sources: tuple[str, ...]
    events: tuple[_FileEvent, ...]

    @property
    def phase_label(self) -> str:
        return "+".join(self.phases)

    @property
    def source_label(self) -> str:
        return "+".join(self.sources)


@dataclass(frozen=True, slots=True)
class PolicyTraceViewProvider:
    """Register policy projection rows into the shared trace viewer."""

    db_path: Path | None = None

    def trace_view_specs(self) -> Sequence[TraceViewSpec]:
        return (
            TraceViewSpec(
                id="policy",
                title="Policy",
                description="Policy projection overview for this trajectory.",
                shortcut="5",
                build=lambda snapshot, query: self._build_view(
                    "policy",
                    "Policy",
                    snapshot,
                    query,
                    categories=_SUMMARY_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-effects",
                title="Policy Effects",
                description="Persisted policy rule firings.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-effects",
                    "Policy Effects",
                    snapshot,
                    query,
                    categories=_EFFECT_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-tools",
                title="Policy Tools",
                description="Raw and processed tool observations seen by policy.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-tools",
                    "Policy Tools",
                    snapshot,
                    query,
                    categories=_TOOL_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-files",
                title="Policy Files",
                description="Per-file read/write history.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-files",
                    "Policy Files",
                    snapshot,
                    query,
                    categories=_FILE_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-file-tool-stream",
                title="File Tool Stream",
                description="Direct read/write/edit file-tool stream.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-file-tool-stream",
                    "File Tool Stream",
                    snapshot,
                    query,
                    categories=_FILE_TOOL_STREAM_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-bash-commands",
                title="Bash Commands",
                description="Bash command templates clustered from token sequences.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-bash-commands",
                    "Bash Commands",
                    snapshot,
                    query,
                    categories=_BASH_COMMAND_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-bash-file-stream",
                title="Bash File Stream",
                description="Bash-derived file operations and references.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-bash-file-stream",
                    "Bash File Stream",
                    snapshot,
                    query,
                    categories=_BASH_FILE_STREAM_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-file-stream",
                title="File Stream",
                description="Chronological structured file-tool event stream.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-file-stream",
                    "File Stream",
                    snapshot,
                    query,
                    categories=_FILE_STREAM_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-entities",
                title="Policy Entities",
                description="Entity/evidence state extracted by policy.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-entities",
                    "Policy Entities",
                    snapshot,
                    query,
                    categories=_ENTITY_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-errors",
                title="Policy Errors",
                description="Policy evaluation errors.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-errors",
                    "Policy Errors",
                    snapshot,
                    query,
                    categories=_ERROR_CATEGORIES,
                ),
            ),
        )

    def _build_view(
        self,
        view_id: str,
        title: str,
        snapshot: TraceSnapshot,
        query: TraceQuery,
        *,
        categories: frozenset[str],
    ) -> TraceView:
        db_path = self.db_path or default_policy_db_path()
        rows = load_policy_trace_rows(
            db_path,
            snapshot.session_id,
            categories=categories,
        )
        filtered = filter_trace_rows(rows, query)
        return TraceView(
            id=view_id,
            title=title,
            rows=filtered,
            summary=_summary_text(db_path, rows, filtered),
            empty_text="No policy rows match the current query.",
            columns=_columns_for_categories(categories),
        )


def _columns_for_categories(
    categories: frozenset[str],
) -> tuple[TraceTableColumn, ...]:
    if categories == _FILE_CATEGORIES:
        return _FILE_COLUMNS
    if categories == _FILE_STREAM_CATEGORIES:
        return _FILE_STREAM_COLUMNS
    if categories == _FILE_TOOL_STREAM_CATEGORIES:
        return _FILE_STREAM_COLUMNS
    if categories == _BASH_FILE_STREAM_CATEGORIES:
        return _BASH_FILE_STREAM_COLUMNS
    if categories == _BASH_COMMAND_CATEGORIES:
        return _BASH_COMMAND_COLUMNS
    return ()


def build_policy_trace_view_registry(db_path: Path | None = None) -> TraceViewRegistry:
    """Build AgentM's shared trajectory app registry with policy tabs."""

    return build_trace_view_registry((PolicyTraceViewProvider(db_path=db_path),))


def run_policy_trace_viewer(
    query: TrajectoryQueryStore,
    session_id: str,
    *,
    db_path: Path | None = None,
    follow: bool = False,
) -> None:
    """Run AgentM's shared Textual trace viewer with policy projection tabs."""

    run_textual_viewer(
        query,
        session_id,
        follow=follow,
        registry=build_policy_trace_view_registry(db_path),
    )


def default_policy_db_path() -> Path:
    agentm_home = os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm"))
    return Path(agentm_home) / "policy_state" / "policy.db"


def load_policy_trace_rows(
    db_path: Path,
    session_id: str,
    *,
    categories: frozenset[str],
) -> tuple[TraceRow, ...]:
    if not db_path.exists():
        return (
            _diagnostic_row(
                session_id,
                title="Policy DB Missing",
                preview=str(db_path),
                content=(
                    "No persisted policy database was found.\n\n"
                    f"db_path: {db_path}\n"
                    "Run policy_engine backfill or start a session with the "
                    "policy extension enabled first."
                ),
            ),
        )

    engine = create_sqlite_engine(db_path)
    try:
        with engine.connect() as conn:
            existing = _existing_tables(conn)
            if not existing.intersection(_POLICY_TABLES):
                return (
                    _diagnostic_row(
                        session_id,
                        title="Policy Tables Missing",
                        preview=str(db_path),
                        content=f"No policy projection tables were found in {db_path}",
                    ),
                )
            counts = _session_counts(conn, session_id, existing)
            rows: list[TraceRow] = []
            if "summary" in categories:
                rows.extend(_summary_rows(conn, session_id, db_path, existing, counts))
            if "effects" in categories and "event_log" in existing:
                rows.extend(_effect_rows(conn, session_id))
            if "tools" in categories and "policy_tool_events" in existing:
                rows.extend(_tool_rows(conn, session_id))
            if "files" in categories and "policy_file_state" in existing:
                rows.extend(_file_rows(conn, session_id))
            if "file_stream" in categories and "policy_tool_events" in existing:
                rows.extend(
                    _file_stream_rows(
                        conn,
                        session_id,
                        include_bash=False,
                        category="file_stream",
                    )
                )
            if "file_tool_stream" in categories and "policy_tool_events" in existing:
                rows.extend(
                    _file_stream_rows(
                        conn,
                        session_id,
                        include_bash=False,
                        category="file_tool_stream",
                        tool_names=_STRUCTURED_FILE_TOOL_NAMES,
                    )
                )
            if "bash_file_stream" in categories and "policy_tool_events" in existing:
                rows.extend(
                    _file_stream_rows(
                        conn,
                        session_id,
                        include_bash=True,
                        category="bash_file_stream",
                    )
                )
            if "bash_commands" in categories and "policy_tool_events" in existing:
                rows.extend(_bash_command_rows(conn, session_id))
            if "entities" in categories and "policy_entity_state" in existing:
                rows.extend(_entity_rows(conn, session_id))
            if "errors" in categories and "policy_eval_error" in existing:
                rows.extend(_eval_error_rows(conn, session_id))
            if not rows:
                rows.append(
                    _empty_category_row(session_id, db_path, counts, categories)
                )
            return tuple(rows)
    finally:
        engine.dispose()


def _rows(
    conn: Connection,
    sql: str,
    params: Sequence[object] = (),
) -> list[RowMapping]:
    return list(conn.exec_driver_sql(sql, tuple(params)).mappings().all())


def _row(
    conn: Connection,
    sql: str,
    params: Sequence[object] = (),
) -> RowMapping | None:
    return conn.exec_driver_sql(sql, tuple(params)).mappings().fetchone()


def _summary_rows(
    conn: Connection,
    session_id: str,
    db_path: Path,
    existing: set[str],
    counts: Mapping[str, int],
) -> list[TraceRow]:
    rows = [
        TraceRow(
            key=f"policy:summary:counts:{session_id}",
            kind="policy",
            title="Projection Counts",
            preview=_counts_preview(counts),
            content=_json_content(
                {
                    "session_id": session_id,
                    "db_path": str(db_path),
                    "counts": dict(counts),
                }
            ),
            metadata={"category": "summary", "db_path": str(db_path)},
        )
    ]

    if "policy_session_summary" in existing:
        row = _row(
            conn,
            """
            SELECT updated_at, summary_json
            FROM policy_session_summary
            WHERE session_id = ?
            """,
            (session_id,),
        )
        if row is not None:
            summary = _loads(row["summary_json"])
            rows.append(
                TraceRow(
                    key=f"policy:summary:session:{session_id}",
                    kind="policy",
                    title="Session Summary",
                    preview=_json_preview(summary),
                    content=_json_content(
                        {
                            "session_id": session_id,
                            "updated_at": row["updated_at"],
                            "summary": summary,
                        }
                    ),
                    metadata={"category": "summary"},
                )
            )

    if "event_log" in existing and counts.get("event_log", 0):
        rows.append(_aggregate_row(conn, session_id, "effects"))
    if "policy_tool_events" in existing and counts.get("policy_tool_events", 0):
        rows.append(_aggregate_row(conn, session_id, "tools"))
    if "policy_entity_state" in existing and counts.get("policy_entity_state", 0):
        rows.append(_aggregate_row(conn, session_id, "entities"))
    if "policy_file_state" in existing and counts.get("policy_file_state", 0):
        rows.append(_aggregate_row(conn, session_id, "file_hotspots"))

    if "policy_turn_summary" in existing:
        for row in _rows(
            conn,
            """
            SELECT turn_index, updated_at, summary_json
            FROM policy_turn_summary
            WHERE session_id = ?
            ORDER BY turn_index ASC
            """,
            (session_id,),
        ):
            summary = _loads(row["summary_json"])
            rows.append(
                TraceRow(
                    key=f"policy:summary:turn:{session_id}:{row['turn_index']}",
                    kind="policy",
                    title=f"Turn Summary T{row['turn_index']}",
                    preview=_turn_summary_preview(summary),
                    content=_json_content(
                        {
                            "session_id": session_id,
                            "turn_index": row["turn_index"],
                            "updated_at": row["updated_at"],
                            "summary": summary,
                        }
                    ),
                    turn_index=row["turn_index"],
                    metadata={"category": "summary"},
                )
            )

    return rows


def _aggregate_row(
    conn: Connection,
    session_id: str,
    category: str,
) -> TraceRow:
    if category == "effects":
        data = [
            dict(row)
            for row in _rows(
                conn,
                """
                SELECT rule_id, mode, effect, COUNT(*) AS count
                FROM event_log
                WHERE session_id = ?
                GROUP BY rule_id, mode, effect
                ORDER BY count DESC, rule_id ASC
                """,
                (session_id,),
            )
        ]
        return TraceRow(
            key=f"policy:aggregate:effects:{session_id}",
            kind="policy",
            title="Effects by Rule",
            preview=_json_preview(data),
            content=_json_content(data),
            metadata={"category": "summary", "aggregate": "effects"},
        )
    if category == "tools":
        data = [
            dict(row)
            for row in _rows(
                conn,
                """
                SELECT tool_name, phase, COUNT(*) AS count
                FROM policy_tool_events
                WHERE session_id = ?
                GROUP BY tool_name, phase
                ORDER BY count DESC, tool_name ASC, phase ASC
                """,
                (session_id,),
            )
        ]
        return TraceRow(
            key=f"policy:aggregate:tools:{session_id}",
            kind="policy",
            title="Tool Events by Tool",
            preview=_json_preview(data),
            content=_json_content(data),
            metadata={"category": "summary", "aggregate": "tools"},
        )
    if category == "entities":
        data = [
            dict(row)
            for row in _rows(
                conn,
                """
                SELECT entity_type, COUNT(*) AS count,
                       SUM(occurrence_count) AS occurrences
                FROM policy_entity_state
                WHERE session_id = ?
                GROUP BY entity_type
                ORDER BY count DESC, entity_type ASC
                """,
                (session_id,),
            )
        ]
        return TraceRow(
            key=f"policy:aggregate:entities:{session_id}",
            kind="policy",
            title="Entities by Type",
            preview=_json_preview(data),
            content=_json_content(data),
            metadata={"category": "summary", "aggregate": "entities"},
        )

    data = [
        dict(row)
        for row in _rows(
            conn,
            """
            SELECT path, read_count, write_count, first_read_turn,
                   last_read_turn, last_write_turn, content_hash
            FROM policy_file_state
            WHERE session_id = ?
            ORDER BY (read_count + write_count) DESC, path ASC
            LIMIT 50
            """,
            (session_id,),
        )
    ]
    return TraceRow(
        key=f"policy:aggregate:file_hotspots:{session_id}",
        kind="policy",
        title="File Hotspots",
        preview=_json_preview(data),
        content=_json_content(data),
        metadata={"category": "summary", "aggregate": "file_hotspots"},
    )


def _effect_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM event_log
        WHERE session_id = ?
        ORDER BY ts ASC, id ASC
        """,
        (session_id,),
    ):
        context = _loads(row["context_json"])
        state = f"{row['mode'] or '-'}:{row['effect'] or '-'}"
        yield TraceRow(
            key=f"policy:effect:{row['id']}",
            kind="policy",
            title=row["rule_id"] or "Policy Effect",
            preview=row["reason"] or _json_preview(context),
            content=_json_content(
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "session_id": row["session_id"],
                    "rule_id": row["rule_id"],
                    "mode": row["mode"],
                    "effect": row["effect"],
                    "reason": row["reason"],
                    "turn": row["turn"],
                    "context": context,
                }
            ),
            turn_index=row["turn"],
            cause=row["effect"],
            is_error=row["mode"] == "enforce" or row["effect"] in {"block", "deny"},
            metadata={"category": "effect", "rule_id": row["rule_id"], "state": state},
        )


def _tool_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM policy_tool_events
        WHERE session_id = ?
        ORDER BY turn ASC, id ASC
        """,
        (session_id,),
    ):
        args = _loads(row["args_json"])
        result = _loads(row["result_json"])
        state = _loads(row["state_json"])
        processed = _loads(row["processed_json"])
        error = _tool_is_error(row, result, state, processed)
        display_name = f"{row['phase']} {row['tool_name']}"
        state_label = _tool_policy_state_label(row, state, processed, error)
        preview = _tool_policy_preview(row, args, result, state, processed)
        yield TraceRow(
            key=f"policy:tool:{row['id']}",
            kind="policy",
            title=display_name,
            preview=preview,
            content=_json_content(
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "session_id": row["session_id"],
                    "turn": row["turn"],
                    "phase": row["phase"],
                    "tool_call_id": row["tool_call_id"],
                    "tool_name": row["tool_name"],
                    "args_hash": row["args_hash"],
                    "args": args,
                    "result": result,
                    "state": state,
                    "processed": processed,
                    "exit_code": row["exit_code"],
                    "duration_ms": row["duration_ms"],
                    "result_content_hash": row["result_content_hash"],
                    "policy_display": {
                        "name": display_name,
                        "state": state_label,
                        "preview": preview,
                    },
                    "rule_author_event": _tool_rule_author_event(
                        row,
                        args,
                        result,
                        state,
                        processed,
                    ),
                }
            ),
            turn_index=row["turn"],
            tool_name=row["tool_name"],
            display_name=display_name,
            is_error=error,
            cause=state_label,
            metadata={
                "category": "tool",
                "phase": row["phase"],
                "tool_call_id": row["tool_call_id"],
                "args_hash": row["args_hash"],
                "exit_code": row["exit_code"],
                "duration_ms": row["duration_ms"],
                "error_category": _tool_error_category(state, processed),
                "error_fingerprint": _tool_error_fingerprint(state, processed),
            },
        )


def _file_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    events_by_path = _file_events_by_path(
        _merge_file_events(_file_events(conn, session_id))
    )
    state_paths: set[str] = set()
    for row in _rows(
        conn,
        """
        SELECT *
        FROM policy_file_state
        WHERE session_id = ?
        ORDER BY (read_count + write_count) DESC, path ASC
        """,
        (session_id,),
    ):
        state_paths.add(row["path"])
        history = events_by_path.get(row["path"], [])
        last_turn = _last_turn(row["last_write_turn"], row["last_read_turn"])
        refs = sum(1 for event in history if event.event.operation == "reference")
        latest = _latest_file_event(history)
        content_signals = _file_content_signals(history)
        state = (
            f"r:{row['read_count']} w:{row['write_count']} ref:{refs} ev:{len(history)}"
        )
        yield TraceRow(
            key=f"policy:file:{row['path']}",
            kind="policy",
            title=row["path"],
            preview=(
                f"first:{_dash(row['first_read_turn'])} "
                f"last:{_dash(last_turn)} latest:{latest} "
                f"hash:{row['content_hash'] or '-'}"
            ),
            content=_json_content(
                {
                    "session_id": row["session_id"],
                    "path": row["path"],
                    "updated_at": row["updated_at"],
                    "first_read_turn": row["first_read_turn"],
                    "last_read_turn": row["last_read_turn"],
                    "last_write_turn": row["last_write_turn"],
                    "read_count": row["read_count"],
                    "write_count": row["write_count"],
                    "content_hash": row["content_hash"],
                    "reverts_to_prior_hash": bool(row["reverts_to_prior_hash"]),
                    "state": _loads(row["state_json"]),
                    "history": [_merged_file_event_json(event) for event in history],
                    "content_signals": content_signals,
                }
            ),
            turn_index=last_turn,
            cause=state,
            metadata={
                "category": "file",
                "path": row["path"],
                "state": state,
                "reads": row["read_count"],
                "writes": row["write_count"],
                "refs": refs,
                "events": len(history),
                "latest": latest,
            },
        )

    for path, history in sorted(events_by_path.items()):
        if path in state_paths:
            continue
        if not history:
            continue
        reads = sum(1 for event in history if event.event.operation == "read")
        writes = sum(1 for event in history if event.event.operation == "write")
        refs = sum(1 for event in history if event.event.operation == "reference")
        latest = history[-1]
        state = f"r:{reads} w:{writes} ref:{refs} ev:{len(history)}"
        latest_label = _latest_file_event(history)
        content_signals = _file_content_signals(history)
        yield TraceRow(
            key=f"policy:file:derived:{path}",
            kind="policy",
            title=path,
            preview=f"latest:{latest_label}",
            content=_json_content(
                {
                    "session_id": session_id,
                    "path": path,
                    "state": "derived from policy_tool_events",
                    "history": [_merged_file_event_json(event) for event in history],
                    "content_signals": content_signals,
                }
            ),
            turn_index=latest.event.turn,
            cause=state,
            metadata={
                "category": "file",
                "path": path,
                "state": state,
                "reads": reads,
                "writes": writes,
                "refs": refs,
                "events": len(history),
                "latest": latest_label,
            },
        )


def _file_stream_rows(
    conn: Connection,
    session_id: str,
    *,
    include_bash: bool,
    category: str,
    tool_names: frozenset[str] | None = None,
) -> Iterable[TraceRow]:
    for merged in _merge_file_events(_file_events(conn, session_id)):
        event = merged.event
        is_bash = event.tool_name == "bash"
        if include_bash != is_bash:
            continue
        if tool_names is not None and event.tool_name not in tool_names:
            continue
        is_error = event.tool_status.startswith(("error", "exit:"))
        tool_call_id = event.policy_tool_event.get("tool_call_id")
        phase = merged.phase_label
        source = merged.source_label
        command = _single_line(
            _command_text(event.policy_tool_event.get("args")),
            limit=240,
        )
        evidence = _file_evidence_label(event, source)
        yield TraceRow(
            key=(
                f"policy:{category}:{event.turn}:{tool_call_id}:"
                f"{event.operation}:{event.path}:{phase}"
            ),
            kind="policy",
            title=event.path,
            preview=(
                f"hash:{_dash(_short_text(event.args_hash, 8))} "
                f"call:{_dash(tool_call_id)} raw:{len(merged.events)}"
            ),
            content=_json_content(
                {
                    "session_id": session_id,
                    "file_event": _merged_file_event_json(merged),
                    "raw_file_events": [
                        _file_event_json(raw_event) for raw_event in merged.events
                    ],
                    "policy_tool_event": event.policy_tool_event,
                }
            ),
            turn_index=event.turn,
            tool_name=event.tool_name,
            display_name=event.path,
            is_error=is_error,
            cause=event.tool_status,
            metadata={
                "category": category,
                "path": event.path,
                "operation": event.operation,
                "phase": phase,
                "result": event.tool_status,
                "file_result": event.file_status,
                "tool_result": event.tool_status,
                "evidence": evidence,
                "source": source,
                "tool": event.tool_name,
                "tool_call_id": tool_call_id,
                "tool_exit_code": event.tool_exit_code,
                "duration_ms": event.duration_ms,
                "command": command,
                "args_hash": event.args_hash,
                "raw_event_count": len(merged.events),
                "error_category": event.error_category,
                "error_fingerprint": event.error_fingerprint,
            },
        )


def _bash_command_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    clusters: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in _preferred_bash_tool_rows(conn, session_id):
        for segment in _bash_command_segments(row):
            clusters[str(segment["template"])].append(segment)

    for template, segments in sorted(
        clusters.items(),
        key=lambda item: (-len(item[1]), str(item[0])),
    ):
        first = min(segments, key=lambda item: (item["turn"], item["event_id"]))
        last = max(segments, key=lambda item: (item["turn"], item["event_id"]))
        errors = sum(1 for item in segments if item["is_error"])
        families = Counter(str(item["family"]) for item in segments)
        commands = Counter(str(item["command"]) for item in segments)
        family = families.most_common(1)[0][0]
        command = commands.most_common(1)[0][0]
        examples = _unique_command_examples(segments)
        yield TraceRow(
            key=f"policy:bash_command:{_stable_id(template)}",
            kind="policy",
            title=template,
            preview=(
                f"count:{len(segments)} errors:{errors} "
                f"family:{family} example:{_single_line(examples[0], limit=160)}"
            ),
            content=_json_content(
                {
                    "session_id": session_id,
                    "template": template,
                    "template_tokens": first["template_tokens"],
                    "count": len(segments),
                    "errors": errors,
                    "family_counts": dict(families),
                    "command_counts": dict(commands),
                    "first_turn": first["turn"],
                    "last_turn": last["turn"],
                    "examples": examples[:8],
                    "segments": segments[:200],
                }
            ),
            turn_index=last["turn"] if isinstance(last["turn"], int) else None,
            tool_name="bash",
            display_name=template,
            is_error=errors > 0,
            cause=family,
            metadata={
                "category": "bash_commands",
                "count": len(segments),
                "family": family,
                "command": command,
                "template": template,
                "errors": errors,
                "first_turn": first["turn"],
                "last_turn": last["turn"],
                "example": _single_line(examples[0], limit=240),
            },
        )


def _preferred_bash_tool_rows(
    conn: Connection,
    session_id: str,
) -> list[RowMapping]:
    rows = _rows(
        conn,
        """
        SELECT *
        FROM policy_tool_events
        WHERE session_id = ? AND tool_name = 'bash'
        ORDER BY turn ASC, id ASC
        """,
        (session_id,),
    )
    by_call: dict[str, RowMapping] = {}
    order: list[str] = []
    for row in rows:
        key = str(row["tool_call_id"] or f"event:{row['id']}")
        if key not in by_call:
            order.append(key)
            by_call[key] = row
            continue
        if by_call[key]["phase"] != "post" and row["phase"] == "post":
            by_call[key] = row
    return [by_call[key] for key in order]


def _bash_command_segments(row: RowMapping) -> Iterable[dict[str, object]]:
    args = _loads(row["args_json"])
    cmd = _command_text(args)
    if not cmd:
        return ()
    result = _loads(row["result_json"])
    state = _loads(row["state_json"])
    processed = _loads(row["processed_json"])
    error = _tool_is_error(row, result, state, processed)
    tool_status = _file_tool_status(row, state, processed, error)
    segments: list[dict[str, object]] = []
    for index, segment in enumerate(parse_bash_segments(cmd)):
        command = list(segment.argv)
        if not command:
            continue
        template_tokens = _bash_segment_template_tokens(segment)
        command_text = _bash_segment_text(segment)
        segments.append(
            {
                "event_id": row["id"],
                "turn": row["turn"],
                "phase": row["phase"],
                "tool_call_id": row["tool_call_id"],
                "command_index": index,
                "command": _command_name(command),
                "family": _bash_segment_family(segment),
                "tokens": list(command),
                "template_tokens": list(template_tokens),
                "template": " ".join(template_tokens),
                "command_text": command_text,
                "full_command": _single_line(cmd, limit=600),
                "parser": segment.parser,
                "pipeline_index": segment.pipeline_index,
                "depth": segment.depth,
                "redirects": [
                    _redirect_json(redirect) for redirect in segment.redirects
                ],
                "tool_result": tool_status,
                "exit_code": _tool_exit_code(row, processed),
                "duration_ms": _coerce_int(row["duration_ms"]),
                "is_error": error,
            }
        )
    return segments


def _unique_command_examples(segments: Sequence[Mapping[str, object]]) -> list[str]:
    examples: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        text = _single_line(segment.get("command_text"), limit=600)
        if not text or text in seen:
            continue
        seen.add(text)
        examples.append(text)
    return examples or ["-"]


def _redirect_json(redirect: BashRedirect) -> dict[str, object]:
    return {
        "kind": redirect.kind,
        "operator": redirect.operator,
        "destination": redirect.destination,
        "descriptor": redirect.descriptor,
        "text": redirect.text,
    }


def _file_events(conn: Connection, session_id: str) -> list[_FileEvent]:
    events: list[_FileEvent] = []
    known_paths = _known_file_paths(conn, session_id)
    for row in _rows(
        conn,
        """
        SELECT *
        FROM policy_tool_events
        WHERE session_id = ?
        ORDER BY turn ASC, id ASC
        """,
        (session_id,),
    ):
        args = _loads(row["args_json"])
        result = _loads(row["result_json"])
        state = _loads(row["state_json"])
        processed = _loads(row["processed_json"])
        error = _tool_is_error(row, result, state, processed)
        paths = _file_event_paths(
            row["tool_name"],
            args,
            processed,
            state,
            known_paths=known_paths,
        )
        if not paths:
            continue
        tool_status = _file_tool_status(row, state, processed, error)
        tool_exit_code = _tool_exit_code(row, processed)
        policy_tool_event = {
            "id": row["id"],
            "ts": row["ts"],
            "turn": row["turn"],
            "phase": row["phase"],
            "tool_call_id": row["tool_call_id"],
            "tool_name": row["tool_name"],
            "args_hash": row["args_hash"],
            "args": args,
            "result": result,
            "processed": processed,
            "state": state,
            "exit_code": row["exit_code"],
            "duration_ms": row["duration_ms"],
            "result_content_hash": row["result_content_hash"],
        }
        for path, source in paths:
            operation = _file_operation(row["tool_name"], args, processed, source)
            file_status = _file_operation_status(
                row["tool_name"],
                row["phase"],
                tool_status,
            )
            known_paths.add(path)
            events.append(
                _FileEvent(
                    event_id=row["id"],
                    ts=row["ts"],
                    turn=row["turn"],
                    phase=row["phase"],
                    tool_name=row["tool_name"],
                    operation=operation,
                    path=path,
                    source=source,
                    args_hash=row["args_hash"],
                    file_status=file_status,
                    tool_status=tool_status,
                    tool_exit_code=tool_exit_code,
                    duration_ms=_coerce_int(
                        row["duration_ms"]
                        if row["duration_ms"] is not None
                        else (
                            processed.get("duration_ms")
                            if isinstance(processed, Mapping)
                            else None
                        )
                    ),
                    error_category=_tool_error_category(state, processed),
                    error_fingerprint=_tool_error_fingerprint(state, processed),
                    result_content_hash=row["result_content_hash"],
                    content_hash=_mapping_str(processed, "content_hash"),
                    previous_content_hash=_mapping_str(
                        processed, "previous_content_hash"
                    ),
                    policy_tool_event=policy_tool_event,
                )
            )
    return events


def _merge_file_events(events: Sequence[_FileEvent]) -> list[_MergedFileEvent]:
    grouped: dict[tuple[object, ...], list[_FileEvent]] = {}
    order: list[tuple[object, ...]] = []
    for event in events:
        key = _file_event_merge_key(event)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(event)

    merged: list[_MergedFileEvent] = []
    for key in order:
        raw_events = grouped[key]
        display_event = next(
            (event for event in reversed(raw_events) if event.phase == "post"),
            raw_events[-1],
        )
        merged.append(
            _MergedFileEvent(
                event=display_event,
                phases=_unique_labels(event.phase for event in raw_events),
                sources=_unique_labels(event.source for event in raw_events),
                events=tuple(raw_events),
            )
        )
    return merged


def _file_event_merge_key(event: _FileEvent) -> tuple[object, ...]:
    tool_call_id = event.policy_tool_event.get("tool_call_id")
    if tool_call_id is None:
        return ("event", event.event_id)
    return (
        "tool_call_file",
        event.turn,
        tool_call_id,
        event.tool_name,
        event.operation,
        event.path,
    )


def _unique_labels(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    labels: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        labels.append(value)
    return tuple(labels)


def _file_events_by_path(
    events: Sequence[_MergedFileEvent],
) -> dict[str, list[_MergedFileEvent]]:
    grouped: defaultdict[str, list[_MergedFileEvent]] = defaultdict(list)
    for event in events:
        grouped[event.event.path].append(event)
    return dict(grouped)


def _known_file_paths(conn: Connection, session_id: str) -> set[str]:
    paths: set[str] = set()
    try:
        rows = _rows(
            conn,
            """
            SELECT path
            FROM policy_file_state
            WHERE session_id = ?
            """,
            (session_id,),
        )
    except Exception as exc:
        logger.debug("policy known file lookup failed: {}", exc)
        return paths
    for row in rows:
        path = _clean_path(row["path"])
        if path:
            paths.add(path)
    return paths


def _file_event_paths(
    tool_name: str,
    args: object,
    processed: object,
    state: object,
    *,
    known_paths: set[str],
) -> list[tuple[str, str]]:
    candidates: list[tuple[object, str]] = []
    if isinstance(args, Mapping):
        candidates.extend(
            [
                (args.get("path"), "args.path"),
                (args.get("file_path"), "args.file_path"),
            ]
        )
    if isinstance(processed, Mapping):
        candidates.append((processed.get("path"), "processed.path"))
    entry = _tool_log_entry(state)
    candidates.append((entry.get("path"), "tool_log.path"))

    cmd = _command_text(args)
    if cmd and tool_name == "bash":
        candidates.extend(_paths_from_command(cmd, known_paths=known_paths))

    seen: set[str] = set()
    paths: list[tuple[str, str]] = []
    for raw, source in candidates:
        path = _clean_path(raw)
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append((path, source))
    return paths


def _file_operation(
    tool_name: str,
    args: object,
    processed: object,
    source: str,
) -> str:
    raw_op = _mapping_str(processed, "file_op")
    if raw_op:
        lowered = raw_op.lower()
        if lowered in {"read", "open", "glob", "list"}:
            return "read"
        if lowered in {"edit", "write", "patch", "delete", "remove"}:
            return "write"
    if tool_name in {"read", "glob"}:
        return "read"
    if tool_name in {"edit", "write"}:
        return "write"
    if source == "cmd.redirect.write":
        return "write"
    if source in {"cmd.reader", "cmd.redirect.read"}:
        return "read"
    if source == "cmd.query":
        return "query"
    if source == "cmd.known":
        return _bash_file_operation(_command_text(args))
    return "reference"


def _bash_file_operation(cmd: str) -> str:
    categories = [
        _bash_segment_category(segment) for segment in parse_bash_segments(cmd)
    ]
    if "write" in categories:
        return "write"
    if "query" in categories:
        return "query"
    if "read" in categories:
        return "read"
    return "reference"


def _file_tool_status(
    row: RowMapping,
    state: object,
    processed: object,
    error: bool,
) -> str:
    if row["phase"] == "pre":
        return "intent"
    if error:
        category = _tool_error_category(state, processed) or "unknown"
        exit_code = _tool_exit_code(row, processed)
        if exit_code is not None:
            return f"exit:{exit_code} {category}"
        return f"error:{category}"
    return f"ok:{_dash(_tool_exit_code(row, processed))}"


def _file_operation_status(tool_name: str, phase: str, tool_status: str) -> str:
    if phase == "pre":
        return "intent"
    if tool_name == "bash":
        return "observed"
    return tool_status


def _file_evidence_label(event: _FileEvent, source: str) -> str:
    if event.tool_name != "bash":
        return "structured"
    if event.phase == "pre":
        return "intent"
    source_parts = set(source.split("+"))
    if "cmd.query" in source_parts:
        return "query"
    if source_parts.intersection(
        {"cmd.reader", "cmd.redirect.read", "cmd.redirect.write"}
    ):
        return "shell-op"
    return "reference"


def _tool_exit_code(row: RowMapping, processed: object) -> int | None:
    exit_code = _coerce_int(row["exit_code"])
    if exit_code is not None:
        return exit_code
    if isinstance(processed, Mapping):
        return _coerce_int(processed.get("exit_code"))
    return None


def _file_event_json(event: _FileEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "ts": event.ts,
        "turn": event.turn,
        "phase": event.phase,
        "tool_name": event.tool_name,
        "operation": event.operation,
        "path": event.path,
        "source": event.source,
        "args_hash": event.args_hash,
        "status": event.tool_status,
        "file_status": event.file_status,
        "tool_status": event.tool_status,
        "tool_exit_code": event.tool_exit_code,
        "duration_ms": event.duration_ms,
        "error_category": event.error_category,
        "error_fingerprint": event.error_fingerprint,
        "result_content_hash": event.result_content_hash,
        "content_hash": event.content_hash,
        "previous_content_hash": event.previous_content_hash,
    }


def _merged_file_event_json(event: _MergedFileEvent) -> dict[str, object]:
    data = _file_event_json(event.event)
    data["phase"] = event.phase_label
    data["source"] = event.source_label
    data["raw_event_count"] = len(event.events)
    return data


def _latest_file_event(history: Sequence[_MergedFileEvent]) -> str:
    if not history:
        return "-"
    event = history[-1]
    return (
        f"T{event.event.turn} {event.event.operation}/"
        f"{event.phase_label} {event.event.tool_name}"
    )


def _file_content_signals(
    history: Sequence[_MergedFileEvent],
) -> list[dict[str, object]]:
    signals: list[dict[str, object]] = []
    for merged in history:
        event = merged.event
        text = _file_event_content_text(event)
        if not any(
            (
                event.content_hash,
                event.previous_content_hash,
                event.result_content_hash,
                text,
            )
        ):
            continue
        tokens = _content_identifier_tokens(text)
        signals.append(
            {
                "turn": event.turn,
                "operation": event.operation,
                "phase": merged.phase_label,
                "content_hash": event.content_hash,
                "previous_content_hash": event.previous_content_hash,
                "result_content_hash": event.result_content_hash,
                "content_preview": _single_line(text, limit=240) if text else "",
                "identifier_tokens": [
                    {"token": token, "count": count}
                    for token, count in Counter(tokens).most_common(30)
                ],
            }
        )
    return signals


def _file_event_content_text(event: _FileEvent) -> str:
    if event.tool_name != "read" or event.operation != "read":
        return ""
    raw = _tool_result_text_for_rule(event.policy_tool_event.get("result"))
    return raw if isinstance(raw, str) else ""


def _content_identifier_tokens(text: str) -> list[str]:
    if not text:
        return []
    return [match.group(0) for match in _IDENTIFIER_RE.finditer(text[:200_000])]


def _command_text(args: object) -> str:
    if not isinstance(args, Mapping):
        return ""
    raw = args.get("cmd") or args.get("command")
    return raw if isinstance(raw, str) else ""


def _paths_from_command(
    cmd: str,
    *,
    known_paths: set[str],
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    for segment in parse_bash_segments(cmd):
        command = segment.argv
        command_category = _bash_segment_category(segment)
        candidates.extend(
            _known_paths_from_tokens(
                command,
                known_paths,
                source=_known_path_source_for_command(command_category),
            )
        )
        candidates.extend(_redirect_paths_from_segment(segment))
        candidates.extend(_reader_paths_from_command(command))
        candidates.extend(_query_paths_from_command(command))

    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for path, source in candidates:
        clean = _clean_path(path)
        if not clean or _is_virtual_filesystem_path(clean):
            continue
        item = (clean, source)
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _strip_heredoc_bodies(cmd: str) -> str:
    lines = cmd.splitlines()
    if not lines:
        return cmd

    kept: list[str] = []
    markers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if markers:
            if stripped == markers[0]:
                markers.pop(0)
            continue

        kept.append(line)
        for match in _HEREDOC_RE.finditer(line):
            marker = match.group("quoted") or match.group("bare")
            if marker:
                markers.append(marker)
    return "\n".join(kept)


def _shell_tokens(cmd: str) -> list[str]:
    try:
        lexer = shlex.shlex(
            cmd.replace("\n", " ; "), posix=True, punctuation_chars=True
        )
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return cmd.split()


def _shell_commands(tokens: Sequence[str]) -> Iterable[list[str]]:
    command: list[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            if command:
                yield command
                command = []
            continue
        command.append(token)
    if command:
        yield command


def _bash_segment_template_tokens(segment: BashSegment) -> tuple[str, ...]:
    tokens = list(_bash_command_template_tokens(segment.argv))
    for redirect in segment.redirects:
        operator = redirect.operator
        if redirect.descriptor:
            operator = f"{redirect.descriptor}{operator}"
        if operator:
            tokens.append(operator)
        if redirect.kind == "heredoc":
            tokens.append("<heredoc>")
        elif redirect.destination:
            tokens.append(_normalized_bash_template_token(redirect.destination))
    return tuple(tokens)


def _bash_segment_text(segment: BashSegment) -> str:
    if not segment.redirects or segment.parser == "shlex-fallback":
        return segment.text
    return " ".join([segment.text, *(redirect.text for redirect in segment.redirects)])


def _bash_command_template_tokens(command: Sequence[str]) -> tuple[str, ...]:
    pattern_indexes = _bash_pattern_token_indexes(command)
    return tuple(
        _bash_template_token(command, index, pattern_indexes)
        for index in range(len(command))
    )


def _bash_pattern_token_indexes(command: Sequence[str]) -> set[int]:
    name = _command_name(command)
    if name == "timeout":
        inner_index = _timeout_inner_index(command)
        inner_indexes = _bash_pattern_token_indexes(command[inner_index:])
        return {inner_index + index for index in inner_indexes}
    indexes: set[int] = set()
    if name in {"grep", "rg", "ag", "ack", "fd"} or _is_git_query(command):
        operands = _query_operand_indexes(command)
        if operands:
            indexes.add(operands[0])
    if name == "find":
        for index, token in enumerate(command[:-1]):
            if token in {"-name", "-path", "-regex", "-wholename"}:
                indexes.add(index + 1)
    return indexes


def _bash_template_token(
    command: Sequence[str],
    index: int,
    pattern_indexes: set[int],
) -> str:
    token = command[index]
    if index in pattern_indexes:
        return "<pattern>"
    if _preserve_bash_template_token(command, index):
        return _normalize_flag_assignment(token)
    return _normalized_bash_template_token(token)


def _preserve_bash_template_token(command: Sequence[str], index: int) -> bool:
    token = command[index]
    name = _command_name(command)
    if index == 0 or token in _SHELL_REDIRECTS:
        return True
    if token.startswith("-") and "=" not in token:
        return True
    if name in _SUBCOMMAND_TOOLS and index == 1 and not token.startswith("-"):
        return True
    if name == "timeout":
        inner_index = _timeout_inner_index(command)
        if index == inner_index:
            return True
        if inner_index < len(command):
            inner_name = _command_name(command[inner_index:])
            if inner_name in _SUBCOMMAND_TOOLS and index in {
                inner_index + 1,
                inner_index + 2,
            }:
                return not token.startswith("-")
    return False


def _normalize_flag_assignment(token: str) -> str:
    if token.startswith("-") and "=" in token:
        key, value = token.split("=", 1)
        return f"{key}={_normalized_bash_template_token(value)}"
    return token


def _normalized_bash_template_token(token: str) -> str:
    if not token:
        return "<arg>"
    if _NUMERIC_TOKEN_RE.fullmatch(token):
        return "<num>"
    if _HEX_TOKEN_RE.fullmatch(token):
        return "<hash>"
    if _is_probable_shell_path(token):
        return "<path>"
    if _looks_like_glob_or_pattern(token):
        return "<pattern>"
    if any(ch.isspace() for ch in token) or len(token) > 80:
        return "<arg>"
    if any(ch in token for ch in "{}()[],:"):
        return "<arg>"
    return token


def _looks_like_glob_or_pattern(token: str) -> bool:
    return any(char in token for char in "*?[]")


def _query_operand_indexes(command: Sequence[str]) -> list[int]:
    indexes: list[int] = []
    after_options = False
    for index, token in enumerate(command[1:], start=1):
        if token == "--":
            after_options = True
            continue
        if not after_options and token.startswith("-"):
            continue
        indexes.append(index)
    return indexes


def _known_paths_from_tokens(
    tokens: Sequence[str],
    known_paths: set[str],
    *,
    source: str,
) -> Iterable[tuple[str, str]]:
    if not known_paths:
        return ()
    known_by_variant: dict[str, str] = {}
    for path in known_paths:
        for variant in _known_path_variants(path):
            known_by_variant.setdefault(variant, path)

    matches: list[tuple[str, str]] = []
    for token in tokens:
        clean = _clean_path(token)
        if not clean:
            continue
        known = known_by_variant.get(clean)
        if known is not None:
            matches.append((known, source))
    return matches


def _known_path_source_for_command(category: str) -> str:
    if category == "query":
        return "cmd.query"
    if category == "read":
        return "cmd.reader"
    return "cmd.known"


def _known_path_variants(path: str) -> Iterable[str]:
    clean = _clean_path(path)
    if not clean:
        return ()
    variants = [clean]
    if clean.startswith("/"):
        parts = [part for part in clean.split("/") if part]
        variants.extend("/".join(parts[index:]) for index in range(1, len(parts) - 1))
    return variants


def _redirect_paths_from_segment(segment: BashSegment) -> Iterable[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for redirect in segment.redirects:
        if redirect.kind != "file":
            continue
        if redirect.operator in {"<", "<>"}:
            source = "cmd.redirect.read"
        elif redirect.operator in {">", ">>", ">|", "&>", "&>>"}:
            source = "cmd.redirect.write"
        else:
            continue
        if _is_probable_shell_path(redirect.destination):
            paths.append((redirect.destination, source))
    return paths


def _redirect_paths_from_command(command: Sequence[str]) -> Iterable[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for index, token in enumerate(command[:-1]):
        if token not in _SHELL_REDIRECTS:
            continue
        target = command[index + 1]
        if token in {"<", "<>"}:
            source = "cmd.redirect.read"
        elif token in {"<<", "<<-", "<<<"}:
            continue
        else:
            source = "cmd.redirect.write"
        if _is_probable_shell_path(target):
            paths.append((target, source))
    return paths


def _reader_paths_from_command(command: Sequence[str]) -> Iterable[tuple[str, str]]:
    if not command:
        return ()
    if _bash_command_category(command) != "read":
        return ()

    paths: list[tuple[str, str]] = []
    skip_next = False
    for token in command[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in _SHELL_REDIRECTS:
            skip_next = token not in {"<<", "<<-", "<<<"}
            continue
        if token.startswith("-"):
            continue
        if _is_probable_shell_path(token):
            paths.append((token, "cmd.reader"))
    return paths


def _query_paths_from_command(command: Sequence[str]) -> Iterable[tuple[str, str]]:
    if _bash_command_category(command) != "query":
        return ()
    name = _command_name(command)
    if name == "find":
        return _find_query_paths(command)
    if name in {"ls", "tree"}:
        return _all_path_operands(command[1:], "cmd.query")
    operands = _query_operands(command)
    if name in {"grep", "rg", "ag", "ack", "fd"} or _is_git_query(command):
        operands = operands[1:]
    return _all_path_operands(operands, "cmd.query")


def _query_operands(command: Sequence[str]) -> list[str]:
    return [command[index] for index in _query_operand_indexes(command)]


def _find_query_paths(command: Sequence[str]) -> Iterable[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for token in command[1:]:
        if token in {"!", "(", ")"} or token.startswith("-"):
            break
        if _is_probable_shell_path(token):
            paths.append((token, "cmd.query"))
    return paths


def _all_path_operands(
    operands: Iterable[str],
    source: str,
) -> Iterable[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for token in operands:
        if token.startswith("-"):
            continue
        if _is_probable_shell_path(token):
            paths.append((token, source))
    return paths


def _bash_segment_category(segment: BashSegment) -> str:
    if _segment_has_write_redirect(segment):
        return "write"
    return _bash_command_category(segment.argv)


def _segment_has_write_redirect(segment: BashSegment) -> bool:
    for redirect in segment.redirects:
        if redirect.kind != "file" or redirect.operator not in {
            ">",
            ">>",
            "<>",
            ">|",
            "&>",
            "&>>",
        }:
            continue
        if redirect.descriptor == "2":
            continue
        if _is_virtual_filesystem_path(_clean_path(redirect.destination) or ""):
            continue
        return True
    return False


def _bash_command_category(command: Sequence[str]) -> str:
    name = _command_name(command)
    if not name:
        return "reference"
    if _has_write_redirect(command):
        return "write"
    if name == "timeout":
        return _bash_command_category(_timeout_inner_command(command))
    if _is_git_query(command) or name in _FILE_QUERY_COMMANDS:
        return "query"
    if name == "sed" and any(token.startswith("-i") for token in command[1:]):
        return "write"
    if name == "perl" and any(
        "i" in token for token in command[1:] if token.startswith("-")
    ):
        return "write"
    if name in _FILE_READER_COMMANDS:
        return "read"
    if name in _FILE_WRITER_COMMANDS:
        return "write"
    if name == "xargs" and any(
        _command_name((token,)) in _FILE_QUERY_COMMANDS for token in command[1:]
    ):
        return "query"
    return "reference"


def _bash_segment_family(segment: BashSegment) -> str:
    category = _bash_segment_category(segment)
    if category != "reference":
        return category
    name = _command_name(segment.argv)
    if name in _CONTROL_COMMANDS:
        return "control"
    if name:
        return "exec"
    return "unknown"


def _has_write_redirect(command: Sequence[str]) -> bool:
    for index, token in enumerate(command[:-1]):
        if token in {">", ">>", "<>", ">|", "&>", "&>>"}:
            if index > 0 and command[index - 1] == "2":
                continue
            if _is_virtual_filesystem_path(_clean_path(command[index + 1]) or ""):
                continue
            return True
    return False


def _bash_command_family(command: Sequence[str]) -> str:
    category = _bash_command_category(command)
    if category != "reference":
        return category
    name = _command_name(command)
    if name in _CONTROL_COMMANDS:
        return "control"
    if name:
        return "exec"
    return "unknown"


def _timeout_inner_command(command: Sequence[str]) -> Sequence[str]:
    return command[_timeout_inner_index(command) :]


def _timeout_inner_index(command: Sequence[str]) -> int:
    for index, token in enumerate(command[1:], start=1):
        if token.startswith("-") or token.replace(".", "", 1).isdigit():
            continue
        return index
    return len(command)


def _is_git_query(command: Sequence[str]) -> bool:
    return (
        len(command) > 1
        and _command_name(command) == "git"
        and command[1] in _GIT_QUERY_SUBCOMMANDS
    )


def _command_name(command: Sequence[str]) -> str:
    if not command:
        return ""
    return Path(command[0]).name


def _is_probable_shell_path(value: object) -> bool:
    path = _clean_path(value)
    if not path or _is_virtual_filesystem_path(path):
        return False
    return path.startswith(("/", "./", "../")) or "/" in path or "." in path


def _is_virtual_filesystem_path(path: str) -> bool:
    parts = [part for part in path.split("/") if part]
    return bool(path.startswith("/") and parts and parts[0] in {"dev", "proc", "sys"})


def _clean_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.strip().strip("\"'`,;:()[]{}")
    if not path or len(path) < 3:
        return None
    if path.startswith("-") or "*" in path or "://" in path:
        return None
    return path


def _entity_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM policy_entity_state
        WHERE session_id = ?
        ORDER BY occurrence_count DESC, updated_at DESC, entity ASC
        """,
        (session_id,),
    ):
        evidence = _loads(row["evidence_json"])
        yield TraceRow(
            key=f"policy:entity:{row['entity']}",
            kind="policy",
            title=row["entity"],
            preview=(
                f"type:{row['entity_type']} count:{row['occurrence_count']} "
                f"evidence:{_evidence_preview(evidence)}"
            ),
            content=_json_content(
                {
                    "session_id": row["session_id"],
                    "entity": row["entity"],
                    "entity_type": row["entity_type"],
                    "updated_at": row["updated_at"],
                    "first_seen_turn": row["first_seen_turn"],
                    "last_seen_turn": row["last_seen_turn"],
                    "occurrence_count": row["occurrence_count"],
                    "evidence": evidence,
                }
            ),
            turn_index=row["last_seen_turn"],
            metadata={
                "category": "entity",
                "entity_type": row["entity_type"],
                "occurrence_count": row["occurrence_count"],
            },
        )


def _eval_error_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM policy_eval_error
        WHERE session_id = ?
        ORDER BY ts ASC, id ASC
        """,
        (session_id,),
    ):
        yield TraceRow(
            key=f"policy:eval_error:{row['id']}",
            kind="policy",
            title=row["rule_id"],
            preview=row["error"],
            content=_json_content(dict(row)),
            turn_index=row["turn"],
            tool_name=row["tool_name"],
            is_error=True,
            cause="eval_error",
            metadata={"category": "eval_error", "channel": row["channel"]},
        )


def _diagnostic_row(
    session_id: str,
    *,
    title: str,
    preview: str,
    content: str,
) -> TraceRow:
    return TraceRow(
        key=f"policy:diagnostic:{session_id}:{title}",
        kind="policy",
        title=title,
        preview=preview,
        content=content,
        is_error=True,
        cause="missing",
        metadata={"category": "diagnostic"},
    )


def _empty_category_row(
    session_id: str,
    db_path: Path,
    counts: Mapping[str, int],
    categories: frozenset[str],
) -> TraceRow:
    title = "No Policy Projection"
    preview = "No persisted policy rows for this session."
    hint = (
        "Run policy_engine backfill --session "
        f"{session_id} to compute policy projection rows."
    )
    if categories == _ERROR_CATEGORIES:
        title = "No Policy Eval Errors"
        preview = "No policy evaluation errors for this session."
        hint = "This session has no persisted policy evaluation errors."
    elif categories == _EFFECT_CATEGORIES:
        title = "No Policy Effects"
        preview = "No policy rule firings for this session."
        hint = "This session has no persisted policy rule firings."
    elif categories == _TOOL_CATEGORIES:
        title = "No Policy Tool Events"
        preview = "No policy-observed tool events for this session."
    elif categories == _FILE_CATEGORIES:
        title = "No Policy File State"
        preview = "No persisted policy file state for this session."
    elif categories == _FILE_STREAM_CATEGORIES:
        title = "No Policy File Stream"
        preview = "No structured non-bash file events could be derived."
    elif categories == _FILE_TOOL_STREAM_CATEGORIES:
        title = "No Policy File Tool Stream"
        preview = "No direct read/write/edit file-tool events could be derived."
    elif categories == _BASH_FILE_STREAM_CATEGORIES:
        title = "No Policy Bash File Stream"
        preview = "No bash-derived file events could be derived for this session."
    elif categories == _BASH_COMMAND_CATEGORIES:
        title = "No Policy Bash Commands"
        preview = "No bash command templates could be derived for this session."
    elif categories == _ENTITY_CATEGORIES:
        title = "No Policy Entity State"
        preview = "No persisted policy entity state for this session."

    return TraceRow(
        key=f"policy:empty:{session_id}:{','.join(sorted(categories))}",
        kind="policy",
        title=title,
        preview=preview,
        content=_json_content(
            {
                "session_id": session_id,
                "db_path": str(db_path),
                "counts": dict(counts),
                "categories": sorted(categories),
                "hint": hint,
            }
        ),
        metadata={"category": "diagnostic"},
    )


def _existing_tables(conn: Connection) -> set[str]:
    return {
        str(row["name"])
        for row in _rows(conn, "SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _session_counts(
    conn: Connection,
    session_id: str,
    existing: set[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in _POLICY_TABLES:
        if table not in existing:
            counts[table] = 0
            continue
        row = _row(
            conn,
            f"SELECT COUNT(*) AS count FROM {table} WHERE session_id = ?",  # noqa: S608
            (session_id,),
        )
        counts[table] = int(row["count"]) if row is not None else 0
    return counts


def _loads(raw: object) -> object:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _json_content(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _json_preview(value: object, *, limit: int = 150) -> str:
    if value is None:
        return ""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _counts_preview(counts: Mapping[str, int]) -> str:
    parts = [
        f"effects:{counts.get('event_log', 0)}",
        f"tools:{counts.get('policy_tool_events', 0)}",
        f"files:{counts.get('policy_file_state', 0)}",
        f"entities:{counts.get('policy_entity_state', 0)}",
        f"errors:{counts.get('policy_eval_error', 0)}",
    ]
    return " ".join(parts)


def _turn_summary_preview(summary: object) -> str:
    if not isinstance(summary, Mapping):
        return _json_preview(summary)
    parts = []
    for key in (
        "tool_calls_count",
        "tool_log_entries",
        "effect_log_entries",
        "error_count",
        "context_tokens",
        "total_context_tokens",
    ):
        if key in summary:
            parts.append(f"{key}:{summary[key]}")
    return " ".join(parts) or _json_preview(summary)


def _tool_policy_state_label(
    row: RowMapping,
    state: object,
    processed: object,
    is_error: bool,
) -> str:
    phase = str(row["phase"])
    if phase == "pre":
        labels = _taint_labels(state)
        joined = ",".join(labels) if labels else "-"
        return f"pre taint:{joined}"
    if is_error:
        category = _tool_error_category(state, processed) or "error"
        return f"post err:{category}"
    exit_code = row["exit_code"]
    if exit_code is None and isinstance(processed, Mapping):
        exit_code = processed.get("exit_code")
    return f"post exit:{_dash(exit_code)}"


def _tool_policy_preview(
    row: RowMapping,
    args: object,
    result: object,
    state: object,
    processed: object,
) -> str:
    phase = str(row["phase"])
    parts = [f"hash:{_short_text(row['args_hash'], 8)}"]
    if phase == "pre":
        labels = _taint_labels(state)
        parts.append(f"taint:{','.join(labels) if labels else '-'}")
    else:
        parts.extend(
            [
                f"exit:{_dash(row['exit_code'])}",
                f"err:{_dash(_tool_error_category(state, processed))}",
                f"fp:{_dash(_tool_error_fingerprint(state, processed))}",
                f"result:{_dash(row['result_content_hash'])}",
                f"len:{_dash(_tool_result_length(result, processed, state))}",
            ]
        )
    source = _tool_source_preview(args, result, state, processed)
    if source:
        parts.append(source)
    return " ".join(parts)


def _tool_rule_author_event(
    row: RowMapping,
    args: object,
    result: object,
    state: object,
    processed: object,
) -> dict[str, object]:
    phase = str(row["phase"])
    if phase == "pre":
        return {
            "channel": "tool_call_pre",
            "event.tool_name": row["tool_name"],
            "event.args": args,
            "event.result": {},
            "event.taint": _taint_labels(state),
            "query_notes": [
                "This is the current event inspected by rules on tool_call_pre.",
                "Rules can also query prior calls via tool_log/file_state/entity_evidence.",
            ],
        }

    return {
        "channel": "tool_call_post",
        "event.tool_name": row["tool_name"],
        "event.args": {},
        "event.result": {
            "text": _tool_result_text_for_rule(result),
            "error": _tool_error_text(result, processed, state),
        },
        "event.taint": [],
        "current_tool_args": args,
        "query_notes": [
            "Current tool args are persisted for debugging here, but are not passed "
            "directly as event.args to tool_call_post predicates today.",
            "Post-result rules usually query the recorded call through tool_log.last(...).",
        ],
    }


def _tool_preview(
    args: object,
    result: object,
    state: object,
    processed: object,
) -> str:
    for value in _candidate_values(args, ("cmd", "command", "path", "file_path")):
        return str(value)
    for value in _candidate_values(processed, ("error", "error_category", "summary")):
        return str(value)
    for value in _candidate_values(result, ("error", "message", "output", "content")):
        return str(value)
    for value in _candidate_values(state, ("processed", "tool_log_entry")):
        return _json_preview(value)
    return _json_preview({"args": args, "result": result, "processed": processed})


def _tool_source_preview(
    args: object,
    result: object,
    state: object,
    processed: object,
) -> str:
    for value in _candidate_values(args, ("cmd", "command", "path", "file_path")):
        return _single_line(value)
    for value in _candidate_values(processed, ("path", "file_op", "summary")):
        return _single_line(value)
    for value in _candidate_values(result, ("error", "message", "output", "content")):
        return _single_line(value)
    for value in _candidate_values(state, ("tool_log_entry", "processed")):
        return _json_preview(value, limit=80)
    return ""


def _candidate_values(value: object, keys: Sequence[str]) -> Iterable[object]:
    if not isinstance(value, Mapping):
        return ()
    return (value[key] for key in keys if key in value and value[key] not in (None, ""))


def _tool_is_error(
    row: RowMapping,
    result: object,
    state: object,
    processed: object,
) -> bool:
    if row["exit_code"] not in (None, 0):
        return True
    for value in (result, processed):
        if isinstance(value, Mapping) and value.get("is_error") is True:
            return True
    if isinstance(state, Mapping):
        entry = state.get("tool_log_entry")
        if isinstance(entry, Mapping):
            return bool(entry.get("error") or entry.get("error_category"))
    return False


def _taint_labels(state: object) -> list[str]:
    if not isinstance(state, Mapping):
        return []
    raw = state.get("taint_labels")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _tool_error_category(state: object, processed: object) -> str | None:
    value = _mapping_value(processed, "error_category")
    if value:
        return str(value)
    entry = _tool_log_entry(state)
    value = _mapping_value(entry, "error_category")
    return str(value) if value else None


def _tool_error_fingerprint(state: object, processed: object) -> str | None:
    value = _mapping_value(processed, "error_fingerprint")
    if value:
        return str(value)
    entry = _tool_log_entry(state)
    value = _mapping_value(entry, "error_fingerprint")
    return str(value) if value else None


def _tool_error_text(result: object, processed: object, state: object) -> object:
    for value in _candidate_values(processed, ("error",)):
        return value
    for value in _candidate_values(result, ("error", "message")):
        return value
    entry = _tool_log_entry(state)
    value = _mapping_value(entry, "error")
    return value


def _tool_result_length(result: object, processed: object, state: object) -> object:
    for value in _candidate_values(processed, ("result_length", "text_length")):
        return value
    entry = _tool_log_entry(state)
    value = _mapping_value(entry, "result_length")
    if value is not None:
        return value
    text = _tool_result_text_for_rule(result)
    return len(text) if isinstance(text, str) else None


def _tool_result_text_for_rule(result: object) -> object:
    if isinstance(result, Mapping):
        text = result.get("text")
        if isinstance(text, str):
            return text
        content = result.get("content")
        if isinstance(content, list):
            parts = [
                text
                for item in content
                if isinstance(item, Mapping)
                and isinstance((text := item.get("text")), str)
            ]
            return "".join(parts) if parts else None
    return None


def _tool_log_entry(state: object) -> Mapping[str, object]:
    if not isinstance(state, Mapping):
        return {}
    raw = state.get("tool_log_entry")
    return raw if isinstance(raw, Mapping) else {}


def _mapping_value(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _mapping_str(value: object, key: str) -> str | None:
    raw = _mapping_value(value, key)
    return str(raw) if raw not in (None, "") else None


def _single_line(value: object, *, limit: int = 100) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _short_text(value: object, keep: int) -> str:
    text = "" if value is None else str(value)
    return text[:keep]


def _stable_id(value: object, *, keep: int = 16) -> str:
    raw = "" if value is None else str(value)
    return hashlib.sha256(raw.encode()).hexdigest()[:keep]


def _evidence_preview(evidence: object) -> str:
    if not isinstance(evidence, list):
        return _json_preview(evidence)
    counts: dict[str, int] = {}
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("type") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _last_turn(*values: object) -> int | None:
    ints = [coerced for value in values if (coerced := _coerce_int(value)) is not None]
    return max(ints) if ints else None


def _dash(value: object) -> str:
    return "-" if value is None else str(value)


def _summary_text(
    db_path: Path,
    rows: Sequence[TraceRow],
    filtered: Sequence[TraceRow],
) -> str:
    return (
        f"{len(filtered)} row(s) | total:{len(rows)} | policy_db:{_short_path(db_path)}"
    )


def _short_path(path: Path) -> str:
    try:
        return str(path.expanduser()).replace(str(Path.home()), "~", 1)
    except RuntimeError:
        return str(path)


__all__ = [
    "PolicyTraceViewProvider",
    "build_policy_trace_view_registry",
    "default_policy_db_path",
    "load_policy_trace_rows",
    "run_policy_trace_viewer",
]
