# code-health: ignore-file[AM025] -- policy engine normalizes untyped YAML, SQLite, and runtime event payloads
"""Policy-owned trace view provider for the shared AgentM Textual viewer."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentm.cli._trace_model import (
    TraceQuery,
    TraceRow,
    TraceSnapshot,
    TraceView,
    TraceViewRegistry,
    TraceViewSpec,
    default_trace_view_specs,
    filter_trace_rows,
)
from agentm.core.abi.query import TrajectoryQueryStore

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
_ENTITY_CATEGORIES = frozenset({"entities"})
_ERROR_CATEGORIES = frozenset({"errors"})


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
                description="Per-file policy state snapshots.",
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
        )


def build_policy_trace_view_registry(db_path: Path | None = None) -> TraceViewRegistry:
    """Build the default trace registry with the policy placeholder replaced."""

    registry = TraceViewRegistry(
        spec for spec in default_trace_view_specs() if spec.id != "policy"
    )
    registry.extend(PolicyTraceViewProvider(db_path=db_path))
    return registry


def run_policy_trace_viewer(
    query: TrajectoryQueryStore,
    session_id: str,
    *,
    db_path: Path | None = None,
    follow: bool = False,
) -> None:
    """Run AgentM's shared Textual trace viewer with policy projection tabs."""

    from agentm.cli._trace_textual import run_textual_viewer

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

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
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
        if "entities" in categories and "policy_entity_state" in existing:
            rows.extend(_entity_rows(conn, session_id))
        if "errors" in categories and "policy_eval_error" in existing:
            rows.extend(_eval_error_rows(conn, session_id))
        if not rows:
            rows.append(_empty_category_row(session_id, db_path, counts, categories))
        return tuple(rows)
    finally:
        conn.close()


def _summary_rows(
    conn: sqlite3.Connection,
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
        row = conn.execute(
            """
            SELECT updated_at, summary_json
            FROM policy_session_summary
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
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
        for row in conn.execute(
            """
            SELECT turn_index, updated_at, summary_json
            FROM policy_turn_summary
            WHERE session_id = ?
            ORDER BY turn_index ASC
            """,
            (session_id,),
        ).fetchall():
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
    conn: sqlite3.Connection,
    session_id: str,
    category: str,
) -> TraceRow:
    if category == "effects":
        data = [
            dict(row)
            for row in conn.execute(
                """
                SELECT rule_id, mode, effect, COUNT(*) AS count
                FROM event_log
                WHERE session_id = ?
                GROUP BY rule_id, mode, effect
                ORDER BY count DESC, rule_id ASC
                """,
                (session_id,),
            ).fetchall()
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
            for row in conn.execute(
                """
                SELECT tool_name, phase, COUNT(*) AS count
                FROM policy_tool_events
                WHERE session_id = ?
                GROUP BY tool_name, phase
                ORDER BY count DESC, tool_name ASC, phase ASC
                """,
                (session_id,),
            ).fetchall()
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
            for row in conn.execute(
                """
                SELECT entity_type, COUNT(*) AS count,
                       SUM(occurrence_count) AS occurrences
                FROM policy_entity_state
                WHERE session_id = ?
                GROUP BY entity_type
                ORDER BY count DESC, entity_type ASC
                """,
                (session_id,),
            ).fetchall()
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
        for row in conn.execute(
            """
            SELECT path, read_count, write_count, first_read_turn,
                   last_read_turn, last_write_turn, content_hash
            FROM policy_file_state
            WHERE session_id = ?
            ORDER BY (read_count + write_count) DESC, path ASC
            LIMIT 50
            """,
            (session_id,),
        ).fetchall()
    ]
    return TraceRow(
        key=f"policy:aggregate:file_hotspots:{session_id}",
        kind="policy",
        title="File Hotspots",
        preview=_json_preview(data),
        content=_json_content(data),
        metadata={"category": "summary", "aggregate": "file_hotspots"},
    )


def _effect_rows(conn: sqlite3.Connection, session_id: str) -> Iterable[TraceRow]:
    for row in conn.execute(
        """
        SELECT *
        FROM event_log
        WHERE session_id = ?
        ORDER BY ts ASC, id ASC
        """,
        (session_id,),
    ).fetchall():
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


def _tool_rows(conn: sqlite3.Connection, session_id: str) -> Iterable[TraceRow]:
    for row in conn.execute(
        """
        SELECT *
        FROM policy_tool_events
        WHERE session_id = ?
        ORDER BY turn ASC, id ASC
        """,
        (session_id,),
    ).fetchall():
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


def _file_rows(conn: sqlite3.Connection, session_id: str) -> Iterable[TraceRow]:
    for row in conn.execute(
        """
        SELECT *
        FROM policy_file_state
        WHERE session_id = ?
        ORDER BY (read_count + write_count) DESC, path ASC
        """,
        (session_id,),
    ).fetchall():
        last_turn = _last_turn(row["last_write_turn"], row["last_read_turn"])
        state = f"reads:{row['read_count']} writes:{row['write_count']}"
        yield TraceRow(
            key=f"policy:file:{row['path']}",
            kind="policy",
            title=row["path"],
            preview=(
                f"{state} first:{_dash(row['first_read_turn'])} "
                f"last:{_dash(last_turn)} hash:{row['content_hash'] or '-'}"
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
                }
            ),
            turn_index=last_turn,
            metadata={"category": "file", "path": row["path"], "state": state},
        )


def _entity_rows(conn: sqlite3.Connection, session_id: str) -> Iterable[TraceRow]:
    for row in conn.execute(
        """
        SELECT *
        FROM policy_entity_state
        WHERE session_id = ?
        ORDER BY occurrence_count DESC, updated_at DESC, entity ASC
        """,
        (session_id,),
    ).fetchall():
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


def _eval_error_rows(conn: sqlite3.Connection, session_id: str) -> Iterable[TraceRow]:
    for row in conn.execute(
        """
        SELECT *
        FROM policy_eval_error
        WHERE session_id = ?
        ORDER BY ts ASC, id ASC
        """,
        (session_id,),
    ).fetchall():
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


def _existing_tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def _session_counts(
    conn: sqlite3.Connection,
    session_id: str,
    existing: set[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in _POLICY_TABLES:
        if table not in existing:
            counts[table] = 0
            continue
        row = conn.execute(
            f"SELECT COUNT(*) AS count FROM {table} WHERE session_id = ?",  # noqa: S608
            (session_id,),
        ).fetchone()
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
    row: sqlite3.Row,
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
    row: sqlite3.Row,
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
    row: sqlite3.Row,
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
    row: sqlite3.Row,
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


def _single_line(value: object, *, limit: int = 100) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _short_text(value: object, keep: int) -> str:
    text = "" if value is None else str(value)
    return text[:keep]


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
