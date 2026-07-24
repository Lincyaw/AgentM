# code-health: ignore-file[AM025] -- tool payloads are untyped at the boundary
"""Raw-data recording: every tool result into a per-session SQLite file.

Schema-compatible with the previous ``policy_tool_events`` table so all
existing session databases, replay tooling and calibration corpora keep
working. Only ``post`` rows are written (the replay consumers never used
``pre`` rows); unused analysis columns stay NULL.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

_SCHEMA = """
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
"""

_RESULT_TEXT_CLIP = 2000


@dataclass(slots=True)
class ToolEventRecorder:
    """Append-only recorder for one session's tool events."""

    db_path: Path
    session_id: str
    _conn: sqlite3.Connection | None = field(default=None, repr=False)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            self._conn = conn
        return self._conn

    def record(
        self,
        *,
        turn: int,
        tool_name: str,
        tool_call_id: str | None,
        args: dict[str, object],
        is_error: bool,
        result_text: str,
        exit_code: int | None,
        duration_ms: int | None,
        cwd: str | None,
    ) -> None:
        result_json = json.dumps(
            {
                "is_error": is_error,
                "content": [{"type": "text", "text": result_text[:_RESULT_TEXT_CLIP]}],
            }
        )
        try:
            conn = self._connection()
            conn.execute(
                "INSERT INTO policy_tool_events "
                "(ts, session_id, turn, phase, tool_call_id, tool_name, "
                " args_json, result_json, exit_code, duration_ms, cwd) "
                "VALUES (?, ?, ?, 'post', ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    self.session_id,
                    turn,
                    tool_call_id,
                    tool_name,
                    json.dumps(args, default=str),
                    result_json,
                    exit_code,
                    duration_ms,
                    cwd,
                ),
            )
            conn.commit()
        except sqlite3.Error as exc:
            logger.warning("policy recording failed for {}: {}", tool_name, exc)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.debug("policy recorder close: {}", exc)
            self._conn = None
