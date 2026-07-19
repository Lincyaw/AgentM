"""Policy engine persistence — SQLite WAL for cross-session state."""

from __future__ import annotations

from loguru import logger
import sqlite3
import time
from pathlib import Path


from .types import EffectRecord


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
"""


class PolicyPersistence:
    """Cross-session state store backed by SQLite WAL."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._pending_effects: list[tuple[str, EffectRecord]] = []

    def open(self) -> None:
        if self._db_path is None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def queue_effect(self, session_id: str, record: EffectRecord) -> None:
        """Queue an effect record for batch write at turn end."""
        self._pending_effects.append((session_id, record))

    def flush(self) -> None:
        """Write all pending effects to disk in one transaction."""
        if not self._conn or not self._pending_effects:
            return

        now = time.time()
        try:
            with self._conn:
                for session_id, rec in self._pending_effects:
                    self._conn.execute(
                        "INSERT INTO event_log (ts, session_id, rule_id, mode, effect, reason, turn) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (now, session_id, rec.rule_id, rec.mode, rec.effect,
                         rec.reason, rec.turn),
                    )
            self._pending_effects.clear()
        except sqlite3.Error as e:
            logger.warning("policy persistence flush failed: {}", e)

    def count_cross_session(
        self,
        rule_id: str | None = None,
        error_fingerprint: str | None = None,
        ttl_days: int = 14,
    ) -> int:
        """Count matching events across sessions within TTL."""
        if not self._conn:
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
            row = self._conn.execute(sql, params).fetchone()
            return row[0] if row else 0
        except sqlite3.Error:
            return 0

    def prune(self, ttl_days: int = 30) -> int:
        """Remove records older than TTL. Returns count deleted."""
        if not self._conn:
            return 0
        cutoff = time.time() - (ttl_days * 86400)
        try:
            cursor = self._conn.execute(
                "DELETE FROM event_log WHERE ts < ?", (cutoff,)
            )
            self._conn.commit()
            return cursor.rowcount
        except sqlite3.Error:
            return 0
