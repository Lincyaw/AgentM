"""SQLite WAL-backed default implementations of OutboxStore + InboxLog.

One SQLite file holds the outbox, dead-letter and inbox tables. WAL
mode + ``synchronous=NORMAL`` gives durable writes with reasonable
concurrency for a single-process server.

Crash recovery model
--------------------
Each lease sets ``leased_until = now + LEASE_TTL_SECONDS``. If the
server crashes between lease and ack, the row stays in the outbox
with an expired lease; the next :meth:`SqliteOutbox.lease` call after
``leased_until`` reissues it. There is no recovery thread inside the
store — the store is mechanism, not a daemon.

Thread safety
-------------
Each store/inbox owns a single sqlite3 connection opened with
``check_same_thread=False`` and serialises mutating operations behind
a ``threading.Lock``. The server's sender / reconnect-prefill paths call
into the store via ``asyncio.to_thread`` so this synchronous API stays
simple.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from agentm.gateway.wire import Envelope

from .errors import OutboxClosed
from .protocol import OutboxRecord

LEASE_TTL_SECONDS: float = 30.0


_OUTBOX_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    peer_id TEXT NOT NULL,
    envelope_id TEXT NOT NULL,
    envelope_json TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    enqueued_at REAL NOT NULL,
    next_retry_at REAL NOT NULL DEFAULT 0,
    leased_until REAL NOT NULL DEFAULT 0,
    UNIQUE(peer_id, envelope_id)
);
CREATE INDEX IF NOT EXISTS idx_outbox_peer_ready
    ON outbox(peer_id, next_retry_at, leased_until);

CREATE TABLE IF NOT EXISTS dead_letter (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    peer_id TEXT NOT NULL,
    envelope_id TEXT NOT NULL,
    envelope_json TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    reason TEXT NOT NULL,
    moved_at REAL NOT NULL
);
"""

_INBOX_SCHEMA = """
CREATE TABLE IF NOT EXISTS inbox_seen (
    peer_id TEXT NOT NULL,
    envelope_id TEXT NOT NULL,
    seen_at REAL NOT NULL,
    PRIMARY KEY (peer_id, envelope_id)
);
"""


def _open(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class SqliteOutbox:
    """Default OutboxStore implementation. See module docstring."""

    def __init__(
        self,
        path: str,
        *,
        lease_ttl: float = LEASE_TTL_SECONDS,
    ) -> None:
        self._path = path
        self._lease_ttl = lease_ttl
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = _open(path)
        self._conn.executescript(_OUTBOX_SCHEMA)

    # -- internal -----------------------------------------------------

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            raise OutboxClosed("outbox is closed")
        return self._conn

    # -- OutboxStore --------------------------------------------------

    def enqueue(self, peer_id: str, env: Envelope) -> int:
        payload = json.dumps(env.to_dict(), separators=(",", ":"))
        with self._lock:
            c = self._c()
            cur = c.execute(
                "INSERT OR IGNORE INTO outbox "
                "(peer_id, envelope_id, envelope_json, attempts, "
                " enqueued_at, next_retry_at, leased_until) "
                "VALUES (?, ?, ?, 0, ?, 0, 0)",
                (peer_id, env.id, payload, env.ts),
            )
            # Idempotent on (peer_id, env.id): a duplicate INSERT is IGNOREd
            # (rowcount 0), so resolve the existing row's id rather than
            # returning the unreliable lastrowid — the caller acks by this id.
            if cur.rowcount:
                row_id = int(cur.lastrowid or 0)
            else:
                row = c.execute(
                    "SELECT id FROM outbox WHERE peer_id = ? AND envelope_id = ?",
                    (peer_id, env.id),
                ).fetchone()
                row_id = int(row[0]) if row else 0
        return row_id

    def lease(
        self, peer_id: str, batch_max: int, now: float
    ) -> list[OutboxRecord]:
        if batch_max <= 0:
            return []
        with self._lock:
            c = self._c()
            rows = c.execute(
                "SELECT id, envelope_json, attempts, enqueued_at, next_retry_at "
                "FROM outbox "
                "WHERE peer_id = ? AND next_retry_at <= ? AND leased_until <= ? "
                "ORDER BY id ASC LIMIT ?",
                (peer_id, now, now, batch_max),
            ).fetchall()
            if not rows:
                return []
            ids = [int(r[0]) for r in rows]
            placeholders = ",".join("?" * len(ids))
            c.execute(
                f"UPDATE outbox SET attempts = attempts + 1, leased_until = ? "
                f"WHERE id IN ({placeholders})",
                (now + self._lease_ttl, *ids),
            )
            out: list[OutboxRecord] = []
            for row in rows:
                rid, env_json, attempts, enqueued_at, next_retry_at = row
                env = Envelope.from_dict(json.loads(env_json))
                out.append(
                    OutboxRecord(
                        id=int(rid),
                        peer_id=peer_id,
                        envelope=env,
                        attempts=int(attempts) + 1,
                        enqueued_at=float(enqueued_at),
                        next_retry_at=float(next_retry_at),
                    )
                )
            return out

    def ack(self, record_ids: list[int]) -> None:
        if not record_ids:
            return
        with self._lock:
            c = self._c()
            placeholders = ",".join("?" * len(record_ids))
            c.execute(
                f"DELETE FROM outbox WHERE id IN ({placeholders})",
                tuple(record_ids),
            )

    def nack(self, record_ids: list[int], next_retry_at: float) -> None:
        if not record_ids:
            return
        with self._lock:
            c = self._c()
            placeholders = ",".join("?" * len(record_ids))
            c.execute(
                f"UPDATE outbox SET leased_until = 0, next_retry_at = ? "
                f"WHERE id IN ({placeholders})",
                (next_retry_at, *record_ids),
            )

    def dead_letter(self, record_id: int, reason: str) -> None:
        with self._lock:
            c = self._c()
            c.execute("BEGIN")
            try:
                row = c.execute(
                    "SELECT peer_id, envelope_id, envelope_json, attempts "
                    "FROM outbox WHERE id = ?",
                    (record_id,),
                ).fetchone()
                if row is None:
                    c.execute("COMMIT")
                    return
                peer_id, envelope_id, env_json, attempts = row
                # Use the envelope ts as a deterministic moved_at fallback
                moved_at = _now_from_envelope_json(env_json)
                c.execute(
                    "INSERT INTO dead_letter "
                    "(peer_id, envelope_id, envelope_json, attempts, reason, moved_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (peer_id, envelope_id, env_json, attempts, reason, moved_at),
                )
                c.execute("DELETE FROM outbox WHERE id = ?", (record_id,))
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def pending_count(self, peer_id: str) -> int:
        with self._lock:
            c = self._c()
            row = c.execute(
                "SELECT COUNT(*) FROM outbox WHERE peer_id = ?", (peer_id,)
            ).fetchone()
            return int(row[0])

    def dead_letter_count(self, peer_id: str) -> int:
        """Test/observability helper — count dead-lettered rows for a peer."""
        with self._lock:
            c = self._c()
            row = c.execute(
                "SELECT COUNT(*) FROM dead_letter WHERE peer_id = ?", (peer_id,)
            ).fetchone()
            return int(row[0])

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


class SqliteInbox:
    """Default InboxLog implementation."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = _open(path)
        self._conn.executescript(_INBOX_SCHEMA)

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            raise OutboxClosed("inbox is closed")
        return self._conn

    def record_seen(self, peer_id: str, envelope_id: str, ts: float) -> bool:
        with self._lock:
            c = self._c()
            cur = c.execute(
                "INSERT OR IGNORE INTO inbox_seen (peer_id, envelope_id, seen_at) "
                "VALUES (?, ?, ?)",
                (peer_id, envelope_id, ts),
            )
            return cur.rowcount > 0

    def prune(self, older_than: float) -> int:
        with self._lock:
            c = self._c()
            cur = c.execute(
                "DELETE FROM inbox_seen WHERE seen_at < ?", (older_than,)
            )
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _now_from_envelope_json(env_json: str) -> float:
    """Use the envelope's own ts to stamp the dead-letter row.

    The store has no clock dependency — callers control time via
    envelope timestamps and the ``now`` arg to :meth:`lease`. Keeping
    a clock out of the store is what makes the lease-expiry test
    deterministic without mocking ``time.time``.
    """
    try:
        data: dict[str, Any] = json.loads(env_json)
        return float(data.get("ts", 0.0))
    except (ValueError, TypeError):
        return 0.0


__all__ = ["LEASE_TTL_SECONDS", "SqliteInbox", "SqliteOutbox"]
