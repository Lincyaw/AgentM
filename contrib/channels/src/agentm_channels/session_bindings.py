"""Persistent ``session_key → host`` binding store.

This is the gateway's routing table. A *session* is the first-class
routable thing; *workers* are just hosts that happen to be holding it.
A binding survives:

* worker / host disconnect — next inbound for the same ``session_key``
  picks any online host, and the new host resumes the AgentSession
  via the persisted ``resume_id`` so the conversation continues.
* gateway restart — the table lives in SQLite next to the outbox, so
  reconnecting peers find the same routing they had before.

The store is intentionally schema-thin (one table, four columns).
Anything richer (load-tracking, host capability fingerprints, last-
heard time series) belongs to a presence / load subsystem, not the
binding table.

Atomic writes only — concurrent ``upsert`` and ``get`` from multiple
asyncio tasks are safe because every write goes through a single
threading lock and SQLite's per-statement atomicity. Callers offload
to ``asyncio.to_thread`` if they care about not blocking the loop.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SessionBinding", "SessionBindingStore"]


@dataclass(frozen=True)
class SessionBinding:
    session_key: str
    host_id: str
    resume_id: str | None
    last_seen_at: float


class SessionBindingStore:
    """SQLite-backed persistent map. One table, columns:

    ``session_key`` (PK), ``host_id``, ``resume_id``, ``last_seen_at``.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS session_bindings (
        session_key   TEXT PRIMARY KEY,
        host_id       TEXT NOT NULL,
        resume_id     TEXT,
        last_seen_at  REAL NOT NULL
    )
    """

    def __init__(self, db_path: Path | str) -> None:
        # ``":memory:"`` is honored as a special sqlite path so tests
        # can spin up a store without touching the filesystem. The
        # in-memory connection is opened eagerly and held — the data
        # disappears when ``close()`` is called.
        if isinstance(db_path, str) and db_path == ":memory:":
            self._path: Path | str = ":memory:"
            self._in_memory = True
        else:
            p = Path(db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._path = p
            self._in_memory = False
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # -- lifecycle ---------------------------------------------------

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            path_arg: str = ":memory:" if self._in_memory else str(self._path)
            self._conn = sqlite3.connect(
                path_arg, isolation_level=None, check_same_thread=False
            )
            if not self._in_memory:
                # WAL only makes sense on disk-backed files.
                self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.execute("PRAGMA synchronous = NORMAL")
        return self._conn

    def _ensure_schema(self) -> None:
        with self._lock:
            self._c().execute(self._DDL)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- reads -------------------------------------------------------

    def get(self, session_key: str) -> SessionBinding | None:
        with self._lock:
            row = self._c().execute(
                "SELECT session_key, host_id, resume_id, last_seen_at "
                "FROM session_bindings WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        if row is None:
            return None
        return SessionBinding(
            session_key=row[0],
            host_id=row[1],
            resume_id=row[2],
            last_seen_at=float(row[3]),
        )

    def all_for_host(self, host_id: str) -> list[SessionBinding]:
        with self._lock:
            rows = self._c().execute(
                "SELECT session_key, host_id, resume_id, last_seen_at "
                "FROM session_bindings WHERE host_id = ?",
                (host_id,),
            ).fetchall()
        return [
            SessionBinding(
                session_key=r[0], host_id=r[1], resume_id=r[2], last_seen_at=float(r[3])
            )
            for r in rows
        ]

    # -- writes ------------------------------------------------------

    def upsert(
        self,
        session_key: str,
        host_id: str,
        resume_id: str | None = None,
    ) -> None:
        """Bind ``session_key`` to ``host_id``. If ``resume_id`` is
        ``None`` the existing one (if any) is preserved — letting the
        bridge record host changes without losing the resumption
        anchor that the worker reports separately."""
        now = time.time()
        with self._lock:
            c = self._c()
            if resume_id is None:
                c.execute(
                    "INSERT INTO session_bindings "
                    "(session_key, host_id, resume_id, last_seen_at) "
                    "VALUES (?, ?, NULL, ?) "
                    "ON CONFLICT(session_key) DO UPDATE SET "
                    "host_id = excluded.host_id, "
                    "last_seen_at = excluded.last_seen_at",
                    (session_key, host_id, now),
                )
            else:
                c.execute(
                    "INSERT INTO session_bindings "
                    "(session_key, host_id, resume_id, last_seen_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_key) DO UPDATE SET "
                    "host_id = excluded.host_id, "
                    "resume_id = excluded.resume_id, "
                    "last_seen_at = excluded.last_seen_at",
                    (session_key, host_id, resume_id, now),
                )

    def set_resume_id(self, session_key: str, resume_id: str) -> None:
        """Update only the resume_id (called when a host reports its
        AgentSession id after first inbound)."""
        with self._lock:
            self._c().execute(
                "UPDATE session_bindings SET resume_id = ?, last_seen_at = ? "
                "WHERE session_key = ?",
                (resume_id, time.time(), session_key),
            )

    def delete(self, session_key: str) -> None:
        with self._lock:
            self._c().execute(
                "DELETE FROM session_bindings WHERE session_key = ?", (session_key,)
            )
