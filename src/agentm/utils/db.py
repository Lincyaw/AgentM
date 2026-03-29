"""Shared SQLite database utilities.

Provides a context manager for safe connection handling with automatic
transaction management, error handling, and resource cleanup.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def sqlite_cursor(
    db_path: str | Path,
    *,
    commit: bool = False,
    row_factory: type | None = None,
    check_same_thread: bool = True,
    timeout: float = 5.0,
) -> Generator[sqlite3.Cursor, None, None]:
    """Context manager for SQLite cursor with automatic connection handling.

    Args:
        db_path: Path to the SQLite database file.
        commit: If True, commits transaction on success; rolls back on error.
        row_factory: Optional row factory (e.g., sqlite3.Row) for results.
        check_same_thread: Passed to sqlite3.connect. Set False for multi-threading.
        timeout: Connection timeout in seconds.

    Yields:
        sqlite3.Cursor: Configured cursor ready for execution.

    Example:
        # Read-only query
        with sqlite_cursor("data.db") as cur:
            rows = cur.execute("SELECT * FROM items").fetchall()

        # Transaction with commit
        with sqlite_cursor("data.db", commit=True) as cur:
            cur.execute("INSERT INTO items VALUES (?)", (value,))
    """
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=check_same_thread,
        timeout=timeout,
    )
    if row_factory is not None:
        conn.row_factory = row_factory

    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    except Exception:
        if commit:
            conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


@contextmanager
def sqlite_connection(
    db_path: str | Path,
    *,
    commit: bool = False,
    row_factory: type | None = None,
    check_same_thread: bool = True,
    timeout: float = 5.0,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connection with automatic cleanup.

    Use this when you need direct connection access (e.g., for passing
    connection to other functions that manage their own cursors).

    Args:
        db_path: Path to the SQLite database file.
        commit: If True, commits transaction on success; rolls back on error.
        row_factory: Optional row factory (e.g., sqlite3.Row) for results.
        check_same_thread: Passed to sqlite3.connect. Set False for multi-threading.
        timeout: Connection timeout in seconds.

    Yields:
        sqlite3.Connection: Configured connection ready for use.

    Example:
        # Pass connection to helper functions
        with sqlite_connection("data.db", commit=True) as conn:
            result = some_helper_function(conn)
    """
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=check_same_thread,
        timeout=timeout,
    )
    if row_factory is not None:
        conn.row_factory = row_factory

    try:
        yield conn
        if commit:
            conn.commit()
    except Exception:
        if commit:
            conn.rollback()
        raise
    finally:
        conn.close()
