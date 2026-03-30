"""Memory tools for reading trajectory checkpoints.

Reads directly from the SQLite checkpoints.db using langgraph's serde,
so workers can call these sync tools without needing the async checkpointer.
Builder calls memory_module.set_db_path(path) at startup.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentm.utils.db import sqlite_cursor
from agentm.utils.serde import deserialize_typed

logger = logging.getLogger(__name__)


class MemoryStore:
    """Read-only access to trajectory checkpoints via SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def read_trajectory(self, thread_id: str) -> str:
        """Read all messages from the most recent checkpoint of a trajectory.

        Args:
            thread_id: The thread ID of the trajectory to read.
        """
        try:
            with sqlite_cursor(self._db_path) as cur:
                row = cur.execute(
                    "SELECT type, checkpoint FROM checkpoints "
                    "WHERE thread_id=? AND checkpoint_ns='' "
                    "ORDER BY checkpoint_id DESC LIMIT 1",
                    (thread_id,),
                ).fetchone()
        except Exception as exc:
            return f"Error reading trajectory {thread_id!r}: {exc}"

        if row is None:
            return f"No checkpoint found for thread_id={thread_id!r}."

        try:
            obj = deserialize_typed((row[0], row[1]))
        except Exception as exc:
            return f"Error deserializing checkpoint for {thread_id!r}: {exc}"

        messages = obj.get("channel_values", {}).get("messages", [])
        if not messages:
            return f"Checkpoint found for {thread_id!r} but contains no messages."

        lines: list[str] = [f"# Trajectory: {thread_id}", ""]
        for i, msg in enumerate(messages, 1):
            role = getattr(msg, "type", "unknown")
            content = str(getattr(msg, "content", ""))
            tool_calls = getattr(msg, "tool_calls", None)

            if tool_calls:
                tc_str = ", ".join(
                    f"{tc.get('name')}({json.dumps(tc.get('args', {}), default=str)[:100]})"
                    for tc in tool_calls
                )
                lines.append(f"[{i}] {role.upper()}: [tool_calls: {tc_str}]")
            elif content:
                preview = content
                lines.append(f"[{i}] {role.upper()}: {preview}")
            else:
                lines.append(f"[{i}] {role.upper()}: (empty)")

        return "\n".join(lines)

    def get_checkpoint_history(self, thread_id: str, limit: int = 50) -> str:
        """List checkpoint history steps for a trajectory thread.

        Returns metadata (step index, checkpoint_id, channels updated) without
        loading full message content.

        Args:
            thread_id: The thread ID to inspect.
            limit: Maximum number of checkpoints to return (default 50).
        """
        try:
            with sqlite_cursor(self._db_path) as cur:
                rows = cur.execute(
                    "SELECT checkpoint_id, type, checkpoint FROM checkpoints "
                    "WHERE thread_id=? AND checkpoint_ns='' "
                    "ORDER BY checkpoint_id DESC LIMIT ?",
                    (thread_id, limit),
                ).fetchall()
        except Exception as exc:
            return f"Error listing checkpoint history for {thread_id!r}: {exc}"

        if not rows:
            return f"No checkpoint history found for thread_id={thread_id!r}."

        entries: list[dict[str, Any]] = []
        for i, (checkpoint_id, type_, data) in enumerate(rows):
            try:
                obj = deserialize_typed((type_, data))
                ts = obj.get("ts", "")
                channels = list(obj.get("channel_versions", {}).keys())
            except Exception as exc:
                logger.debug("Failed to parse checkpoint %s: %s", checkpoint_id, exc)
                ts = ""
                channels = []
            entries.append(
                {
                    "step": i,
                    "checkpoint_id": checkpoint_id,
                    "ts": ts,
                    "channels_updated": channels,
                }
            )

        return json.dumps(entries, default=str, indent=2)


# ---------------------------------------------------------------------------
# Module-level backward-compatible API
# ---------------------------------------------------------------------------

_default_store: MemoryStore | None = None


def set_db_path(path: str) -> None:
    """Set the SQLite DB path. Called by builder at startup."""
    global _default_store
    _default_store = MemoryStore(path)


def read_trajectory(thread_id: str) -> str:
    """Read all messages from the most recent checkpoint of a trajectory.

    Args:
        thread_id: The thread ID of the trajectory to read.
    """
    if _default_store is None:
        return "Memory DB path not set — call set_db_path() first."
    return _default_store.read_trajectory(thread_id)


def get_checkpoint_history(thread_id: str, limit: int = 50) -> str:
    """List checkpoint history steps for a trajectory thread.

    Returns metadata (step index, checkpoint_id, channels updated) without
    loading full message content.

    Args:
        thread_id: The thread ID to inspect.
        limit: Maximum number of checkpoints to return (default 50).
    """
    if _default_store is None:
        return "Memory DB path not set — call set_db_path() first."
    return _default_store.get_checkpoint_history(thread_id, limit)
