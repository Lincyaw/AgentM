"""Memory tools for reading trajectory checkpoints.

Uses the ContextVar pattern (same as knowledge.py) for checkpointer injection.
Builder calls memory_module.set_checkpointer(checkpointer) at startup.
"""

from __future__ import annotations

import contextvars
import json
from typing import Any

# Module-level checkpointer reference, set by builder at startup
_checkpointer_var: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "memory_checkpointer", default=None
)


def set_checkpointer(checkpointer: Any) -> None:
    """Set the LangGraph checkpointer. Called by builder at startup."""
    _checkpointer_var.set(checkpointer)


def read_trajectory(thread_id: str) -> str:
    """Read all messages from a completed trajectory checkpoint.

    Loads the most recent checkpoint for the given thread_id and returns
    the full message history formatted as readable text.

    Args:
        thread_id: The thread ID of the trajectory to read.
    """
    checkpointer = _checkpointer_var.get()
    if checkpointer is None:
        return "Memory checkpointer not initialized — call set_checkpointer() first."

    config = {"configurable": {"thread_id": thread_id}}
    try:
        # get() returns a CheckpointTuple or None
        checkpoint_tuple = checkpointer.get(config)
    except Exception as exc:
        return f"Error reading trajectory {thread_id!r}: {exc}"

    if checkpoint_tuple is None:
        return f"No checkpoint found for thread_id={thread_id!r}."

    messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
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
            preview = content[:500] + ("..." if len(content) > 500 else "")
            lines.append(f"[{i}] {role.upper()}: {preview}")
        else:
            lines.append(f"[{i}] {role.upper()}: (empty)")

    return "\n".join(lines)


def get_checkpoint_history(thread_id: str, limit: int = 50) -> str:
    """List checkpoint history steps for a trajectory thread.

    Returns metadata about each checkpoint (step number, timestamp, channels
    updated) without loading full message content.

    Args:
        thread_id: The thread ID to inspect.
        limit: Maximum number of checkpoints to return (default 50).
    """
    checkpointer = _checkpointer_var.get()
    if checkpointer is None:
        return "Memory checkpointer not initialized — call set_checkpointer() first."

    config = {"configurable": {"thread_id": thread_id}}
    try:
        history = list(checkpointer.list(config, limit=limit))
    except Exception as exc:
        return f"Error listing checkpoint history for {thread_id!r}: {exc}"

    if not history:
        return f"No checkpoint history found for thread_id={thread_id!r}."

    entries: list[dict[str, Any]] = []
    for i, ct in enumerate(history):
        ts = getattr(ct.checkpoint, "ts", None) or getattr(ct, "created_at", "")
        channel_versions = ct.checkpoint.get("channel_versions", {})
        entries.append(
            {
                "step": i,
                "ts": str(ts),
                "channels_updated": list(channel_versions.keys()),
                "metadata": ct.metadata or {},
            }
        )

    return json.dumps(entries, default=str, indent=2)
