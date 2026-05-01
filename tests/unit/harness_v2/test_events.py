"""Phase 2.0 contract tests for ``harness/events.py``.

We're not testing dataclass plumbing in general (that's Python). We're
testing the **contract** documented in §10b.1:

- Compaction events form a before/after pair the SDK guarantees.
- ``BeforeCompactEvent.messages`` must be **mutable** (so a handler can
  rewrite the buffer before compaction kicks off).
- ``AfterCompactEvent`` must be **immutable** (it is a finalized record).
- Child-session events must be immutable (event records are append-only
  history; mutating one would corrupt parent-side rollups).
"""

from __future__ import annotations

import dataclasses
from typing import cast

import pytest

from agentm.core.kernel import AgentMessage
from agentm.core.kernel import text_message
from agentm.harness.events import (
    AfterCompactEvent,
    BeforeCompactEvent,
    ChildSessionEndEvent,
    ChildSessionStartEvent,
)


def test_before_compact_messages_is_mutable() -> None:
    """Handlers must be able to swap entries in the messages buffer in place
    before compaction runs. Same contract as ``ContextEvent``."""

    msgs = cast(
        list[AgentMessage],
        [text_message("a", timestamp=0.0), text_message("b", timestamp=1.0)],
    )
    event = BeforeCompactEvent(messages=msgs, reason="auto_overflow")

    # Mutation must succeed (event is not frozen).
    event.messages.append(text_message("c", timestamp=2.0))
    event.reason = "manual"  # field reassignment also allowed

    assert len(event.messages) == 3
    assert event.reason == "manual"


def test_after_compact_is_frozen() -> None:
    """``AfterCompactEvent`` is an immutable receipt — handlers must not be
    able to retroactively edit a finalized compaction record."""

    event = AfterCompactEvent(
        summary="trimmed", kept_message_count=3, discarded_message_count=10
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.summary = "tampered"  # type: ignore[misc]


def test_child_session_events_are_frozen() -> None:
    """Child-session lifecycle events are append-only history."""

    start = ChildSessionStartEvent(
        child_session_id="abc",
        parent_session_id="root",
        purpose="subagent:test",
    )
    end = ChildSessionEndEvent(
        child_session_id="abc",
        parent_session_id="root",
        final_message_count=4,
        error=None,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        start.purpose = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        end.final_message_count = 99  # type: ignore[misc]
