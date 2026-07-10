"""Fail-stop test for sub-agent attempt fencing (reliability-substrate §4.1 C1).

User-authorized invariant (2026-07-10): a superseded task's late result is
never delivered as current. Delivery payloads are produced by
``_format_subagent_result`` at finalize time, so the invariant's load-bearing
half is: a state with ``superseded_by`` set always carries the stale marker,
and one without it never does.
"""

from __future__ import annotations

import asyncio

from agentm.extensions.builtin.sub_agent import _ChildTask, _format_subagent_result


def _state(**overrides: object) -> _ChildTask:
    defaults: dict[str, object] = {
        "task_id": "task-old-1",
        "purpose": "research",
        "session": None,
        "task": None,
        "abort_signal": asyncio.Event(),
        "summary": "the late finding",
    }
    defaults.update(overrides)
    return _ChildTask(**defaults)  # type: ignore[arg-type]


def test_superseded_result_always_carries_the_stale_marker() -> None:
    block = _format_subagent_result(_state(superseded_by="task-new-2"))
    assert 'stale="true"' in block
    assert 'superseded_by=task-new-2' in block
    # The parent-facing warning names the replacing task explicitly.
    assert "task task-new-2" in block
    assert "stale" in block.lower()
    # The payload itself is still delivered (flagged, not dropped).
    assert "the late finding" in block


def test_current_result_never_carries_the_stale_marker() -> None:
    block = _format_subagent_result(_state())
    assert "stale" not in block
    assert "superseded_by" not in block
    assert "the late finding" in block
