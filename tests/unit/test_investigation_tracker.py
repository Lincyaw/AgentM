"""Focused tests for InvestigationTracker contracts."""

from __future__ import annotations

import concurrent.futures
from dataclasses import FrozenInstanceError

import pytest

from agentm.scenarios.rca.sanitizer.models import InvestigationEvent, SanitizerFinding, Severity
from agentm.scenarios.rca.sanitizer.tracker import InvestigationTracker


@pytest.fixture
def tracker_with_events() -> InvestigationTracker:
    tracker = InvestigationTracker()
    tracker.record(1, "dispatch", {"agent": "scout-1", "target": "svc-a"})
    tracker.record(1, "tool_call", {"tool_name": "query_service_profile", "service": "svc-a"})
    tracker.record(2, "task_complete", {"agent": "scout-1", "result": "anomaly found"})
    tracker.record(2, "hypothesis_change", {"id": "h1", "status": "formed"})
    tracker.record(3, "dispatch", {"agent": "deep-1", "target": "svc-b"})
    return tracker


def test_query_helpers_and_copy_semantics(tracker_with_events: InvestigationTracker) -> None:
    assert all(e.event_type == "dispatch" for e in tracker_with_events.dispatches())
    assert all(e.event_type == "task_complete" for e in tracker_with_events.task_completions())
    assert all(e.event_type == "hypothesis_change" for e in tracker_with_events.hypothesis_changes())
    assert len(tracker_with_events.tool_calls_for("query_service_profile")) == 1

    later = tracker_with_events.events_after(round=1)
    assert all(event.round > 1 for event in later)

    copied = tracker_with_events.all_events()
    copied.clear()
    assert len(tracker_with_events.all_events()) > 0


def test_concurrent_recording_preserves_all_events() -> None:
    tracker = InvestigationTracker()
    threads = 8
    per_thread = 50

    def record_batch(thread_id: int) -> None:
        for seq in range(per_thread):
            tracker.record(round=thread_id, event_type="dispatch", data={"thread": thread_id, "seq": seq})

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(record_batch, i) for i in range(threads)]
        for future in futures:
            future.result()

    assert len(tracker.all_events()) == threads * per_thread


def test_tracker_models_are_frozen() -> None:
    finding = SanitizerFinding(code="TEST", severity=Severity.BLOCK, message="m", details={})
    with pytest.raises(FrozenInstanceError):
        finding.code = "X"  # type: ignore[misc]

    event = InvestigationEvent(round=1, event_type="dispatch", data={"a": 1})
    with pytest.raises(FrozenInstanceError):
        event.round = 2  # type: ignore[misc]
