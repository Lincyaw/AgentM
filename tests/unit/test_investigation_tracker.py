"""Tests for InvestigationTracker — verifies query filtering and thread safety."""

from __future__ import annotations

import concurrent.futures
from dataclasses import FrozenInstanceError

import pytest

from agentm.scenarios.rca.sanitizer.models import (
    InvestigationEvent,
    SanitizerFinding,
    Severity,
)
from agentm.scenarios.rca.sanitizer.tracker import InvestigationTracker


@pytest.fixture
def tracker_with_events() -> InvestigationTracker:
    """Create a tracker pre-loaded with a realistic mix of events."""
    tracker = InvestigationTracker()
    tracker.record(1, "dispatch", {"agent": "scout-1", "target": "svc-a"})
    tracker.record(1, "tool_call", {"tool_name": "query_service_profile", "service": "svc-a"})
    tracker.record(1, "tool_call", {"tool_name": "query_logs", "service": "svc-a"})
    tracker.record(2, "task_complete", {"agent": "scout-1", "result": "anomaly found"})
    tracker.record(2, "hypothesis_change", {"id": "h1", "status": "formed"})
    tracker.record(3, "dispatch", {"agent": "deep-1", "target": "svc-b"})
    tracker.record(3, "tool_call", {"tool_name": "query_service_profile", "service": "svc-b"})
    tracker.record(4, "task_complete", {"agent": "deep-1", "result": "confirmed"})
    tracker.record(4, "hypothesis_change", {"id": "h1", "status": "confirmed"})
    tracker.record(5, "dispatch", {"agent": "verify-1", "target": "svc-b"})
    return tracker


class TestQueryFiltering:
    """Verify that each query method returns the correct subset of events."""

    def test_dispatches_returns_only_dispatch_events_in_order(
        self, tracker_with_events: InvestigationTracker
    ) -> None:
        dispatches = tracker_with_events.dispatches()
        assert len(dispatches) == 3
        assert all(e.event_type == "dispatch" for e in dispatches)
        assert [e.round for e in dispatches] == [1, 3, 5]

    def test_task_completions_returns_only_task_complete_events(
        self, tracker_with_events: InvestigationTracker
    ) -> None:
        completions = tracker_with_events.task_completions()
        assert len(completions) == 2
        assert all(e.event_type == "task_complete" for e in completions)
        assert [e.round for e in completions] == [2, 4]

    def test_hypothesis_changes_returns_only_hypothesis_change_events(
        self, tracker_with_events: InvestigationTracker
    ) -> None:
        changes = tracker_with_events.hypothesis_changes()
        assert len(changes) == 2
        assert all(e.event_type == "hypothesis_change" for e in changes)
        assert changes[0].data["status"] == "formed"
        assert changes[1].data["status"] == "confirmed"

    def test_tool_calls_for_filters_by_tool_name(
        self, tracker_with_events: InvestigationTracker
    ) -> None:
        profile_calls = tracker_with_events.tool_calls_for("query_service_profile")
        assert len(profile_calls) == 2
        assert all(
            e.data["tool_name"] == "query_service_profile" for e in profile_calls
        )

        log_calls = tracker_with_events.tool_calls_for("query_logs")
        assert len(log_calls) == 1

        missing_calls = tracker_with_events.tool_calls_for("nonexistent_tool")
        assert len(missing_calls) == 0

    def test_events_after_round_returns_later_events(
        self, tracker_with_events: InvestigationTracker
    ) -> None:
        after_3 = tracker_with_events.events_after(round=3)
        assert all(e.round > 3 for e in after_3)
        assert len(after_3) == 3  # rounds 4, 4, 5

    def test_events_after_round_with_type_filter(
        self, tracker_with_events: InvestigationTracker
    ) -> None:
        after_3_dispatches = tracker_with_events.events_after(round=3, event_type="dispatch")
        assert len(after_3_dispatches) == 1
        assert after_3_dispatches[0].round == 5
        assert after_3_dispatches[0].event_type == "dispatch"

    def test_all_events_returns_copy(
        self, tracker_with_events: InvestigationTracker
    ) -> None:
        all_events = tracker_with_events.all_events()
        assert len(all_events) == 10
        # Mutating the returned list does not affect the tracker
        all_events.clear()
        assert len(tracker_with_events.all_events()) == 10


class TestThreadSafety:
    """Verify concurrent writes produce no lost events."""

    def test_concurrent_recording_preserves_all_events(self) -> None:
        tracker = InvestigationTracker()
        num_threads = 10
        events_per_thread = 100

        def record_batch(thread_id: int) -> None:
            for i in range(events_per_thread):
                tracker.record(
                    round=thread_id,
                    event_type="dispatch",
                    data={"thread": thread_id, "seq": i},
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(record_batch, t) for t in range(num_threads)]
            concurrent.futures.wait(futures)
            # Re-raise any exceptions from threads
            for f in futures:
                f.result()

        all_events = tracker.all_events()
        assert len(all_events) == num_threads * events_per_thread


class TestSanitizerFindingFrozen:
    """Verify SanitizerFinding is immutable."""

    def test_cannot_modify_finding_fields(self) -> None:
        finding = SanitizerFinding(
            code="TEST001",
            severity=Severity.BLOCK,
            message="test finding",
            details={"key": "value"},
        )
        with pytest.raises(FrozenInstanceError):
            finding.code = "CHANGED"  # type: ignore[misc]

    def test_cannot_modify_investigation_event_fields(self) -> None:
        event = InvestigationEvent(round=1, event_type="dispatch", data={"a": 1})
        with pytest.raises(FrozenInstanceError):
            event.round = 99  # type: ignore[misc]
