"""InvestigationTracker — thread-safe event log for RCA investigations.

Records and queries investigation events (dispatches, tool calls,
hypothesis changes, task completions) to support sanitizer checks.
"""

from __future__ import annotations

import threading
from typing import Any

from agentm_rca.sanitizer.models import InvestigationEvent


class InvestigationTracker:
    """Thread-safe, append-only event log for a single investigation run.

    Uses the same lock-based pattern as ``ThreadSafeStore`` but stores
    an ordered list rather than a keyed dictionary.
    """

    def __init__(self) -> None:
        self._events: list[InvestigationEvent] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, round: int, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Append a new event to the log."""
        event = InvestigationEvent(round=round, event_type=event_type, data=data or {})
        with self._lock:
            self._events.append(event)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def dispatches(self) -> list[InvestigationEvent]:
        """Return all dispatch events in insertion order."""
        with self._lock:
            return [e for e in self._events if e.event_type == "dispatch"]

    def task_completions(self) -> list[InvestigationEvent]:
        """Return all task_complete events in insertion order."""
        with self._lock:
            return [e for e in self._events if e.event_type == "task_complete"]

    def hypothesis_changes(self) -> list[InvestigationEvent]:
        """Return all hypothesis_change events in insertion order."""
        with self._lock:
            return [e for e in self._events if e.event_type == "hypothesis_change"]

    def tool_calls_for(self, tool_name: str) -> list[InvestigationEvent]:
        """Return tool_call events matching *tool_name*."""
        with self._lock:
            return [
                e
                for e in self._events
                if e.event_type == "tool_call" and e.data.get("tool_name") == tool_name
            ]

    def events_after(
        self, round: int, event_type: str | None = None
    ) -> list[InvestigationEvent]:
        """Return events with round strictly greater than *round*.

        Optionally filter by *event_type*.
        """
        with self._lock:
            results = [e for e in self._events if e.round > round]
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        return results

    def all_events(self) -> list[InvestigationEvent]:
        """Return a copy of all recorded events."""
        with self._lock:
            return list(self._events)
