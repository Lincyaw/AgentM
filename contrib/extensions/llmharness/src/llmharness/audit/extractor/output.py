"""Frozen view over the v3 extractor's per-firing ``ExtractionState``.

V2 parsed a JSON payload off the terminal tool-call's ``arguments`` dict.
V3 has no payload tool: the state IS the output. After the child loop
terminates (via ``submit_extraction``), the adapter calls
``RawExtractorOutput.from_state(state)`` to snapshot events, edges, and
the partial-payload buffer of dropped edges (design §4.f, §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...schema import Edge, Event
from .state import ExtractionState


@dataclass(frozen=True)
class RawExtractorOutput:
    """Frozen snapshot of one extractor firing.

    ``dropped_edges`` carries any edges the LLM attempted but failed to
    witness three times — the adapter persists them as the
    ``llmharness.extractor_partial`` payload.
    """

    events: tuple[Event, ...]
    edges: tuple[Edge, ...]
    dropped_edges: tuple[dict[str, Any], ...]

    @classmethod
    def from_state(cls, state: ExtractionState) -> RawExtractorOutput:
        events, edges, dropped = state.freeze()
        return cls(events=events, edges=edges, dropped_edges=tuple(dropped))

    @classmethod
    def from_dict(cls, raw: Any) -> RawExtractorOutput:
        """v2-shim. v3 builds output from ExtractionState, not a tool payload.

        Kept callable so the legacy adapter path imports cleanly until
        commit 3 rewrites the call site. Always raises — any real
        extractor firing under v3 must go through ``from_state``.
        """

        del raw
        raise NotImplementedError(
            "RawExtractorOutput.from_dict is a v2 path; v3 uses "
            "RawExtractorOutput.from_state(state). The adapter "
            "rewrite in commit 3 deletes this call site."
        )

    def to_events(self, *, next_id: int) -> list[Event]:
        """v2-shim. v3 events already carry ids assigned by ExtractionState.

        Returns ``list(self.events)`` ignoring ``next_id`` — kept only
        so the adapter import surface compiles until commit 3.
        """

        del next_id
        return list(self.events)


__all__ = ["RawExtractorOutput"]
