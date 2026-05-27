"""Frozen view over the v3 extractor's per-firing ``ExtractionState``.

V3 / V3.1 store committed events, edges, and dropped refs directly on
the ``ExtractionState`` instance. The adapter takes a snapshot via
:meth:`RawExtractorOutput.from_state` after the child loop terminates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...schema import Edge, Event
from .state import ExtractionState


@dataclass(frozen=True)
class RawExtractorOutput:
    """Frozen snapshot of one extractor firing.

    ``dropped_edges`` carries refs the LLM submitted but failed witness
    validation — the adapter persists them as the
    ``llmharness.extractor_partial`` payload.
    """

    events: tuple[Event, ...]
    edges: tuple[Edge, ...]
    dropped_edges: tuple[dict[str, Any], ...]

    @classmethod
    def from_state(cls, state: ExtractionState) -> RawExtractorOutput:
        return cls(
            events=tuple(state.events),
            edges=tuple(state.edges),
            dropped_edges=tuple(state.dropped_edges),
        )

    @classmethod
    def salvage(cls, state: ExtractionState) -> RawExtractorOutput:
        """Commit-on-stop chokepoint: finalize if needed, then snapshot.

        The single owner of commit-on-stop for every extractor path —
        live :meth:`HarnessRunner.fire_extractor_once`, offline
        full-trajectory replay (same method via the offline seam), and
        single-firing replay (:func:`replay_extractor_record`). The child
        loop terminates on ModelEndTurn or the tool-call budget regardless
        of whether the optional ``finalize_extraction`` terminator fired;
        every ``apply_node_upsert`` / ``apply_edge_upsert`` already
        appended its op to ``state.pending_ops``, so freezing the state
        commits from whatever ops were applied. :meth:`ExtractionState.finalize`
        is idempotent (no-op when the terminator already finalized) and
        reads the op log back into ``events`` / ``edges`` even when the
        terminator was never called. Reading ops back here keeps the
        design's live ≡ offline invariant intact.
        """
        if not state.committed:
            state.finalize()
        return cls.from_state(state)


__all__ = ["RawExtractorOutput"]
