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


__all__ = ["RawExtractorOutput"]
