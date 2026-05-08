"""Typed coercion of the ``submit_events`` tool-call arguments.

The extractor child terminates by calling ``submit_events(events=[...])``.
The kernel records that call as a :class:`ToolCallBlock` whose
``arguments`` is a ``dict[str, Any]`` already validated against the JSON
Schema declared in :mod:`llmharness.audit.extractor.submit_tool`. This
module gives the adapter a typed view over that dict, coerced into a
``list[Event]`` with monotonic ids.

Unlike V0's best-effort silent coercion, the extractor surface raises
:class:`ExtractorOutputError` on malformed input so the adapter can
classify the firing as ``extractor_error`` rather than letting bad shapes
slip into the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...schema import Event, EventKind


class ExtractorOutputError(ValueError):
    """Raised when the ``submit_events`` payload is malformed.

    The adapter catches this and writes an ``extractor_error`` entry so
    the failure is visible in the session entry stream rather than
    silently dropped.
    """


def _coerce_int_list(raw: Any, *, field_name: str) -> list[int]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ExtractorOutputError(
            f"submit_events: field {field_name!r} must be array, got {type(raw).__name__}"
        )
    out: list[int] = []
    for item in raw:
        # Reject bools (int subclasses) and any non-int entries.
        if isinstance(item, bool) or not isinstance(item, int):
            raise ExtractorOutputError(
                f"submit_events: field {field_name!r} contains non-integer entry "
                f"{item!r}"
            )
        out.append(item)
    return out


@dataclass(frozen=True)
class _RawExtractorEvent:
    """One event entry, parsed from the ``events`` array, pre-id-stamping."""

    kind: EventKind
    summary: str
    source_turns: list[int] = field(default_factory=list)
    refs: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Any, *, index: int) -> _RawExtractorEvent:
        if not isinstance(raw, dict):
            raise ExtractorOutputError(
                f"submit_events: events[{index}] must be object, got "
                f"{type(raw).__name__}"
            )
        kind_str = raw.get("kind")
        if not isinstance(kind_str, str):
            raise ExtractorOutputError(
                f"submit_events: events[{index}].kind must be string"
            )
        try:
            kind = EventKind(kind_str)
        except ValueError as exc:
            raise ExtractorOutputError(
                f"submit_events: events[{index}].kind {kind_str!r} not in EventKind"
            ) from exc
        summary = raw.get("summary")
        if not isinstance(summary, str):
            raise ExtractorOutputError(
                f"submit_events: events[{index}].summary must be string"
            )
        return cls(
            kind=kind,
            summary=summary,
            source_turns=_coerce_int_list(
                raw.get("source_turns"), field_name=f"events[{index}].source_turns"
            ),
            refs=_coerce_int_list(raw.get("refs"), field_name=f"events[{index}].refs"),
        )


@dataclass(frozen=True)
class RawExtractorOutput:
    """The parsed ``{events: [...]}`` payload from a successful submit."""

    events: list[_RawExtractorEvent] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Any) -> RawExtractorOutput:
        """Parse and validate a ``submit_events`` arguments dict.

        Raises :class:`ExtractorOutputError` on any structural problem.
        An empty ``events`` array is NOT an error here — the schema
        permits it; the adapter classifies empty windows.
        """

        if not isinstance(raw, dict):
            raise ExtractorOutputError(
                f"submit_events arguments must be object, got {type(raw).__name__}"
            )
        events_raw = raw.get("events")
        if events_raw is None:
            raise ExtractorOutputError("submit_events: missing required 'events' field")
        if not isinstance(events_raw, list):
            raise ExtractorOutputError(
                f"submit_events: 'events' must be array, got {type(events_raw).__name__}"
            )
        parsed = [
            _RawExtractorEvent.from_dict(item, index=i)
            for i, item in enumerate(events_raw)
        ]
        return cls(events=parsed)

    def to_events(self, *, next_id: int) -> list[Event]:
        """Stamp monotonic ids onto the parsed events starting at ``next_id``.

        The LLM never emits ``id`` (the schema forbids it); the adapter
        owns id sequencing. Callers seed ``next_id`` from their own state
        — typically ``max(prior_events.id, default=-1) + 1``.
        """

        out: list[Event] = []
        for raw in self.events:
            out.append(
                Event(
                    id=next_id,
                    kind=raw.kind,
                    summary=raw.summary,
                    refs=list(raw.refs),
                    source_turns=list(raw.source_turns),
                )
            )
            next_id += 1
        return out


__all__ = ["ExtractorOutputError", "RawExtractorOutput"]
