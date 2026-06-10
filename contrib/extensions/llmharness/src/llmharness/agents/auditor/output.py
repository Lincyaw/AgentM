"""Typed coercion of the ``submit_verdict`` tool-call arguments (V2).

The auditor terminates by calling ``submit_verdict(verdict=...)``. The
kernel records that call as a :class:`ToolCallBlock` whose ``arguments``
is a ``dict[str, Any]`` validated against
:class:`llmharness.audit.auditor.submit_verdict.SubmitVerdictArgs` (the
pydantic model that backs the tool's JSON schema).
This module gives the adapter a typed view over that dict — coercing it
to a :class:`llmharness.schema.Verdict` so downstream code never sees
``Any``.

Verdict shape (design §6.2): ``surface_reminder``, ``reminder_text``,
``continuation_notes``, ``matched_event_ids``.
No ``drift`` field, no ``DriftType`` enum, no ``downstream_reaction``.

A malformed payload (missing ``verdict`` object, missing / non-bool
``surface_reminder``, or ``surface_reminder=True`` with empty /
whitespace-only ``reminder_text``) raises :class:`AuditorOutputError`.
The adapter catches that and writes an ``audit_error`` entry, making the
failure visible in the entry stream rather than silently downgrading to a
silent verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmharness.schema import Verdict


class AuditorOutputError(Exception):
    """Raised when the auditor's ``submit_verdict`` payload is malformed.

    The adapter records this as an ``audit_error`` entry on the session
    entry tree so the failure is visible to operators rather than silently
    collapsed to a silent verdict.
    """


@dataclass(frozen=True)
class RawVerdictOutput:
    """Typed view over the ``{verdict: {...}}`` payload of submit_verdict (V2).

    Use :meth:`from_dict` to parse and :meth:`to_verdict` to materialize
    a :class:`Verdict` for the adapter. Both methods raise
    :class:`AuditorOutputError` on shape violations.
    """

    surface_reminder: bool
    reminder_text: str
    continuation_notes: list[str]
    matched_event_ids: list[int]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RawVerdictOutput:
        """Parse the tool-call ``arguments`` dict into a typed view.

        Raises :class:`AuditorOutputError` on:
        - missing or non-dict ``verdict`` key,
        - missing or non-bool ``surface_reminder`` field,
        - ``surface_reminder=True`` with empty or whitespace-only ``reminder_text``,
        - ``continuation_notes`` not a list of strings,
        - ``matched_event_ids`` not a list of ints.
        """
        verdict_raw = raw.get("verdict")
        if not isinstance(verdict_raw, dict):
            raise AuditorOutputError("submit_verdict payload missing object-typed 'verdict' field")

        if "surface_reminder" not in verdict_raw:
            raise AuditorOutputError(
                "submit_verdict.verdict missing required 'surface_reminder' field"
            )
        surface_reminder_raw = verdict_raw["surface_reminder"]
        if not isinstance(surface_reminder_raw, bool):
            raise AuditorOutputError(
                f"submit_verdict.verdict.surface_reminder must be bool, "
                f"got {type(surface_reminder_raw).__name__}"
            )

        reminder_text_raw = verdict_raw.get("reminder_text", "")
        if not isinstance(reminder_text_raw, str):
            raise AuditorOutputError("submit_verdict.verdict.reminder_text must be a string")
        if surface_reminder_raw and not reminder_text_raw.strip():
            raise AuditorOutputError(
                "submit_verdict.verdict.reminder_text must be non-empty when surface_reminder=true"
            )

        notes_raw = verdict_raw.get("continuation_notes", [])
        if not isinstance(notes_raw, list) or not all(isinstance(n, str) for n in notes_raw):
            raise AuditorOutputError(
                "submit_verdict.verdict.continuation_notes must be a list of strings"
            )

        matched_raw = verdict_raw.get("matched_event_ids", [])
        if not isinstance(matched_raw, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in matched_raw
        ):
            raise AuditorOutputError(
                "submit_verdict.verdict.matched_event_ids must be a list of integers"
            )

        return cls(
            surface_reminder=surface_reminder_raw,
            reminder_text=reminder_text_raw,
            continuation_notes=list(notes_raw),
            matched_event_ids=list(matched_raw),
        )

    def to_verdict(self) -> Verdict:
        """Materialize a :class:`Verdict` for the adapter."""
        return Verdict(
            surface_reminder=self.surface_reminder,
            reminder_text=self.reminder_text,
            continuation_notes=list(self.continuation_notes),
            matched_event_ids=list(self.matched_event_ids),
        )


__all__ = ["AuditorOutputError", "RawVerdictOutput"]
