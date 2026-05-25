"""Smoke tests: package surface + typed-payload round-trip (V2).

V1 ``DriftType`` / ``Verdict(drift=...)`` / ``Reminder(type=...)`` shape
tests were removed in the V2 breaking change (issue #134, 2026-05-10).
This file pins the V2 public contract so downstream consumers do not
break silently.
"""

from __future__ import annotations

import pytest

from llmharness.audit.auditor.output import AuditorOutputError, RawVerdictOutput






# --- RawVerdictOutput negative cases ---


def test_raw_verdict_missing_verdict_key() -> None:
    """Missing 'verdict' key raises AuditorOutputError."""
    with pytest.raises(AuditorOutputError, match="missing object-typed 'verdict'"):
        RawVerdictOutput.from_dict({})


def test_raw_verdict_surface_reminder_not_bool() -> None:
    """Non-bool surface_reminder raises AuditorOutputError."""
    with pytest.raises(AuditorOutputError, match="must be bool"):
        RawVerdictOutput.from_dict(
            {
                "verdict": {
                    "surface_reminder": "yes",
                    "reminder_text": "hello",
                    "continuation_notes": [],
                    "matched_event_ids": [],
                    "cited_cards": [],
                }
            }
        )


def test_raw_verdict_surface_reminder_true_empty_text() -> None:
    """surface_reminder=True with empty reminder_text raises AuditorOutputError."""
    with pytest.raises(AuditorOutputError, match="must be non-empty"):
        RawVerdictOutput.from_dict(
            {
                "verdict": {
                    "surface_reminder": True,
                    "reminder_text": "",
                    "continuation_notes": [],
                    "matched_event_ids": [],
                    "cited_cards": [],
                }
            }
        )




def test_raw_verdict_continuation_notes_not_list_of_strings() -> None:
    """continuation_notes containing non-strings raises AuditorOutputError."""
    with pytest.raises(AuditorOutputError, match="continuation_notes"):
        RawVerdictOutput.from_dict(
            {
                "verdict": {
                    "surface_reminder": False,
                    "reminder_text": "",
                    "continuation_notes": [1, 2, 3],
                    "matched_event_ids": [],
                    "cited_cards": [],
                }
            }
        )












