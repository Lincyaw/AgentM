"""Smoke tests: package surface + typed-payload round-trip (V2).

V1 ``DriftType`` / ``Verdict(drift=...)`` / ``Reminder(type=...)`` shape
tests were removed in the V2 breaking change (issue #134, 2026-05-10).
This file pins the V2 public contract so downstream consumers do not
break silently.
"""

from __future__ import annotations

import pytest

from llmharness import Verdict
from llmharness.audit.auditor.output import AuditorOutputError, RawVerdictOutput


def test_verdict_round_trip_surface_reminder() -> None:
    """V2 Verdict with surface_reminder=True round-trips through to_dict/from_dict."""
    v = Verdict(
        surface_reminder=True,
        reminder_text="consider whether ev #5 is still alive",
        continuation_notes=["recheck whether the dropped branch closed"],
        matched_event_ids=[5, 8],
        cited_cards=["AFC-0012"],
    )
    restored = Verdict.from_dict(v.to_dict())
    assert restored == v


def test_verdict_round_trip_silent() -> None:
    """V2 silent verdict (surface_reminder=False) round-trips cleanly."""
    v = Verdict(
        surface_reminder=False,
        reminder_text="",
        continuation_notes=[],
        matched_event_ids=[],
        cited_cards=[],
    )
    restored = Verdict.from_dict(v.to_dict())
    assert restored == v


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


def test_raw_verdict_surface_reminder_true_whitespace_only_text() -> None:
    """surface_reminder=True with whitespace-only reminder_text raises AuditorOutputError."""
    with pytest.raises(AuditorOutputError, match="must be non-empty"):
        RawVerdictOutput.from_dict(
            {
                "verdict": {
                    "surface_reminder": True,
                    "reminder_text": "   \t\n  ",
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


def test_raw_verdict_continuation_notes_not_a_list() -> None:
    """continuation_notes as a non-list raises AuditorOutputError."""
    with pytest.raises(AuditorOutputError, match="continuation_notes"):
        RawVerdictOutput.from_dict(
            {
                "verdict": {
                    "surface_reminder": False,
                    "reminder_text": "",
                    "continuation_notes": "note",
                    "matched_event_ids": [],
                    "cited_cards": [],
                }
            }
        )


def test_raw_verdict_matched_event_ids_not_list_of_ints() -> None:
    """matched_event_ids containing non-ints raises AuditorOutputError."""
    with pytest.raises(AuditorOutputError, match="matched_event_ids"):
        RawVerdictOutput.from_dict(
            {
                "verdict": {
                    "surface_reminder": False,
                    "reminder_text": "",
                    "continuation_notes": [],
                    "matched_event_ids": ["not-an-int"],
                    "cited_cards": [],
                }
            }
        )


def test_raw_verdict_cited_cards_not_list_of_strings() -> None:
    """cited_cards containing non-strings raises AuditorOutputError."""
    with pytest.raises(AuditorOutputError, match="cited_cards"):
        RawVerdictOutput.from_dict(
            {
                "verdict": {
                    "surface_reminder": False,
                    "reminder_text": "",
                    "continuation_notes": [],
                    "matched_event_ids": [],
                    "cited_cards": [123],
                }
            }
        )


def test_raw_verdict_happy_path_silent() -> None:
    """A well-shaped silent verdict parses and round-trips to Verdict."""
    raw = RawVerdictOutput.from_dict(
        {
            "verdict": {
                "surface_reminder": False,
                "reminder_text": "",
                "continuation_notes": [],
                "matched_event_ids": [],
                "cited_cards": [],
            }
        }
    )
    v = raw.to_verdict()
    assert v.surface_reminder is False
    assert v.reminder_text == ""
    assert v.continuation_notes == []


def test_raw_verdict_happy_path_with_reminder() -> None:
    """A well-shaped verdict with surface_reminder=True parses correctly."""
    raw = RawVerdictOutput.from_dict(
        {
            "verdict": {
                "surface_reminder": True,
                "reminder_text": "the dropped branch may still be open",
                "continuation_notes": ["check event #3"],
                "matched_event_ids": [3, 7],
                "cited_cards": ["AFC-0010"],
            }
        }
    )
    v = raw.to_verdict()
    assert v.surface_reminder is True
    assert v.reminder_text == "the dropped branch may still be open"
    assert v.continuation_notes == ["check event #3"]
    assert v.matched_event_ids == [3, 7]
    assert v.cited_cards == ["AFC-0010"]


