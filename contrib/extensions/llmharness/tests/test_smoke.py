"""Smoke tests: package surface + typed-payload round-trip.

V0 ``compose_extensions`` / ``RawAuditOutput`` / ``AUDIT_SYSTEM_PROMPT`` /
``SUBMIT_AUDIT_TOOL_NAME`` shape tests were deleted in the 2026-05-08 hard
cut to V1 (design ``llmharness-cognitive-audit.md`` §7.1, §11). The V1
fail-stop integration test is owned by task 07; this file deliberately
keeps only the public-contract assertions that V1 must preserve so
downstream ``rca-autorl`` does not break.
"""

from __future__ import annotations

from llmharness import DriftType, Reminder, Verdict


def test_package_surface() -> None:
    """Public symbols stay reachable from the top-level module.

    These are the schema names ``rca-autorl`` imports; V1 must preserve them.
    """

    import llmharness

    for name in ("DriftType", "Event", "EventKind", "Reminder", "Verdict"):
        assert hasattr(llmharness, name), f"missing public symbol: {name}"


def test_reminder_is_typed_payload() -> None:
    """Reminder survived the file-store removal as a typed in-memory payload."""

    r = Reminder(type=DriftType.TASK_DRIFT, confidence=0.9, text="refocus")
    assert r.text == "refocus"


def test_verdict_round_trip() -> None:
    """Verdict.to_dict / from_dict survives the session-tree migration —
    payloads on the entry tree must round-trip cleanly."""

    v = Verdict(
        drift=True,
        type=DriftType.STUCK_LOOP,
        confidence=0.7,
        reminder="break the loop",
        cited_cards=["AFC-0010"],
    )
    restored = Verdict.from_dict(v.to_dict())
    assert restored == v
