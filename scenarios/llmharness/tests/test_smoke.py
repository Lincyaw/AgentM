"""Smoke tests: package surface + audit composition + JSON parser shape."""

from __future__ import annotations

from llmharness import DriftType, EventKind, Reminder, Verdict
from llmharness.audit import (
    AUDIT_SYSTEM_PROMPT,
    RawAuditOutput,
    compose_extensions,
    extract_json,
)


def test_package_surface() -> None:
    """Public symbols stay reachable from the top-level module."""

    import llmharness

    for name in ("DriftType", "Event", "EventKind", "Reminder", "Verdict"):
        assert hasattr(llmharness, name), f"missing public symbol: {name}"


def test_compose_extensions_default_shape() -> None:
    """The default audit child uses observability + cards_tools + system_prompt
    in declaration order, with the canonical AUDIT_SYSTEM_PROMPT injected."""

    exts = compose_extensions()
    modules = [m for m, _ in exts]
    assert modules == [
        "agentm.extensions.builtin.observability",
        "llmharness.atoms.cards_tools",
        "agentm.extensions.builtin.system_prompt",
    ]
    sys_prompt_cfg = exts[2][1]
    assert sys_prompt_cfg["prompt"] is AUDIT_SYSTEM_PROMPT


def test_compose_extensions_drops_optional_when_none() -> None:
    """Passing ``None`` for cards_tools_config / observability_config drops
    those entries entirely — system_prompt always survives."""

    exts = compose_extensions(cards_tools_config=None, observability_config=None)
    modules = [m for m, _ in exts]
    assert modules == ["agentm.extensions.builtin.system_prompt"]


def test_audit_output_round_trip() -> None:
    """The audit child's emitted JSON parses into typed Event + Verdict."""

    payload = {
        "events": [
            {"id": 0, "kind": "task", "summary": "find the bug", "source_turns": [0]},
            {"id": 1, "kind": "decision", "summary": "search logs", "source_turns": [1]},
        ],
        "verdict": {
            "drift": True,
            "type": "task_drift",
            "confidence": 0.8,
            "reminder": "you are off-task",
            "matched_event_ids": [1],
            "cited_cards": ["AFC-0001"],
        },
    }
    parsed = RawAuditOutput.from_dict(payload)
    assert parsed is not None

    events = parsed.to_events(next_id=10)
    assert [e.id for e in events] == [10, 11]
    assert events[0].kind is EventKind.TASK

    verdict = parsed.to_verdict()
    assert verdict.drift is True
    assert verdict.type is DriftType.TASK_DRIFT
    assert verdict.cited_cards == ["AFC-0001"]


def test_extract_json_prefers_fenced_block() -> None:
    """When the assistant emits prose then a fenced ```json block, the
    extractor picks the fenced block."""

    text = 'thinking out loud...\n```json\n{"drift": false}\n```\n'
    data = extract_json(text)
    assert data == {"drift": False}


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
