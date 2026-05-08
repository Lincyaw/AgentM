"""Smoke tests: package surface + audit composition + JSON parser shape."""

from __future__ import annotations

from llmharness import DriftType, EventKind, Reminder, Verdict
from llmharness.audit import (
    AUDIT_SYSTEM_PROMPT,
    RawAuditOutput,
    compose_extensions,
)
from llmharness.audit.submit_tool import (
    SUBMIT_AUDIT_PARAMETERS,
    SUBMIT_AUDIT_TOOL_NAME,
)


def test_package_surface() -> None:
    """Public symbols stay reachable from the top-level module."""

    import llmharness

    for name in ("DriftType", "Event", "EventKind", "Reminder", "Verdict"):
        assert hasattr(llmharness, name), f"missing public symbol: {name}"


def test_compose_extensions_default_shape() -> None:
    """The default audit child loads observability + cards_tools + submit_tool
    + system_prompt in declaration order, with AUDIT_SYSTEM_PROMPT injected.
    submit_tool is mandatory — the audit terminates by calling submit_audit."""

    exts = compose_extensions()
    modules = [m for m, _ in exts]
    assert modules == [
        "agentm.extensions.builtin.observability",
        "llmharness.atoms.cards_tools",
        "llmharness.audit.submit_tool",
        "agentm.extensions.builtin.system_prompt",
    ]
    sys_prompt_cfg = exts[-1][1]
    assert sys_prompt_cfg["prompt"] is AUDIT_SYSTEM_PROMPT


def test_compose_extensions_keeps_submit_tool_when_optional_dropped() -> None:
    """Passing ``None`` drops cards_tools / observability but submit_tool
    and system_prompt always survive — without submit_audit the audit loop
    has no termination signal and no structured output channel."""

    exts = compose_extensions(cards_tools_config=None, observability_config=None)
    modules = [m for m, _ in exts]
    assert modules == [
        "llmharness.audit.submit_tool",
        "agentm.extensions.builtin.system_prompt",
    ]


def test_submit_audit_schema_matches_verdict_payload() -> None:
    """The submit_audit tool's JSON Schema must accept the exact payload
    shape RawAuditOutput.from_dict expects, otherwise the LLM provider's
    schema validation rejects valid audit submissions."""

    assert SUBMIT_AUDIT_TOOL_NAME == "submit_audit"
    assert SUBMIT_AUDIT_PARAMETERS["required"] == ["events", "verdict"]
    verdict_props = SUBMIT_AUDIT_PARAMETERS["properties"]["verdict"]["properties"]
    assert verdict_props["drift"]["type"] == "boolean"
    # ``type`` field uses an enum that includes None — the LLM may legitimately
    # set null when drift=false.
    assert None in verdict_props["type"]["enum"]
    event_kinds = SUBMIT_AUDIT_PARAMETERS["properties"]["events"]["items"][
        "properties"
    ]["kind"]["enum"]
    assert set(event_kinds) == {k.value for k in EventKind}


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
