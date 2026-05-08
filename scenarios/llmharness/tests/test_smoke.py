"""Smoke tests: package imports + rule-based worker round-trips end to end."""

from __future__ import annotations

from pathlib import Path

from llmharness.schema import Turn, TurnRole
from llmharness.store import HarnessStore
from llmharness.worker import tick


def test_package_surface() -> None:
    import llmharness

    for name in (
        "Event",
        "EventKind",
        "HarnessStore",
        "Turn",
        "TurnRole",
        "Verdict",
        "summarize_turns",
        "detect_drift",
        "tick",
    ):
        assert hasattr(llmharness, name), f"missing public symbol: {name}"


def test_rule_provider_tick(tmp_path: Path) -> None:
    """Default LLMHARNESS_PROVIDER=rule must run without any LLM call."""

    store = HarnessStore(str(tmp_path))
    sid = "smoke"
    store.append_inbox(
        sid,
        [
            Turn(index=0, role=TurnRole.USER, content="locate latency in service-B"),
            Turn(
                index=1,
                role=TurnRole.ASSISTANT,
                content="",
                tool_name="Read",
                tool_args={"path": "logs"},
            ),
        ],
    )
    result = tick(store, sid)
    assert result.last_turn_index == 1
    assert result.new_event_count >= 1
