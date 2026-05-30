"""Fail-stop tests for the pluggable trigger registry.

Pinned positions: OR-semantics (any trigger fires -> auditor runs),
extractor_due only when at least one firing trigger requires it,
idempotent registration, and compound trigger evaluation.
"""

from __future__ import annotations

from llmharness.audit.triggers import (
    TriggerContext,
    TriggerDecision,
    TriggerRegistry,
)


def _ctx(
    turn_count: int = 1,
    terminal_tool_called: str | None = None,
) -> TriggerContext:
    return TriggerContext(
        turn_count=turn_count,
        messages=(),
        latest_assistant_message=None,
        terminal_tool_called=terminal_tool_called,
    )


class _CadenceTrigger:
    """Fires every N turns."""

    name: str = "cadence"

    def __init__(self, interval: int = 5) -> None:
        self._interval = interval

    def should_fire(self, ctx: TriggerContext) -> TriggerDecision:
        if ctx.turn_count % self._interval == 0:
            return TriggerDecision(fire=True, reason="cadence hit")
        return TriggerDecision(fire=False)


class _SubmissionTrigger:
    """Fires when a terminal tool is called."""

    name: str = "on_submission"

    def __init__(self, tool_names: frozenset[str] = frozenset({"submit_final_report"})) -> None:
        self._tool_names = tool_names

    def should_fire(self, ctx: TriggerContext) -> TriggerDecision:
        if ctx.terminal_tool_called and ctx.terminal_tool_called in self._tool_names:
            return TriggerDecision(
                fire=True,
                reason=f"tool {ctx.terminal_tool_called}",
            )
        return TriggerDecision(fire=False)


class _ExtractorFreeTrigger:
    """Fires but does not require the extractor."""

    name: str = "auditor_only"

    def should_fire(self, ctx: TriggerContext) -> TriggerDecision:
        return TriggerDecision(fire=True, reason="always", requires_extractor=False)


def test_empty_registry_evaluates_to_no_fire() -> None:
    reg = TriggerRegistry()
    auditor_due, extractor_due, reasons = reg.evaluate(_ctx())
    assert auditor_due is False
    assert extractor_due is False
    assert reasons == []


def test_single_trigger_fires() -> None:
    reg = TriggerRegistry()
    reg.register_trigger(_CadenceTrigger(interval=5))
    auditor_due, extractor_due, reasons = reg.evaluate(_ctx(turn_count=10))
    assert auditor_due is True
    assert extractor_due is True
    assert len(reasons) == 1
    assert "cadence" in reasons[0]


def test_single_trigger_does_not_fire() -> None:
    reg = TriggerRegistry()
    reg.register_trigger(_CadenceTrigger(interval=5))
    auditor_due, extractor_due, reasons = reg.evaluate(_ctx(turn_count=3))
    assert auditor_due is False
    assert extractor_due is False
    assert reasons == []


def test_or_semantics_compound_triggers() -> None:
    """With cadence + submission triggers, either firing causes auditor to run."""
    reg = TriggerRegistry()
    reg.register_trigger(_CadenceTrigger(interval=5))
    reg.register_trigger(_SubmissionTrigger())

    # Neither fires
    auditor_due, extractor_due, reasons = reg.evaluate(_ctx(turn_count=3))
    assert auditor_due is False
    assert reasons == []

    # Cadence fires (turn 5), submission does not
    auditor_due, extractor_due, reasons = reg.evaluate(_ctx(turn_count=5))
    assert auditor_due is True
    assert extractor_due is True
    assert len(reasons) == 1

    # Submission fires (turn 3), cadence does not
    auditor_due, extractor_due, reasons = reg.evaluate(
        _ctx(turn_count=3, terminal_tool_called="submit_final_report")
    )
    assert auditor_due is True
    assert extractor_due is True
    assert len(reasons) == 1
    assert "submit_final_report" in reasons[0]

    # Both fire (turn 10, with submission)
    auditor_due, extractor_due, reasons = reg.evaluate(
        _ctx(turn_count=10, terminal_tool_called="submit_final_report")
    )
    assert auditor_due is True
    assert extractor_due is True
    assert len(reasons) == 2


def test_extractor_due_false_when_no_trigger_requires_it() -> None:
    """A trigger with requires_extractor=False fires auditor but not extractor."""
    reg = TriggerRegistry()
    reg.register_trigger(_ExtractorFreeTrigger())

    auditor_due, extractor_due, reasons = reg.evaluate(_ctx())
    assert auditor_due is True
    assert extractor_due is False
    assert len(reasons) == 1


def test_extractor_due_true_when_mixed_triggers() -> None:
    """If any firing trigger requires extractor, extractor_due is True."""
    reg = TriggerRegistry()
    reg.register_trigger(_ExtractorFreeTrigger())
    reg.register_trigger(_CadenceTrigger(interval=1))

    auditor_due, extractor_due, reasons = reg.evaluate(_ctx(turn_count=1))
    assert auditor_due is True
    assert extractor_due is True
    assert len(reasons) == 2


def test_idempotent_registration() -> None:
    """Registering the same trigger instance twice is a no-op."""
    reg = TriggerRegistry()
    trigger = _CadenceTrigger(interval=5)
    reg.register_trigger(trigger)
    reg.register_trigger(trigger)
    assert len(reg.registered_triggers()) == 1


def test_distinct_triggers_with_same_name() -> None:
    """Two distinct trigger instances sharing a name both register."""
    reg = TriggerRegistry()
    reg.register_trigger(_CadenceTrigger(interval=5))
    reg.register_trigger(_CadenceTrigger(interval=3))
    assert len(reg.registered_triggers()) == 2
