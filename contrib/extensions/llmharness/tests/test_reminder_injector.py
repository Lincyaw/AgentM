"""Unit tests for the unified reminder injector.

The handler returned by ``_make_reminder_injector`` is registered on
``DecideTurnActionEvent`` and decides — given the kernel default action
and the current pending-reminder queue — whether to override the loop's
next move with ``Inject([reminder_msgs])``.

Three cases are pinned:

* default action ``Step`` (mid-trajectory, model called a tool): inject
  the reminders so they extend ``messages`` for the next turn.
* default action ``Stop`` with a non-final cause (terminal turn from
  ``submit_final_report`` / ``ModelEndTurn``): inject overrides the
  stop, re-opening the loop so the model can revise.
* default action ``Stop`` with a final cause (``MaxTurnsExhausted`` /
  ``SignalAborted``): kernel ignores overrides; handler is a no-op so
  pending reminders are NOT popped.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from agentm.core.abi import (
    DecideTurnActionEvent,
    Inject,
    MaxTurnsExhausted,
    ModelEndTurn,
    Step,
    Stop,
    ToolTerminated,
    TurnObservation,
)

from llmharness.adapters.agentm import _make_reminder_injector
from llmharness.schema import Reminder


def _make_event(default_action: Any) -> DecideTurnActionEvent:
    return DecideTurnActionEvent(
        observation=TurnObservation(
            turn_index=0,
            assistant_message=None,
            tool_outcomes=[],
            default_action=default_action,
        )
    )


def _make_api() -> MagicMock:
    api = MagicMock()
    api.session.append_entry = MagicMock()
    return api


def test_no_pending_reminders_returns_none() -> None:
    api = _make_api()
    handler = _make_reminder_injector(api, [])
    result = handler(_make_event(Step()))
    assert result is None
    api.session.append_entry.assert_not_called()


def test_step_default_with_reminder_injects() -> None:
    api = _make_api()
    pending = [Reminder(text="check baseline traces")]
    handler = _make_reminder_injector(api, pending)
    result = handler(_make_event(Step()))
    assert isinstance(result, Inject)
    assert len(result.messages) == 1
    assert "check baseline traces" in str(result.messages[0])
    assert pending == []  # popped
    api.session.append_entry.assert_called_once()


def test_terminal_non_final_stop_with_reminder_injects() -> None:
    """Stop(ToolTerminated) is non-final → reminder overrides the stop."""
    api = _make_api()
    pending = [Reminder(text="report incomplete")]
    handler = _make_reminder_injector(api, pending)
    result = handler(_make_event(Stop(ToolTerminated(tool_name="submit_final_report", reason=""))))
    assert isinstance(result, Inject)
    assert len(result.messages) == 1
    assert pending == []


def test_model_end_turn_stop_is_overridable() -> None:
    api = _make_api()
    pending = [Reminder(text="you stopped early")]
    handler = _make_reminder_injector(api, pending)
    result = handler(_make_event(Stop(ModelEndTurn())))
    assert isinstance(result, Inject)
    assert pending == []


def test_final_stop_leaves_reminder_pending() -> None:
    """MaxTurnsExhausted is final; kernel ignores overrides — preserve queue."""
    api = _make_api()
    pending = [Reminder(text="too late")]
    handler = _make_reminder_injector(api, pending)
    result = handler(_make_event(Stop(MaxTurnsExhausted())))
    assert result is None
    assert pending == [Reminder(text="too late")]
    api.session.append_entry.assert_not_called()


def test_multiple_reminders_concatenated_in_one_inject() -> None:
    api = _make_api()
    pending = [Reminder(text="one"), Reminder(text="two"), Reminder(text="three")]
    handler = _make_reminder_injector(api, pending)
    result = handler(_make_event(Step()))
    assert isinstance(result, Inject)
    assert len(result.messages) == 3
    assert pending == []
    assert api.session.append_entry.call_count == 3
