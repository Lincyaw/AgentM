"""Unit tests for the prefix-replay reminder seeder atom.

Fail-stop position: the seed atom MUST emit a message indistinguishable
from what the live adapter's reminder injector would have produced for
the same text. If those two paths drift, a student model trained on
live trajectories will see a different prompt-prefix at inference time
and the whole prefix-replay iteration loop loses its train/inference
parity guarantee.

The one-shot guarantee is the second invariant: a seed reminder is a
single experimental stimulus, not a recurring nudge — subsequent turns
must not see the same injection again.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from agentm.core.abi import (
    DecideTurnActionEvent,
    Inject,
    MaxTurnsExhausted,
    ModelEndTurn,
    Step,
    Stop,
    TurnObservation,
)

from llmharness.audit._reminder_format import REMINDER_PREAMBLE
from llmharness.replay.reminder_seed import install


def _make_event(default_action: Any) -> DecideTurnActionEvent:
    return DecideTurnActionEvent(
        observation=TurnObservation(
            turn_index=0,
            assistant_message=None,
            tool_outcomes=[],
            default_action=default_action,
        )
    )


def _make_api() -> tuple[MagicMock, list[Any]]:
    """Return (api_mock, captured_handlers).

    ``api.on`` is recorded so the test can grab the handler the atom
    registered. ``api.on`` returns an unsubscribe callable (also a
    MagicMock) so the atom's unsubscribe path is exercised.
    """
    api = MagicMock()
    api.session.append_entry = MagicMock()
    captured: list[Any] = []

    def _on(channel: str, handler: Any) -> Any:
        captured.append((channel, handler))
        return MagicMock(name="unsubscribe")

    api.on.side_effect = _on
    return api, captured


def test_first_event_injects_with_canonical_preamble() -> None:
    api, captured = _make_api()
    install(api, {"text": "investigate the missing baseline"})
    assert len(captured) == 1
    channel, handler = captured[0]
    assert channel == DecideTurnActionEvent.CHANNEL

    result = handler(_make_event(Step()))
    assert isinstance(result, Inject)
    assert len(result.messages) == 1
    rendered = str(result.messages[0])
    assert REMINDER_PREAMBLE.strip() in rendered
    assert "investigate the missing baseline" in rendered
    api.session.append_entry.assert_called_once()
    args, _ = api.session.append_entry.call_args
    assert args[1] == {"text": "investigate the missing baseline"}


def test_one_shot_does_not_refire() -> None:
    api, captured = _make_api()
    install(api, {"text": "one shot only"})
    handler = captured[0][1]

    first = handler(_make_event(Step()))
    assert isinstance(first, Inject)

    # Subsequent decide events must be silent.
    second = handler(_make_event(Step()))
    assert second is None
    third = handler(_make_event(Stop(ModelEndTurn())))
    assert third is None
    assert api.session.append_entry.call_count == 1


def test_final_cause_stop_leaves_seed_armed() -> None:
    """Mirror live adapter: final-cause stop ignores overrides; do not pop seed."""
    api, captured = _make_api()
    install(api, {"text": "still pending"})
    handler = captured[0][1]

    result = handler(_make_event(Stop(MaxTurnsExhausted())))
    assert result is None
    api.session.append_entry.assert_not_called()

    # A subsequent non-final turn should still deliver.
    delivered = handler(_make_event(Step()))
    assert isinstance(delivered, Inject)
    assert "still pending" in str(delivered.messages[0])


def test_non_final_stop_is_overridable() -> None:
    """Stop(ModelEndTurn) is non-final — seed re-opens the loop."""
    api, captured = _make_api()
    install(api, {"text": "you stopped early"})
    handler = captured[0][1]
    result = handler(_make_event(Stop(ModelEndTurn())))
    assert isinstance(result, Inject)
    assert "you stopped early" in str(result.messages[0])


def test_missing_text_rejected() -> None:
    api, _ = _make_api()
    with pytest.raises(ValueError, match="non-empty string"):
        install(api, {})


def test_empty_text_rejected() -> None:
    api, _ = _make_api()
    with pytest.raises(ValueError, match="non-empty string"):
        install(api, {"text": "   "})
