"""Fail-stop: Router three-case dispatch (§3.2).

The router is the single decision point for what happens to an inbound.
Mis-routing a button click as a prompt (or vice-versa) sends approval
clicks to the LLM and breaks the human-in-the-loop gate.
"""

from __future__ import annotations

import pytest

from agentm.gateway.router import ProtocolError, RouterAction, dispatch
from agentm.gateway.wire import WIRE_VERSION, Envelope


def _inbound(
    content: str = "",
    button_value: str | None = None,
    action: str | None = None,
    policy: str | None = None,
    interaction_id: str | None = None,
    request_id: str | None = None,
    control: str | None = None,
) -> Envelope:
    body: dict[str, object] = {
        "channel": "terminal",
        "chat_id": "t1",
        "sender_id": "u1",
        "content": content,
    }
    if button_value is not None:
        body["button_value"] = button_value
    if action is not None:
        body["action"] = action
    if policy is not None:
        body["policy"] = policy
    if interaction_id is not None:
        body["interaction_id"] = interaction_id
    if request_id is not None:
        body["request_id"] = request_id
    if control is not None:
        body["control"] = control
    return Envelope(
        v=WIRE_VERSION,
        id="i1",
        kind="inbound",
        ts=1.0,
        session_key="terminal:t1",
        body=body,
    )


def test_button_value_routes_to_resolve_approval() -> None:
    decision = dispatch(_inbound(content="[click]", button_value="appr-x:approve"))
    assert decision.action is RouterAction.RESOLVE_APPROVAL


def test_slash_command_routes_to_run_command() -> None:
    decision = dispatch(_inbound(content="/help"))
    assert decision.action is RouterAction.RUN_COMMAND


def test_double_slash_is_a_prompt_not_a_command() -> None:
    decision = dispatch(_inbound(content="//etc/hosts is a path"))
    assert decision.action is RouterAction.PROMPT_SESSION


def test_plain_text_routes_to_prompt_session() -> None:
    decision = dispatch(_inbound(content="hello there"))
    assert decision.action is RouterAction.PROMPT_SESSION


def test_explicit_submit_routes_to_submit() -> None:
    decision = dispatch(
        _inbound(
            content="new message",
            action="submit",
            request_id="r1",
        )
    )
    assert decision.action is RouterAction.SUBMIT


def test_explicit_run_command_routes_to_run_command() -> None:
    decision = dispatch(
        _inbound(
            content="/model gpt",
            action="run_command",
        )
    )
    assert decision.action is RouterAction.RUN_COMMAND


def test_explicit_interrupt_routes_to_interrupt() -> None:
    decision = dispatch(_inbound(action="interrupt"))
    assert decision.action is RouterAction.INTERRUPT


def test_explicit_interaction_response_routes_to_interaction_response() -> None:
    decision = dispatch(
        _inbound(
            action="interaction_response",
            button_value="appr-x:approve",
            interaction_id="ix-1",
        )
    )
    assert decision.action is RouterAction.INTERACTION_RESPONSE
    assert decision.body.interaction_id == "ix-1"


def test_explicit_unknown_action_raises() -> None:
    with pytest.raises(ProtocolError):
        dispatch(_inbound(action="weird"))


def test_button_value_wins_over_slash_content() -> None:
    # A button click whose content happens to look like a command must
    # still resolve the approval, not run a command.
    decision = dispatch(_inbound(content="/whatever", button_value="appr-x:deny"))
    assert decision.action is RouterAction.RESOLVE_APPROVAL


def test_control_interrupt_routes_to_interrupt() -> None:
    decision = dispatch(_inbound(control="interrupt"))
    assert decision.action is RouterAction.INTERRUPT


def test_interrupt_wins_over_button_and_command() -> None:
    # An interrupt is out-of-band: it preempts the in-flight prompt even if
    # the frame also looks like a command or carries a stale button_value.
    decision = dispatch(
        _inbound(content="/help", button_value="x", control="interrupt")
    )
    assert decision.action is RouterAction.INTERRUPT


def test_non_inbound_kind_raises() -> None:
    welcome = Envelope(v=WIRE_VERSION, id="w", kind="welcome", ts=1.0, body={})
    with pytest.raises(ProtocolError):
        dispatch(welcome)
