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
