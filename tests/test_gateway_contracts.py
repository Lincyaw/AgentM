"""Behavior contracts shared by every gateway transport and store backend."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from agentm.gateway import (
    WIRE_VERSION,
    AuthenticationResult,
    GatewayEnvelope,
    OutboundButton,
    OutboundDispatch,
    PeerCredentials,
    SessionBinding,
)


def test_gateway_envelope_freezes_portable_body() -> None:
    source = {"metadata": {"kind": "assistant_text"}, "items": [1, 2]}
    envelope = GatewayEnvelope(
        envelope_id="env-1",
        kind="outbound",
        timestamp=1.0,
        session_key="chat:1",
        body=source,
    )

    source["metadata"] = {"kind": "changed"}
    assert isinstance(envelope.body["metadata"], Mapping)
    assert envelope.body["metadata"]["kind"] == "assistant_text"
    assert envelope.body["items"] == (1, 2)


def test_gateway_envelope_rejects_routing_and_version_drift() -> None:
    with pytest.raises(ValueError, match="requires session_key"):
        GatewayEnvelope(
            envelope_id="env-1",
            kind="inbound",
            timestamp=1.0,
        )
    with pytest.raises(ValueError, match="scenario is only valid"):
        GatewayEnvelope(
            envelope_id="env-2",
            kind="outbound",
            timestamp=1.0,
            session_key="chat:1",
            scenario="chat",
        )
    with pytest.raises(ValueError, match="unsupported gateway wire version"):
        GatewayEnvelope(
            envelope_id="env-3",
            kind="ping",
            timestamp=1.0,
            version=WIRE_VERSION - 1,
        )
    with pytest.raises(ValueError, match="unsupported gateway envelope kind"):
        GatewayEnvelope(
            envelope_id="env-4",
            kind="event",  # type: ignore[arg-type]
            timestamp=1.0,
        )


def test_gateway_delivery_and_authentication_are_explicit() -> None:
    outbound = GatewayEnvelope(
        envelope_id="env-1",
        kind="outbound",
        timestamp=1.0,
        session_key="chat:1",
    )
    dispatch = OutboundDispatch(
        envelope=outbound,
        delivery="durable",
    )
    assert dispatch.delivery == "durable"

    with pytest.raises(ValueError, match="outbound envelope"):
        OutboundDispatch(
            envelope=GatewayEnvelope(
                envelope_id="env-2",
                kind="ping",
                timestamp=1.0,
            ),
            delivery="ephemeral",
        )
    with pytest.raises(ValueError, match="principal_id"):
        AuthenticationResult(accepted=True)
    with pytest.raises(ValueError, match="requires a reason"):
        AuthenticationResult(accepted=False)
    with pytest.raises(ValueError, match="delivery class"):
        OutboundDispatch(
            envelope=outbound,
            delivery="best-effort",  # type: ignore[arg-type]
        )


def test_gateway_peer_and_presenter_values_fail_at_the_boundary() -> None:
    with pytest.raises(ValueError, match="peer transport"):
        PeerCredentials(
            peer_id="peer-1",
            transport="tcp",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="button style"):
        OutboundButton(
            label="Retry",
            value="retry",
            style="warning",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="updated_at"):
        SessionBinding(
            session_key="chat:1",
            session_id="session-1",
            updated_at=float("nan"),
        )
