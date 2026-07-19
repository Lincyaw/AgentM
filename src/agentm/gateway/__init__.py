"""Gateway DTOs and pluggable host ports."""

from agentm.gateway.forwarding import ChildWireForwarder, ForwardedEnvelope
from agentm.gateway.ports import (
    AuthenticationResult,
    Authenticator,
    InboxLog,
    OutboxRecord,
    OutboxStore,
    PeerCredentials,
    SessionBinding,
    SessionBindingStore,
    TransportKind,
)
from agentm.gateway.wire import (
    WIRE_VERSION,
    ButtonStyle,
    DeliveryClass,
    GatewayEnvelope,
    InboundMessage,
    OutboundButton,
    OutboundDispatch,
    OutboundMessage,
    WireKind,
)

__all__ = [
    "WIRE_VERSION",
    "AuthenticationResult",
    "Authenticator",
    "ButtonStyle",
    "ChildWireForwarder",
    "DeliveryClass",
    "ForwardedEnvelope",
    "GatewayEnvelope",
    "InboxLog",
    "InboundMessage",
    "OutboxRecord",
    "OutboxStore",
    "OutboundButton",
    "OutboundDispatch",
    "OutboundMessage",
    "PeerCredentials",
    "SessionBinding",
    "SessionBindingStore",
    "TransportKind",
    "WireKind",
]
