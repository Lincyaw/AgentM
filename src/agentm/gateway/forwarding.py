# code-health: ignore-file[AM025] -- gateway wire DTOs validate external transport payloads
"""Gateway child-session forwarding contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentm.gateway.wire import DeliveryClass, GatewayEnvelope


@dataclass(frozen=True, slots=True)
class ForwardedEnvelope:
    source_session_id: str
    target_session_id: str
    envelope: GatewayEnvelope
    delivery: DeliveryClass

    def __post_init__(self) -> None:
        if not self.source_session_id or not self.target_session_id:
            raise ValueError("forwarded session ids must be non-empty")
        if not isinstance(self.envelope, GatewayEnvelope):
            raise TypeError("forwarded envelope must be a GatewayEnvelope")
        if self.delivery not in {"durable", "ephemeral"}:
            raise ValueError(
                f"unsupported forwarding delivery class: {self.delivery!r}"
            )


@runtime_checkable
class ChildWireForwarder(Protocol):
    """Forward child-session events over a presenter/gateway wire."""

    async def forward(self, envelope: ForwardedEnvelope) -> None: ...


__all__ = ["ChildWireForwarder", "ForwardedEnvelope"]
