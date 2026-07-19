"""Gateway child-session forwarding contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ForwardedEnvelope:
    source_session_id: str
    target_session_id: str
    channel: str
    payload: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class ChildWireForwarder(Protocol):
    """Forward child-session events over a presenter/gateway wire."""

    async def forward(self, envelope: ForwardedEnvelope) -> None: ...


__all__ = ["ChildWireForwarder", "ForwardedEnvelope"]
