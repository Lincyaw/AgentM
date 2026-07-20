"""Transport-neutral DTOs for the version-2 gateway wire."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

from agentm.core.abi.messages import JsonValue, freeze_json

WIRE_VERSION: Final = 2
WireKind = Literal[
    "hello",
    "welcome",
    "inbound",
    "outbound",
    "ack",
    "ping",
    "pong",
    "error",
]
DeliveryClass = Literal["durable", "ephemeral"]
_WIRE_KINDS: Final = frozenset(
    {
        "hello",
        "welcome",
        "inbound",
        "outbound",
        "ack",
        "ping",
        "pong",
        "error",
    }
)
_DELIVERY_CLASSES: Final = frozenset({"durable", "ephemeral"})
_BUTTON_STYLES: Final = frozenset({"primary", "danger", "default"})


def _empty_json_object() -> Mapping[str, JsonValue]:
    return MappingProxyType({})


def _freeze_object(
    value: Mapping[str, object],
    label: str,
) -> Mapping[str, JsonValue]:
    frozen = freeze_json(value)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return frozen


def _nonempty(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class GatewayEnvelope:
    """Minimal encoded envelope shared by every gateway transport."""

    envelope_id: str
    kind: WireKind
    timestamp: float
    session_key: str | None = None
    scenario: str | None = None
    body: Mapping[str, JsonValue] = field(default_factory=_empty_json_object)
    version: int = WIRE_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.envelope_id, "gateway envelope id")
        if self.kind not in _WIRE_KINDS:
            raise ValueError(f"unsupported gateway envelope kind: {self.kind!r}")
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version != WIRE_VERSION
        ):
            raise ValueError(f"unsupported gateway wire version: {self.version}")
        if (
            not isinstance(self.timestamp, (int, float))
            or isinstance(self.timestamp, bool)
            or not math.isfinite(self.timestamp)
        ):
            raise ValueError("gateway envelope timestamp must be finite")
        if self.kind in {"inbound", "outbound"}:
            if self.session_key is None:
                raise ValueError(f"{self.kind} envelope requires session_key")
            _nonempty(self.session_key, "gateway session_key")
        elif self.session_key is not None:
            _nonempty(self.session_key, "gateway session_key")
        if self.scenario is not None:
            if self.kind != "inbound":
                raise ValueError("scenario is only valid on inbound envelopes")
            _nonempty(self.scenario, "gateway scenario")
        object.__setattr__(
            self,
            "body",
            _freeze_object(self.body, "gateway envelope body"),
        )


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """Typed conversational payload carried by an inbound envelope."""

    channel: str
    chat_id: str
    content: str = ""
    thread_id: str | None = None
    sender_id: str = ""
    sender_name: str = ""
    button_value: str | None = None
    raw: Mapping[str, JsonValue] = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        _nonempty(self.channel, "inbound channel")
        _nonempty(self.chat_id, "inbound chat_id")
        for label, value in (
            ("content", self.content),
            ("sender_id", self.sender_id),
            ("sender_name", self.sender_name),
        ):
            if not isinstance(value, str):
                raise TypeError(f"inbound {label} must be a string")
        for label, optional_value in (
            ("thread_id", self.thread_id),
            ("button_value", self.button_value),
        ):
            if optional_value is not None:
                _nonempty(optional_value, f"inbound {label}")
        object.__setattr__(
            self,
            "raw",
            _freeze_object(self.raw, "inbound raw payload"),
        )


ButtonStyle = Literal["primary", "danger", "default"]


@dataclass(frozen=True, slots=True)
class OutboundButton:
    """One transport-neutral human-interaction action."""

    label: str
    value: str
    style: ButtonStyle = "default"

    def __post_init__(self) -> None:
        _nonempty(self.label, "outbound button label")
        _nonempty(self.value, "outbound button value")
        if self.style not in _BUTTON_STYLES:
            raise ValueError(f"unsupported outbound button style: {self.style!r}")


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    """Typed presenter payload carried by an outbound envelope."""

    channel: str
    chat_id: str
    content: str = ""
    thread_id: str | None = None
    buttons: tuple[OutboundButton, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        _nonempty(self.channel, "outbound channel")
        _nonempty(self.chat_id, "outbound chat_id")
        if not isinstance(self.content, str):
            raise TypeError("outbound content must be a string")
        if self.thread_id is not None:
            _nonempty(self.thread_id, "outbound thread_id")
        if not isinstance(self.buttons, tuple) or not all(
            isinstance(button, OutboundButton) for button in self.buttons
        ):
            raise TypeError("outbound buttons must be a tuple of OutboundButton")
        object.__setattr__(
            self,
            "metadata",
            _freeze_object(self.metadata, "outbound metadata"),
        )


@dataclass(frozen=True, slots=True)
class OutboundDispatch:
    """One envelope plus its explicit persistence/replay class."""

    envelope: GatewayEnvelope
    delivery: DeliveryClass

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, GatewayEnvelope):
            raise TypeError("outbound dispatch envelope must be a GatewayEnvelope")
        if self.envelope.kind != "outbound":
            raise ValueError("outbound dispatch requires an outbound envelope")
        if self.delivery not in _DELIVERY_CLASSES:
            raise ValueError(f"unsupported outbound delivery class: {self.delivery!r}")


__all__ = [
    "WIRE_VERSION",
    "ButtonStyle",
    "DeliveryClass",
    "GatewayEnvelope",
    "InboundMessage",
    "OutboundButton",
    "OutboundDispatch",
    "OutboundMessage",
    "WireKind",
]
