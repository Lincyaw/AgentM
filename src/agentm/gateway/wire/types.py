"""Typed inbound / outbound body shapes for the v2 wire envelope.

These dataclasses are the structured view of an :class:`Envelope`'s
``body`` for the two payload-carrying kinds (``inbound`` / ``outbound``,
§2.4 / §2.5). They replace the v1 ``bus.py`` ``InboundMessage`` /
``OutboundMessage`` queue types — the gateway no longer has an internal
bus, so these are pure parse/serialise helpers between the wire dict and
typed access.

Pure module: no I/O, no asyncio. Buttons travel as plain dicts on the
wire; ``Button`` is the typed mirror used gateway-side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ButtonStyle = Literal["primary", "danger", "default"]

# Discriminator for how a chat client renders an outbound (§2.5).
OutboundMetaKind = Literal[
    "assistant_text",
    "approval_request",
    "approval_resolved",
    "diagnostic_warning",
    "diagnostic_error",
]


@dataclass(frozen=True, slots=True)
class Button:
    """A human-in-the-loop action button on an outbound (§2.5).

    ``value`` round-trips back on the inbound side as
    :attr:`InboundBody.button_value` when the user clicks.
    """

    label: str
    value: str
    style: ButtonStyle = "default"

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "value": self.value, "style": self.style}


@dataclass(slots=True)
class InboundBody:
    """Structured view of an ``inbound`` envelope body (§2.4)."""

    channel: str
    chat_id: str
    content: str = ""
    thread_id: str | None = None
    sender_id: str = ""
    sender_name: str = ""
    button_value: str | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> InboundBody:
        button = body.get("button_value")
        raw = body.get("raw")
        return cls(
            channel=str(body.get("channel") or ""),
            chat_id=str(body.get("chat_id") or ""),
            content=str(body.get("content") or ""),
            thread_id=(
                str(body["thread_id"])
                if body.get("thread_id") is not None
                else None
            ),
            sender_id=str(body.get("sender_id") or ""),
            sender_name=str(body.get("sender_name") or ""),
            button_value=str(button) if button is not None else None,
            raw=raw if isinstance(raw, dict) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "channel": self.channel,
            "chat_id": self.chat_id,
            "content": self.content,
        }
        if self.thread_id is not None:
            out["thread_id"] = self.thread_id
        if self.sender_id:
            out["sender_id"] = self.sender_id
        if self.sender_name:
            out["sender_name"] = self.sender_name
        if self.button_value is not None:
            out["button_value"] = self.button_value
        if self.raw is not None:
            out["raw"] = self.raw
        return out


@dataclass(slots=True)
class OutboundBody:
    """Structured view of an ``outbound`` envelope body (§2.5)."""

    channel: str
    chat_id: str
    content: str = ""
    thread_id: str | None = None
    buttons: list[Button] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> OutboundBody:
        buttons: list[Button] = []
        for raw in body.get("buttons") or []:
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label", ""))
            value = str(raw.get("value", ""))
            style = str(raw.get("style", "default"))
            if label and value:
                buttons.append(
                    Button(
                        label=label,
                        value=value,
                        style=style if style in ("primary", "danger", "default") else "default",  # type: ignore[arg-type]
                    )
                )
        meta = body.get("metadata")
        return cls(
            channel=str(body.get("channel") or ""),
            chat_id=str(body.get("chat_id") or ""),
            content=str(body.get("content") or ""),
            thread_id=(
                str(body["thread_id"])
                if body.get("thread_id") is not None
                else None
            ),
            buttons=buttons,
            metadata=dict(meta) if isinstance(meta, dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "channel": self.channel,
            "chat_id": self.chat_id,
            "content": self.content,
            "metadata": dict(self.metadata),
        }
        if self.thread_id is not None:
            out["thread_id"] = self.thread_id
        if self.buttons:
            out["buttons"] = [b.to_dict() for b in self.buttons]
        return out


__all__ = ["Button", "ButtonStyle", "InboundBody", "OutboundBody", "OutboundMetaKind"]
