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
#
# Two delivery classes share this one envelope kind (see
# ``.claude/designs/textual-tui.md`` §4.3): DURABLE kinds go through the
# per-peer outbox (at-least-once, survive reconnect); EPHEMERAL kinds are
# written best-effort straight to the connected peer and dropped if it is
# absent (live decoration — streaming text, tool lifecycle, runtime
# control/observability events). The durable set is the reliability floor;
# everything else is ephemeral.
#
# This module is the *single home* of the wire kind vocabulary AND its
# delivery-class partition. The gateway sink (``agentm.gateway.cli``) imports
# :data:`DURABLE_OUTBOUND_KINDS` to route — it does not keep its own copy — and
# ``test_outbound_routing`` asserts the two sets partition
# ``OutboundMetaKind`` exactly (disjoint, union == every Literal member), so a
# new kind cannot be added to the Literal without being classified, and a
# kind cannot drift between the two layers.
OutboundMetaKind = Literal[
    # -- durable (reliability floor) --
    "assistant_text",
    "approval_request",
    "approval_resolved",
    "diagnostic_warning",
    "diagnostic_error",
    # -- ephemeral: conversation (live transcript) --
    "turn_start",
    "stream_text",
    "stream_thinking",
    "tool_call",
    "tool_result",
    "usage",
    "child_start",
    "child_end",
    "agent_end",
    # -- ephemeral: runtime control / observability --
    "extension_install",
    "extension_reload",
    "extension_unload",
    "api_register",
    "api_send_user_message",
    "resource_write",
    "plan_submitted",
    "after_compact",
    "cost_budget_exceeded",
    "session_ready",
    "command_dispatched",
]

# Delivery-class partition of OutboundMetaKind. DURABLE = the reliability
# floor (outbox, at-least-once); everything else is ephemeral live decoration.
DURABLE_OUTBOUND_KINDS: frozenset[str] = frozenset(
    {
        "assistant_text",
        "approval_request",
        "approval_resolved",
        "diagnostic_warning",
        "diagnostic_error",
    }
)
EPHEMERAL_OUTBOUND_KINDS: frozenset[str] = frozenset(
    {
        "turn_start",
        "stream_text",
        "stream_thinking",
        "tool_call",
        "tool_result",
        "usage",
        "child_start",
        "child_end",
        "agent_end",
        "extension_install",
        "extension_reload",
        "extension_unload",
        "api_register",
        "api_send_user_message",
        "resource_write",
        "plan_submitted",
        "after_compact",
        "cost_budget_exceeded",
        "session_ready",
        "command_dispatched",
    }
)


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
    # Out-of-band control verb that is NOT a conversational turn. Currently
    # ``"interrupt"`` — preempt the in-flight prompt (the gateway routes it to
    # AgentSession.interrupt() inline, never as a new turn). Distinct from
    # ``content`` so an interrupt can't be mistaken for a user message.
    control: str | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> InboundBody:
        button = body.get("button_value")
        control = body.get("control")
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
            control=str(control) if control is not None else None,
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
        if self.control is not None:
            out["control"] = self.control
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


__all__ = [
    "DURABLE_OUTBOUND_KINDS",
    "EPHEMERAL_OUTBOUND_KINDS",
    "Button",
    "ButtonStyle",
    "InboundBody",
    "OutboundBody",
    "OutboundMetaKind",
]
