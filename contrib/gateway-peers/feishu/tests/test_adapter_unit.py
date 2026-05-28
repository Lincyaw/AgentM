"""Unit tests for ``FeishuAdapter`` without contacting Feishu.

We bypass :meth:`FeishuAdapter.start` (which would import lark_oapi
and open a WebSocket) and instead inject a fake ``_channel`` attribute
plus drive the event handlers directly. The wire side is a stub that
records ``send`` calls so we can assert exactly what would have hit
the gateway.
"""

from __future__ import annotations

import time
from typing import Any

from agentm_channels.wire import (
    KIND_OUTBOUND,
    WIRE_VERSION,
    Envelope,
)
from agentm_feishu.adapter import FeishuAdapter, FeishuConfig


class _StubClient:
    """Records :meth:`send` calls so tests can inspect the envelopes."""

    def __init__(self) -> None:
        self.sent: list[Envelope] = []

    async def send(self, env: Envelope) -> None:  # noqa: D401
        self.sent.append(env)


class _FakeSendResult:
    """Minimal stand-in for ``lark_oapi.channel.SendResult``."""

    def __init__(self, message_id: str | None) -> None:
        self.message_id = message_id


class _FakeLarkChannel:
    """Minimal stand-in for ``lark_oapi.channel.FeishuChannel``."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.reactions_added: list[tuple[str, str]] = []
        self.reactions_removed: list[tuple[str, str]] = []
        self.patched: list[tuple[str, dict[str, Any]]] = []
        self._bot = None  # so adapter._is_self() is always False
        self._send_counter = 0

    async def send(
        self, chat_id: str, payload: dict[str, Any]
    ) -> _FakeSendResult:
        self.sent.append((chat_id, payload))
        self._send_counter += 1
        return _FakeSendResult(message_id=f"msg-{self._send_counter}")

    async def update_card(
        self, message_id: str, card: dict[str, Any]
    ) -> _FakeSendResult:
        self.patched.append((message_id, card))
        return _FakeSendResult(message_id=message_id)

    async def add_reaction(self, message_id: str, emoji: str) -> Any:
        self.reactions_added.append((message_id, emoji))
        # Shape mirrors what lark-oapi returns: ``result.raw["data"]["reaction_id"]``.

        class _R:
            raw = {"data": {"reaction_id": f"rx-{message_id}"}}

        return _R()

    async def remove_reaction(self, message_id: str, reaction_id: str) -> None:
        self.reactions_removed.append((message_id, reaction_id))


def _make_adapter(
    allow: list[str] | None = None,
    *,
    stream_debounce_s: float = 0.3,
) -> tuple[FeishuAdapter, _StubClient, _FakeLarkChannel]:
    client = _StubClient()
    cfg = FeishuConfig(
        app_id="cli_xxxx",
        app_secret="secret",
        allow_from=allow if allow is not None else ["*"],
        chat_id_prefix="feishu",
        channel_name="feishu",
        ack_emoji="OK",
        stream_debounce_s=stream_debounce_s,
    )
    adapter = FeishuAdapter(client=client, config=cfg)  # type: ignore[arg-type]
    fake_channel = _FakeLarkChannel()
    adapter._channel = fake_channel  # type: ignore[assignment]
    return adapter, client, fake_channel


# -- inbound: messageReceiveEvent → wire ------------------------------


class _FakeMessage:
    def __init__(
        self,
        sender_id: str,
        chat_id: str,
        content_text: str,
        message_id: str | None = "msg_1",
    ) -> None:
        self.sender_id = sender_id
        self.chat_id = chat_id
        self.content_text = content_text
        self.message_id = message_id




async def test_message_event_disallowed_sender_is_dropped() -> None:
    adapter, client, _fake = _make_adapter(allow=["ou_bob"])
    msg = _FakeMessage(
        sender_id="ou_alice", chat_id="oc_chat", content_text="hi"
    )
    await adapter._on_message(msg)
    assert client.sent == []


async def test_message_event_empty_allow_list_denies_all() -> None:
    adapter, client, _fake = _make_adapter(allow=[])
    msg = _FakeMessage(
        sender_id="ou_alice", chat_id="oc_chat", content_text="hi"
    )
    await adapter._on_message(msg)
    assert client.sent == []


# -- inbound: cardAction → wire ---------------------------------------


class _FakeOperator:
    def __init__(self, open_id: str) -> None:
        self.open_id = open_id
        self.user_id = None


class _FakeAction:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value


class _FakeCardEvent:
    def __init__(self, button_value: str, sender: str, chat_id: str) -> None:
        self.action = _FakeAction({"button_value": button_value})
        self.operator = _FakeOperator(sender)
        self.chat_id = chat_id
        self.message_id = "card_msg_1"






# -- outbound: gateway envelope → Feishu card --------------------------


def _outbound_env(body: dict[str, Any]) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id="out-1",
        kind=KIND_OUTBOUND,
        ts=time.time(),
        body=body,
    )










# -- card construction helper -----------------------------------------






# -- streaming (stream_id) --------------------------------------------












