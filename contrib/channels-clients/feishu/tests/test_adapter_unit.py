"""Unit tests for ``FeishuAdapter`` without contacting Feishu.

We bypass :meth:`FeishuAdapter.start` (which would import lark_oapi
and open a WebSocket) and instead inject a fake ``_channel`` attribute
plus drive the event handlers directly. The wire side is a stub that
records ``send`` calls so we can assert exactly what would have hit
the gateway.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agentm_channels.wire import (
    KIND_INBOUND,
    KIND_OUTBOUND,
    WIRE_VERSION,
    Envelope,
)
from agentm_feishu.adapter import FeishuAdapter, FeishuConfig, _markdown_card


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


async def test_message_event_forwards_inbound_envelope() -> None:
    adapter, client, fake = _make_adapter()
    msg = _FakeMessage(
        sender_id="ou_alice",
        chat_id="oc_chat",
        content_text="hello world",
    )

    await adapter._on_message(msg)

    # ACK-reaction task is fire-and-forget; let it complete.
    await asyncio.sleep(0)
    for tasks in list(adapter._pending_acks.values()):
        for t in tasks:
            await t

    # Exactly one inbound envelope on the wire.
    assert len(client.sent) == 1
    env = client.sent[0]
    assert env.kind == KIND_INBOUND
    assert env.v == WIRE_VERSION
    assert env.body == {
        "channel": "feishu",
        "sender_id": "ou_alice",
        "chat_id": "oc_chat",
        "content": "hello world",
    }
    # ACK reaction was attempted.
    assert fake.reactions_added == [("msg_1", "OK")]


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


async def test_card_action_forwards_button_value() -> None:
    adapter, client, _fake = _make_adapter()
    ev = _FakeCardEvent("approve_abc123", "ou_alice", "oc_chat")
    await adapter._on_card_action(ev)
    assert len(client.sent) == 1
    body = client.sent[0].body
    assert body["channel"] == "feishu"
    assert body["sender_id"] == "ou_alice"
    assert body["chat_id"] == "oc_chat"
    assert body["button_value"] == "approve_abc123"
    assert "[card click: approve_abc123]" in body["content"]


async def test_card_action_without_button_value_is_ignored() -> None:
    adapter, client, _fake = _make_adapter()

    class _Empty:
        action = _FakeAction({})  # no button_value key
        operator = _FakeOperator("ou_alice")
        chat_id = "oc_chat"
        message_id = "x"

    await adapter._on_card_action(_Empty())
    assert client.sent == []


# -- outbound: gateway envelope → Feishu card --------------------------


def _outbound_env(body: dict[str, Any]) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id="out-1",
        kind=KIND_OUTBOUND,
        ts=time.time(),
        body=body,
    )


async def test_handle_outbound_renders_markdown_card() -> None:
    adapter, _client, fake = _make_adapter()
    env = _outbound_env(
        {
            "channel": "feishu",
            "chat_id": "oc_chat",
            "content": "## Hello\n你好",
            "kind": "message",
        }
    )
    await adapter.handle_outbound(env)
    assert len(fake.sent) == 1
    chat_id, payload = fake.sent[0]
    assert chat_id == "oc_chat"
    card = payload["card"]
    assert card["schema"] == "2.0"
    assert card["body"]["elements"][0] == {
        "tag": "markdown",
        "content": "## Hello\n你好",
    }


async def test_handle_outbound_with_buttons_appends_action_block() -> None:
    adapter, _client, fake = _make_adapter()
    env = _outbound_env(
        {
            "channel": "feishu",
            "chat_id": "oc_chat",
            "content": "Approve?",
            "kind": "message",
            "buttons": [
                {"label": "Approve", "value": "ok", "style": "primary"},
                {"label": "Deny", "value": "no", "style": "danger"},
            ],
        }
    )
    await adapter.handle_outbound(env)
    card = fake.sent[0][1]["card"]
    elements = card["body"]["elements"]
    assert len(elements) == 2
    action = elements[1]
    assert action["tag"] == "action"
    actions = action["actions"]
    assert actions[0]["type"] == "primary"
    assert actions[0]["value"] == {"button_value": "ok"}
    assert actions[1]["type"] == "danger"
    assert actions[1]["value"] == {"button_value": "no"}


async def test_handle_outbound_turn_complete_clears_acks() -> None:
    adapter, _client, fake = _make_adapter()
    # Prime an ACK by running an inbound first.
    await adapter._on_message(
        _FakeMessage("ou_alice", "oc_chat", "hi", message_id="msg_1")
    )
    # Let the ACK-add task finish.
    for tasks in adapter._pending_acks.values():
        for t in tasks:
            await t
    env = _outbound_env(
        {"channel": "feishu", "chat_id": "oc_chat", "kind": "turn_complete"}
    )
    await adapter.handle_outbound(env)
    assert fake.reactions_removed == [("msg_1", "rx-msg_1")]
    # No card was sent for turn_complete.
    assert fake.sent == []


async def test_handle_outbound_empty_chat_id_drops() -> None:
    adapter, _client, fake = _make_adapter()
    env = _outbound_env(
        {"channel": "feishu", "chat_id": "", "content": "x", "kind": "message"}
    )
    await adapter.handle_outbound(env)
    assert fake.sent == []


# -- card construction helper -----------------------------------------


def test_markdown_card_empty_text_is_placeholder() -> None:
    card = _markdown_card("", buttons=[])
    assert card["body"]["elements"][0]["content"] == "*(empty)*"


def test_markdown_card_no_buttons_has_one_element() -> None:
    card = _markdown_card("hi", buttons=[])
    assert len(card["body"]["elements"]) == 1


# -- streaming (stream_id) --------------------------------------------


async def test_stream_first_frame_sends_and_remembers_message_id() -> None:
    adapter, _client, fake = _make_adapter(stream_debounce_s=0.01)
    env = _outbound_env(
        {
            "channel": "feishu",
            "chat_id": "oc_chat",
            "content": "thinking...",
            "kind": "message",
            "stream_id": "tool_call_42",
        }
    )
    await adapter.handle_outbound(env)
    assert len(fake.sent) == 1
    assert fake.patched == []
    # State retained until final / turn_complete.
    assert ("oc_chat", "tool_call_42") in adapter._streams


async def test_stream_subsequent_frames_patch_via_update_card() -> None:
    adapter, _client, fake = _make_adapter(stream_debounce_s=0.01)

    async def frame(content: str, *, final: bool = False) -> None:
        await adapter.handle_outbound(
            _outbound_env(
                {
                    "channel": "feishu",
                    "chat_id": "oc_chat",
                    "content": content,
                    "kind": "message",
                    "stream_id": "s1",
                    "final": final,
                }
            )
        )

    await frame("thinking...")
    await frame("reading file foo.py")
    await frame("running tests")
    # Let the debounce + flush task drain.
    await asyncio.sleep(0.05)
    # One real send, then one or more patches with the latest content.
    assert len(fake.sent) == 1
    assert len(fake.patched) >= 1
    last_card = fake.patched[-1][1]
    assert last_card["body"]["elements"][0]["content"] == "running tests"


async def test_stream_intermediate_frames_are_coalesced() -> None:
    adapter, _client, fake = _make_adapter(stream_debounce_s=0.02)
    # Burst-fire four frames before the debounce window expires.
    for i, content in enumerate(["a", "b", "c", "d"]):
        await adapter.handle_outbound(
            _outbound_env(
                {
                    "channel": "feishu",
                    "chat_id": "oc_chat",
                    "content": content,
                    "kind": "message",
                    "stream_id": "burst",
                    "final": i == 3,
                }
            )
        )
    # Wait long enough for the debounce sleep to finish + flush to run.
    await asyncio.sleep(0.1)
    # First send is the initial frame; remaining frames must collapse
    # into a single patch (the last value wins).
    assert len(fake.sent) == 1
    assert len(fake.patched) == 1
    assert fake.patched[0][1]["body"]["elements"][0]["content"] == "d"


async def test_stream_final_flushes_immediately_and_drops_state() -> None:
    adapter, _client, fake = _make_adapter(stream_debounce_s=10.0)
    # First frame — initial send.
    await adapter.handle_outbound(
        _outbound_env(
            {
                "channel": "feishu",
                "chat_id": "oc_chat",
                "content": "step 1",
                "kind": "message",
                "stream_id": "s2",
            }
        )
    )
    # Final frame — must NOT wait the 10 s debounce.
    await adapter.handle_outbound(
        _outbound_env(
            {
                "channel": "feishu",
                "chat_id": "oc_chat",
                "content": "all done",
                "kind": "message",
                "stream_id": "s2",
                "final": True,
            }
        )
    )
    await asyncio.sleep(0.01)
    assert len(fake.sent) == 1
    assert len(fake.patched) == 1
    assert fake.patched[0][1]["body"]["elements"][0]["content"] == "all done"
    assert ("oc_chat", "s2") not in adapter._streams


async def test_stream_distinct_ids_get_distinct_cards() -> None:
    adapter, _client, fake = _make_adapter(stream_debounce_s=0.01)
    # Two streams interleaved in the same chat — each must get its own
    # initial send.
    for sid, content in [("a", "A1"), ("b", "B1")]:
        await adapter.handle_outbound(
            _outbound_env(
                {
                    "channel": "feishu",
                    "chat_id": "oc_chat",
                    "content": content,
                    "kind": "message",
                    "stream_id": sid,
                }
            )
        )
    assert len(fake.sent) == 2
    # Each subsequent same-id frame becomes a patch of *that* card.
    await adapter.handle_outbound(
        _outbound_env(
            {
                "channel": "feishu",
                "chat_id": "oc_chat",
                "content": "A2",
                "kind": "message",
                "stream_id": "a",
                "final": True,
            }
        )
    )
    await asyncio.sleep(0.01)
    assert len(fake.patched) == 1
    # Stream "a" was sent first → message_id "msg-1"; stream "b" → "msg-2".
    assert fake.patched[0][0] == "msg-1"


async def test_stream_turn_complete_finalizes_open_streams() -> None:
    adapter, _client, fake = _make_adapter(stream_debounce_s=10.0)
    # Open a stream, queue a second frame that would normally wait
    # 10 s for the debounce.
    await adapter.handle_outbound(
        _outbound_env(
            {
                "channel": "feishu",
                "chat_id": "oc_chat",
                "content": "frame 1",
                "kind": "message",
                "stream_id": "s3",
            }
        )
    )
    await adapter.handle_outbound(
        _outbound_env(
            {
                "channel": "feishu",
                "chat_id": "oc_chat",
                "content": "frame 2",
                "kind": "message",
                "stream_id": "s3",
            }
        )
    )
    # turn_complete must drain the pending frame without waiting.
    await adapter.handle_outbound(
        _outbound_env(
            {"channel": "feishu", "chat_id": "oc_chat", "kind": "turn_complete"}
        )
    )
    await asyncio.sleep(0.01)
    assert len(fake.patched) == 1
    assert fake.patched[0][1]["body"]["elements"][0]["content"] == "frame 2"
    assert ("oc_chat", "s3") not in adapter._streams
