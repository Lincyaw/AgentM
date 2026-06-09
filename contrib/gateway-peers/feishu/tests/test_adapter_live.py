"""Live-turn rendering: the fail-stop property is that a whole agent run
collapses into ONE in-place card (one ``send`` + N ``update_card``), not a
card per wire event. Without this the post-#187 full event surface would
flood the Feishu chat with a card per token / tool / usage frame.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

import agentm_feishu.adapter as adapter_mod
from agentm_feishu.adapter import FeishuAdapter, FeishuConfig
from agentm.gateway.wire import KIND_INBOUND, KIND_OUTBOUND, WIRE_VERSION, Envelope


class _SendResult:
    """Mirrors lark's structured ``SendResult`` (NOT a bare string)."""

    def __init__(self, message_id: str) -> None:
        self.message_id = message_id
        self.ok = True


class _FakeChannel:
    """Records card sends / patches in order."""

    def __init__(self) -> None:
        self.sends: list[tuple[str, dict[str, Any]]] = []
        self.updates: list[tuple[str, dict[str, Any]]] = []
        self._next = 0

    async def send(
        self, to: str, message: dict[str, Any], opts: Any = None
    ) -> _SendResult:
        self.sends.append((to, message))
        self._next += 1
        return _SendResult(f"om_msg-{self._next}")

    async def update_card(self, message_id: str, card: dict[str, Any]) -> _SendResult:
        self.updates.append((message_id, card))
        return _SendResult(message_id)


def _outbound(chat_id: str, kind: str, **fields: Any) -> Envelope:
    meta = {"kind": kind, **fields.pop("meta", {})}
    body: dict[str, Any] = {
        "channel": "feishu",
        "chat_id": chat_id,
        "content": fields.pop("content", ""),
        "metadata": meta,
        **fields,
    }
    return Envelope(
        v=WIRE_VERSION,
        id=f"out-{time.time_ns()}",
        kind=KIND_OUTBOUND,
        ts=time.time(),
        session_key="feishu:oc_1",
        body=body,
    )


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> tuple[FeishuAdapter, _FakeChannel]:
    # Render inline (no deferred trailing flush) for deterministic counting,
    # and shrink the post-agent_end grace window so tests don't wait 1.5s.
    monkeypatch.setattr(adapter_mod, "_MIN_UPDATE_INTERVAL", 0.0)
    monkeypatch.setattr(adapter_mod, "_AGENT_END_GRACE", 0.02)
    cfg = FeishuConfig(app_id="x", app_secret="y", channel_name="feishu")
    adapter = FeishuAdapter(client=object(), config=cfg)  # type: ignore[arg-type]
    channel = _FakeChannel()
    adapter._channel = channel
    return adapter, channel


def _last_card(channel: _FakeChannel) -> dict[str, Any]:
    if channel.updates:
        return channel.updates[-1][1]
    return channel.sends[-1][1]["card"]


def _element(card: dict[str, Any], element_id: str) -> dict[str, Any] | None:
    for el in card["body"]["elements"]:
        if el.get("element_id") == element_id:
            return el
    return None


@pytest.mark.asyncio
async def test_run_collapses_into_one_card(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, channel = _make_adapter(monkeypatch)
    chat = "oc_1"
    events = [
        _outbound(chat, "session_ready"),  # noise
        _outbound(chat, "turn_start", meta={"turn_id": "t1"}),
        _outbound(
            chat, "tool_call", meta={"tool_call_id": "c1", "name": "read_file",
                                     "args": {"path": "config.py"}}
        ),
        _outbound(chat, "usage", meta={"input_tokens": 10}),  # noise
        _outbound(
            chat, "tool_result", meta={"tool_call_id": "c1", "name": "read_file",
                                       "ok": True}
        ),
        _outbound(chat, "stream_text", content="par"),
        _outbound(chat, "assistant_text", content="the answer"),
        _outbound(chat, "agent_end", meta={"cause": "WaitForUser"}),
    ]
    for env in events:
        await adapter.handle_outbound(env)
    await asyncio.sleep(0.05)  # let the post-agent_end grace window expire

    # Exactly one card created for the whole run; the rest are in-place patches.
    assert len(channel.sends) == 1
    assert len(channel.updates) >= 1
    # Every patch must target the message_id returned by send (not a repr).
    assert all(mid == "om_msg-1" for mid, _ in channel.updates)

    final = _last_card(channel)
    status = _element(final, "status")
    body = _element(final, "body")
    steps = _element(final, "steps")
    assert status is not None and status["content"] == adapter_mod._STATUS_DONE
    assert body is not None and body["content"] == "the answer"
    assert steps is not None and "read_file" in steps["elements"][0]["content"]
    assert "✅" in steps["elements"][0]["content"]  # tool result marked ok
    # Turn retired so the next message starts a fresh card.
    assert adapter._live == {}


@pytest.mark.asyncio
async def test_multi_turn_with_trailing_final_text_stays_one_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real Feishu run: a multi-turn agent emits an assistant_text per
    turn, and the ephemeral agent_end overtakes the durable FINAL
    assistant_text. The intermediate texts must not finalize early, and the
    trailing final text must fill the SAME card — never a second one.
    """
    adapter, channel = _make_adapter(monkeypatch)
    chat = "oc_1"
    for env in [
        # turn 1: intermediate reply
        _outbound(chat, "turn_start", meta={"turn_id": "t1"}),
        _outbound(
            chat, "tool_call",
            meta={"tool_call_id": "c1", "name": "list_dir", "args": {}},
        ),
        _outbound(chat, "tool_result", meta={"tool_call_id": "c1", "ok": True}),
        _outbound(chat, "assistant_text", content="let me look"),  # intermediate
        # turn 2: agent_end overtakes the final answer text
        _outbound(chat, "turn_start", meta={"turn_id": "t2"}),
        _outbound(chat, "usage", meta={"input_tokens": 5}),
        _outbound(chat, "agent_end", meta={"cause": "WaitForUser"}),
        _outbound(chat, "assistant_text", content="the real final answer"),
    ]:
        await adapter.handle_outbound(env)

    # Intermediate assistant_text must NOT have finalized the turn.
    assert adapter._live != {}
    await asyncio.sleep(0.05)  # grace window expires -> finalize

    # Still exactly one card; the trailing final text patched it in place.
    assert len(channel.sends) == 1
    final = _last_card(channel)
    status = _element(final, "status")
    body = _element(final, "body")
    assert status is not None and status["content"] == adapter_mod._STATUS_DONE
    assert body is not None and body["content"] == "the real final answer"
    assert adapter._live == {}


@pytest.mark.asyncio
async def test_text_less_run_finalizes_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pure-tool run with no assistant_text finalizes via the grace timer."""
    adapter, channel = _make_adapter(monkeypatch)
    chat = "oc_1"
    for env in [
        _outbound(chat, "turn_start", meta={"turn_id": "t1"}),
        _outbound(
            chat, "tool_call",
            meta={"tool_call_id": "c1", "name": "edit_file", "args": {}},
        ),
        _outbound(chat, "tool_result", meta={"tool_call_id": "c1", "ok": True}),
        _outbound(chat, "agent_end", meta={"cause": "WaitForUser"}),
    ]:
        await adapter.handle_outbound(env)

    # agent_end alone does not finalize — the grace timer is still pending.
    assert adapter._live != {}
    await asyncio.sleep(0.05)
    final = _last_card(channel)
    status = _element(final, "status")
    assert status is not None and status["content"] == adapter_mod._STATUS_DONE
    assert adapter._live == {}


@pytest.mark.asyncio
async def test_oneshot_command_reply_is_its_own_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slash-command reply (bare assistant_text, no turn_start/agent_end)
    must finalize immediately as its own card. Otherwise it becomes a zombie
    turn with no agent_end to retire it, and EVERY later reply reuses (patches)
    that one stale card — the live bug. Two commands in a row -> two cards."""
    adapter, channel = _make_adapter(monkeypatch)
    chat = "oc_1"

    await adapter.handle_outbound(_outbound(chat, "assistant_text", content="model: doubao"))
    assert len(channel.sends) == 1
    assert adapter._live == {}  # retired immediately, no zombie turn

    await adapter.handle_outbound(_outbound(chat, "assistant_text", content="help text"))
    # A SECOND card, not a patch of the first.
    assert len(channel.sends) == 2
    assert adapter._live == {}

    # And a subsequent real agent turn opens a fresh card too.
    for env in [
        _outbound(chat, "turn_start", meta={"turn_id": "t1"}),
        _outbound(chat, "agent_end", meta={"cause": "WaitForUser"}),
        _outbound(chat, "assistant_text", content="agent answer"),
    ]:
        await adapter.handle_outbound(env)
    await asyncio.sleep(0.05)
    assert len(channel.sends) == 3
    assert _element(_last_card(channel), "body")["content"] == "agent answer"


@pytest.mark.asyncio
async def test_noise_kinds_emit_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, channel = _make_adapter(monkeypatch)
    for kind in ("usage", "child_start", "extension_install", "command_dispatched",
                 "api_register", "resource_write"):
        await adapter.handle_outbound(_outbound("oc_1", kind))
    assert channel.sends == []
    assert channel.updates == []


class _FakeWireClient:
    """Records inbound envelopes the adapter forwards (the observable wire)."""

    def __init__(self) -> None:
        self.sent: list[Envelope] = []

    async def send(self, env: Envelope) -> None:
        self.sent.append(env)


class _CardAction:
    """Mirrors the lark cardAction event shape ``_on_card_action`` reads."""

    def __init__(
        self,
        *,
        value: dict[str, Any],
        chat_id: str,
        sender: str,
        message_id: str = "",
    ) -> None:
        self.action = type("A", (), {"value": value})()
        self.chat_id = chat_id
        self.message_id = message_id
        self.operator = type("O", (), {"open_id": sender})()


def _last_inbound(client: _FakeWireClient) -> dict[str, Any]:
    env = client.sent[-1]
    assert env.kind == KIND_INBOUND
    assert isinstance(env.body, dict)
    return env.body


@pytest.mark.asyncio
async def test_stop_button_click_forwards_interrupt_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fail-stop property: a 停止 click must reach the gateway as an
    out-of-band ``control="interrupt"`` verb (so ``InboundBody.control`` fires
    and the running turn is aborted), NOT as a conversational button_value."""
    adapter, _channel = _make_adapter(monkeypatch)
    client = _FakeWireClient()
    adapter._client = client  # type: ignore[assignment]

    event = _CardAction(
        value={"control": "interrupt"}, chat_id="oc_1", sender="ou_user"
    )
    await adapter._on_card_action(event)

    body = _last_inbound(client)
    assert body["control"] == "interrupt"
    assert "button_value" not in body  # an interrupt is never an approval reply


@pytest.mark.asyncio
async def test_approval_button_click_forwards_button_value_not_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approval buttons must keep round-tripping ``button_value`` and never
    leak a ``control`` verb — the two card vocabularies stay disjoint."""
    adapter, _channel = _make_adapter(monkeypatch)
    client = _FakeWireClient()
    adapter._client = client  # type: ignore[assignment]

    event = _CardAction(
        value={"button_value": "a:approve"}, chat_id="oc_1", sender="ou_user"
    )
    await adapter._on_card_action(event)

    body = _last_inbound(client)
    assert body["button_value"] == "a:approve"
    assert "control" not in body


@pytest.mark.asyncio
async def test_live_card_carries_stop_button_until_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-progress turn card shows the stop button; once finalized it is
    dropped so a retired card cannot invite a second abort."""
    adapter, channel = _make_adapter(monkeypatch)
    chat = "oc_1"
    await adapter.handle_outbound(_outbound(chat, "turn_start", meta={"turn_id": "t1"}))

    # In-flight: the stop action is present and round-trips a control verb.
    live = channel.sends[-1][1]["card"]
    stop = _element(live, "stop")
    assert stop is not None
    assert stop["value"] == {"control": "interrupt"}

    # After the run completes, the stop button is gone from the final card.
    for env in [
        _outbound(chat, "agent_end", meta={"cause": "WaitForUser"}),
        _outbound(chat, "assistant_text", content="done"),
    ]:
        await adapter.handle_outbound(env)
    await asyncio.sleep(0.05)
    assert _element(_last_card(channel), "stop") is None


@pytest.mark.asyncio
async def test_interrupt_marks_card_and_drops_stop_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hitting 停止 repaints the live card to the 已中断 state and removes the
    stop button immediately, before the server-side abort lands."""
    adapter, channel = _make_adapter(monkeypatch)
    client = _FakeWireClient()
    adapter._client = client  # type: ignore[assignment]
    chat = "oc_1"
    await adapter.handle_outbound(_outbound(chat, "turn_start", meta={"turn_id": "t1"}))

    # The card action must carry the card's message_id so the adapter can
    # look up the correct live turn (thread-aware).
    card_mid = channel.sends[-1][0] if not channel.sends else ""
    card_mid = "om_msg-1"  # the _FakeChannel assigned this on the first send
    await adapter._on_card_action(
        _CardAction(
            value={"control": "interrupt"}, chat_id=chat, sender="ou_user",
            message_id=card_mid,
        )
    )

    final = _last_card(channel)
    status = _element(final, "status")
    assert status is not None and status["content"] == adapter_mod._STATUS_INTERRUPTED
    assert _element(final, "stop") is None


@pytest.mark.asyncio
async def test_markdown_image_in_reply_does_not_poison_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A markdown image in agent text must be neutralized before it reaches a
    card — Lark rejects a non-img_key image and crashes the WHOLE card
    (ErrCode 200570 invalid image keys). The body must carry no `![...](...)`."""
    adapter, channel = _make_adapter(monkeypatch)
    chat = "oc_1"
    for env in [
        _outbound(chat, "turn_start", meta={"turn_id": "t1"}),
        _outbound(chat, "assistant_text", content="see ![shot](lark_login.png) done"),
        _outbound(chat, "agent_end", meta={"cause": "WaitForUser"}),
    ]:
        await adapter.handle_outbound(env)
    await asyncio.sleep(0.05)
    body = _element(_last_card(channel), "body")
    assert body is not None
    assert "![" not in body["content"]
    assert "lark_login.png" not in body["content"] or "](" not in body["content"]
    # the alt text survives as plain text
    assert "shot" in body["content"]


def test_md_safe_degrades_images_to_text() -> None:
    assert "![" not in adapter_mod._md_safe("a ![x](k.png) b")
    assert adapter_mod._md_safe("a ![x](k.png) b") == "a x b"
    assert adapter_mod._md_safe("![](only.png)") == "only.png"
    assert adapter_mod._md_safe("no image") == "no image"


class _Mention:
    """Mirrors the lark ``Mention`` fields the adapter reads."""

    def __init__(
        self, *, key: str, open_id: str | None, name: str | None = None
    ) -> None:
        self.key = key
        self.open_id = open_id
        self.name = name


class _Conversation:
    """Mirrors lark's ``Conversation`` dataclass."""

    def __init__(self, *, chat_id: str, thread_id: str | None = None) -> None:
        self.chat_id = chat_id
        self.thread_id = thread_id


class _InboundMsg:
    """Minimal stand-in for lark's ``InboundMessage`` (only the fields
    ``_on_message`` reads via getattr)."""

    def __init__(
        self,
        *,
        content_text: str,
        mentions: list[_Mention] | None = None,
        sender_id: str = "ou_user",
        chat_id: str = "oc_grp",
        message_id: str = "om_1",
        thread_id: str | None = None,
    ) -> None:
        self.content_text = content_text
        self.mentions = mentions or []
        self.sender_id = sender_id
        self.chat_id = chat_id
        self.message_id = message_id
        self.conversation = _Conversation(
            chat_id=chat_id, thread_id=thread_id
        )


_BOT_OID = "ou_bot"


def _make_mention_adapter() -> tuple[FeishuAdapter, _FakeWireClient]:
    """Adapter wired with a bot identity + recording wire client. ``_main_loop``
    stays None so ``_on_message`` skips the (loop-bound) ack reaction."""
    cfg = FeishuConfig(app_id="x", app_secret="y", channel_name="feishu",
                       allow_from=["*"])
    adapter = FeishuAdapter(client=object(), config=cfg)  # type: ignore[arg-type]
    bot = type("Bot", (), {"open_id": _BOT_OID})()
    adapter._channel = type("Ch", (), {"_bot": bot})()
    client = _FakeWireClient()
    adapter._client = client  # type: ignore[assignment]
    return adapter, client


@pytest.mark.asyncio
async def test_group_bot_mention_stripped_yields_clean_command() -> None:
    """Fail-stop: a group message leading with the bot's @-mention must forward
    a CLEAN slash command (``/compact``), not ``@AgentM /compact`` — otherwise
    the gateway never classifies it as a command."""
    adapter, client = _make_mention_adapter()
    await adapter._on_message(
        _InboundMsg(
            content_text="@AgentM /compact",
            mentions=[_Mention(key="@_user_1", open_id=_BOT_OID, name="AgentM")],
        )
    )
    assert _last_inbound(client)["content"] == "/compact"


@pytest.mark.asyncio
async def test_group_bot_mention_stripped_from_normal_prompt() -> None:
    """A normal mentioned message forwards the bare text (no mention noise)."""
    adapter, client = _make_mention_adapter()
    await adapter._on_message(
        _InboundMsg(
            content_text="@AgentM hello there",
            mentions=[_Mention(key="@_user_1", open_id=_BOT_OID, name="AgentM")],
        )
    )
    assert _last_inbound(client)["content"] == "hello there"


@pytest.mark.asyncio
async def test_group_unnamed_bot_placeholder_stripped() -> None:
    """When Lark leaves the raw ``@_user_N`` placeholder (no display name), the
    bot's leading placeholder is still stripped to a clean command."""
    adapter, client = _make_mention_adapter()
    await adapter._on_message(
        _InboundMsg(
            content_text="@_user_1 /help",
            mentions=[_Mention(key="@_user_1", open_id=_BOT_OID, name=None)],
        )
    )
    assert _last_inbound(client)["content"] == "/help"


@pytest.mark.asyncio
async def test_other_user_mention_not_stripped() -> None:
    """A leading mention of ANOTHER user (not the bot) is left untouched."""
    adapter, client = _make_mention_adapter()
    await adapter._on_message(
        _InboundMsg(
            content_text="@Alice please run /compact",
            mentions=[_Mention(key="@_user_1", open_id="ou_alice", name="Alice")],
        )
    )
    assert _last_inbound(client)["content"] == "@Alice please run /compact"


@pytest.mark.asyncio
async def test_p2p_no_mention_passes_through_unchanged() -> None:
    """p2p (direct) chats carry no mention; content must pass through exactly."""
    adapter, client = _make_mention_adapter()
    await adapter._on_message(_InboundMsg(content_text="/compact", mentions=[]))
    assert _last_inbound(client)["content"] == "/compact"


@pytest.mark.asyncio
async def test_approval_is_a_standalone_card(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, channel = _make_adapter(monkeypatch)
    env = _outbound(
        "oc_1",
        "approval_request",
        content="Run rm -rf?",
        buttons=[{"label": "Approve", "value": "a:approve", "style": "primary"}],
    )
    await adapter.handle_outbound(env)
    assert len(channel.sends) == 1
    card = channel.sends[0][1]["card"]
    tags = [el["tag"] for el in card["body"]["elements"]]
    assert "button" in tags  # schema-2.0 standalone button element
    assert adapter._live == {}  # approval does not open a live turn


# -- thread_id / topic-group support ------------------------------------


@pytest.mark.asyncio
async def test_inbound_thread_id_forwarded_to_gateway() -> None:
    """When an inbound message carries a thread_id (topic group), it must
    appear in the wire envelope body AND influence the session_key so
    different topics get separate sessions."""
    adapter, client = _make_mention_adapter()
    await adapter._on_message(
        _InboundMsg(
            content_text="hello", chat_id="oc_grp", thread_id="om_root_42"
        )
    )
    body = _last_inbound(client)
    assert body["thread_id"] == "om_root_42"
    # session_key must contain the thread_id
    env = client.sent[-1]
    assert "om_root_42" in env.session_key


@pytest.mark.asyncio
async def test_inbound_no_thread_id_omits_field() -> None:
    """A non-topic-group message must not carry thread_id at all."""
    adapter, client = _make_mention_adapter()
    await adapter._on_message(_InboundMsg(content_text="hi", chat_id="oc_p2p"))
    body = _last_inbound(client)
    assert "thread_id" not in body


@pytest.mark.asyncio
async def test_different_threads_get_separate_live_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent threads in the same chat must produce two independent
    live cards (separate _live entries)."""
    adapter, channel = _make_adapter(monkeypatch)
    chat = "oc_grp"
    await adapter.handle_outbound(
        _outbound(chat, "turn_start", meta={"turn_id": "t1"}, thread_id="om_t1")
    )
    await adapter.handle_outbound(
        _outbound(chat, "turn_start", meta={"turn_id": "t2"}, thread_id="om_t2")
    )
    # Two distinct cards sent (one per thread).
    assert len(channel.sends) == 2
    # Two entries in _live under different keys.
    assert len(adapter._live) == 2


@pytest.mark.asyncio
async def test_outbound_live_card_passes_reply_to_for_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first card send for a threaded turn must pass opts with reply_to
    so Lark places it in the correct topic thread."""
    monkeypatch.setattr(adapter_mod, "_MIN_UPDATE_INTERVAL", 0.0)
    monkeypatch.setattr(adapter_mod, "_AGENT_END_GRACE", 0.02)
    cfg = FeishuConfig(app_id="x", app_secret="y", channel_name="feishu")
    adapter = FeishuAdapter(client=object(), config=cfg)  # type: ignore[arg-type]

    # Use a channel that records opts
    class _OptsChannel:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any], Any]] = []
            self._n = 0

        async def send(
            self, to: str, message: dict[str, Any], opts: Any = None
        ) -> _SendResult:
            self.calls.append((to, message, opts))
            self._n += 1
            return _SendResult(f"om_msg-{self._n}")

        async def update_card(
            self, message_id: str, card: dict[str, Any]
        ) -> _SendResult:
            return _SendResult(message_id)

    ch = _OptsChannel()
    adapter._channel = ch

    await adapter.handle_outbound(
        _outbound("oc_grp", "turn_start", meta={"turn_id": "t1"}, thread_id="om_root")
    )
    assert len(ch.calls) == 1
    _, _, opts = ch.calls[0]
    assert opts is not None
    assert opts["reply_to"] == "om_root"


@pytest.mark.asyncio
async def test_outbound_no_thread_sends_no_opts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-threaded outbound must pass opts=None (no reply_to)."""
    monkeypatch.setattr(adapter_mod, "_MIN_UPDATE_INTERVAL", 0.0)
    monkeypatch.setattr(adapter_mod, "_AGENT_END_GRACE", 0.02)
    cfg = FeishuConfig(app_id="x", app_secret="y", channel_name="feishu")
    adapter = FeishuAdapter(client=object(), config=cfg)  # type: ignore[arg-type]

    class _OptsChannel:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any], Any]] = []
            self._n = 0

        async def send(
            self, to: str, message: dict[str, Any], opts: Any = None
        ) -> _SendResult:
            self.calls.append((to, message, opts))
            self._n += 1
            return _SendResult(f"om_msg-{self._n}")

        async def update_card(
            self, message_id: str, card: dict[str, Any]
        ) -> _SendResult:
            return _SendResult(message_id)

    ch = _OptsChannel()
    adapter._channel = ch

    await adapter.handle_outbound(
        _outbound("oc_grp", "turn_start", meta={"turn_id": "t1"})
    )
    assert len(ch.calls) == 1
    _, _, opts = ch.calls[0]
    assert opts is None
