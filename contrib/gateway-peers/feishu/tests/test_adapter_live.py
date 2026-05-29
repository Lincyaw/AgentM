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
from agentm.gateway.wire import KIND_OUTBOUND, WIRE_VERSION, Envelope


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

    async def send(self, to: str, message: dict[str, Any]) -> _SendResult:
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
async def test_noise_kinds_emit_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, channel = _make_adapter(monkeypatch)
    for kind in ("usage", "child_start", "extension_install", "command_dispatched",
                 "api_register", "resource_write"):
        await adapter.handle_outbound(_outbound("oc_1", kind))
    assert channel.sends == []
    assert channel.updates == []


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
    assert "action" in tags  # carries the interactive buttons
    assert adapter._live == {}  # approval does not open a live turn
