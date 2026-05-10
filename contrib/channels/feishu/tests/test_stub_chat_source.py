from __future__ import annotations

import asyncio

import pytest

from agentm_feishu.chat_source import (
    CardActionEvent,
    InboundMessage,
    SendResult,
    StubChatSource,
)


@pytest.mark.asyncio
async def test_messages_yields_pushed_items_and_terminates_on_close() -> None:
    src = StubChatSource()
    await src.connect()
    src.push_message(InboundMessage(chat_id="c", user_id="u", text="hi"))
    src.push_message(InboundMessage(chat_id="c", user_id="u", text="bye"))

    seen: list[str] = []

    async def consume() -> None:
        async for m in src.messages():
            seen.append(m.text)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await src.close()
    await asyncio.wait_for(task, timeout=1.0)
    assert seen == ["hi", "bye"]


@pytest.mark.asyncio
async def test_send_card_returns_unique_card_ids() -> None:
    src = StubChatSource()
    await src.connect()
    a: SendResult = await src.send_card("c", {"a": 1})
    b: SendResult = await src.send_card("c", {"b": 2})
    assert a.card_id != b.card_id
    assert {entry["kind"] for entry in src.outbox} == {"card"}


@pytest.mark.asyncio
async def test_card_actions_iterator_drains_pushed_clicks() -> None:
    src = StubChatSource()
    await src.connect()
    src.push_card_action(CardActionEvent(card_id="card-0", user_id="u", action="approve"))

    async def consume() -> CardActionEvent:
        async for action in src.card_actions():
            return action
        raise AssertionError("iterator ended without yielding")

    action = await asyncio.wait_for(consume(), timeout=1.0)
    assert action.action == "approve"
    await src.close()
