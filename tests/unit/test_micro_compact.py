"""Focused tests for MicroCompactMiddleware behavior boundaries."""
from __future__ import annotations

import pytest

from agentm.harness.micro_compact import MicroCompactConfig, MicroCompactMiddleware
from agentm.harness.types import LoopContext, Message


def _ctx(step: int = 0) -> LoopContext:
    return LoopContext(agent_id="test", step=step, max_steps=30, tool_call_count=0, metadata={})


def _system(content: str) -> Message:
    return {"role": "system", "content": content}


def _human(content: str) -> Message:
    return {"role": "human", "content": content}


def _assistant(tool_calls: list[dict[str, str]] | None = None) -> Message:
    msg: dict[str, object] = {"role": "assistant", "content": ""}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg  # type: ignore[return-value]


def _tool_result(tool_call_id: str, content: str) -> Message:
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


def _tc(name: str, tc_id: str) -> dict[str, str]:
    return {"name": name, "id": tc_id}


def _conversation(num_turns: int, tool_name: str = "duckdb_sql", content: str = "rows=[...]") -> list[Message]:
    msgs: list[Message] = [_system("sys")]
    for i in range(num_turns):
        tc_id = f"tc_{i}"
        msgs.append(_assistant(tool_calls=[_tc(tool_name, tc_id)]))
        msgs.append(_tool_result(tc_id, content))
        msgs.append(_human(f"Round {i}"))
    return msgs


@pytest.mark.asyncio
async def test_disabled_or_insufficient_history_returns_unchanged() -> None:
    msgs = _conversation(5)
    disabled = MicroCompactMiddleware(MicroCompactConfig(enabled=False, stale_after_steps=1))
    assert await disabled.on_llm_start(msgs, _ctx(step=100)) == msgs

    not_enough = MicroCompactMiddleware(MicroCompactConfig(stale_after_steps=6))
    assert await not_enough.on_llm_start(msgs, _ctx(step=11)) == msgs


@pytest.mark.asyncio
async def test_old_compactable_tool_results_are_cleared_but_recent_remain() -> None:
    config = MicroCompactConfig(stale_after_steps=3)
    result = await MicroCompactMiddleware(config).on_llm_start(_conversation(8, tool_name="duckdb_sql"), _ctx(step=20))
    tool_results = [m for m in result if isinstance(m, dict) and m.get("role") == "tool"]
    assert any(m["content"] == config.cleared_message for m in tool_results)
    assert tool_results[-1]["content"] != config.cleared_message


@pytest.mark.asyncio
async def test_non_compactable_tools_are_not_cleared() -> None:
    result = await MicroCompactMiddleware(MicroCompactConfig(stale_after_steps=2)).on_llm_start(
        _conversation(8, tool_name="think", content="I think..."), _ctx(step=20)
    )
    for msg in result:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            assert msg["content"] == "I think..."


@pytest.mark.asyncio
async def test_compaction_does_not_mutate_input_messages() -> None:
    msgs = _conversation(8)
    original_contents = [m.get("content") if isinstance(m, dict) else None for m in msgs]
    result = await MicroCompactMiddleware(MicroCompactConfig(stale_after_steps=2)).on_llm_start(msgs, _ctx(step=20))
    assert result is not msgs
    for i, msg in enumerate(msgs):
        if isinstance(msg, dict):
            assert msg.get("content") == original_contents[i]
