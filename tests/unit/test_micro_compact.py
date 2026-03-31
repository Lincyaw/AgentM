"""Tests for MicroCompactMiddleware."""
from __future__ import annotations

import pytest

from agentm.harness.micro_compact import MicroCompactConfig, MicroCompactMiddleware
from agentm.harness.types import LoopContext, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(step: int = 0) -> LoopContext:
    """Build a minimal LoopContext."""
    return LoopContext(
        agent_id="test",
        step=step,
        max_steps=30,
        tool_call_count=0,
        metadata={},
    )


def _system(content: str) -> Message:
    return {"role": "system", "content": content}


def _human(content: str) -> Message:
    return {"role": "human", "content": content}


def _assistant(
    content: str = "",
    tool_calls: list[dict[str, str]] | None = None,
) -> Message:
    msg: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg  # type: ignore[return-value]


def _tool_result(tool_call_id: str, content: str) -> Message:
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


def _tc(name: str, tc_id: str) -> dict[str, str]:
    """Shorthand for a tool_call entry on an assistant message."""
    return {"name": name, "id": tc_id}


def _build_conversation(
    *,
    num_turns: int,
    tool_name: str = "duckdb_sql",
    tool_result_content: str = "rows=[...]",
) -> list[Message]:
    """Build a realistic conversation with the given number of assistant turns.

    Structure per turn: assistant (with tool_call) -> tool result -> human.
    Prepends a system message.
    """
    msgs: list[Message] = [_system("You are a helpful agent.")]
    for i in range(num_turns):
        tc_id = f"tc_{i}"
        msgs.append(_assistant(tool_calls=[_tc(tool_name, tc_id)]))
        msgs.append(_tool_result(tc_id, tool_result_content))
        msgs.append(_human(f"Round {i} feedback"))
    return msgs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_not_enough_history_returns_unchanged() -> None:
    """When ctx.step < stale_after_steps * 2, messages are returned unchanged."""
    config = MicroCompactConfig(stale_after_steps=6)
    mw = MicroCompactMiddleware(config)
    msgs = _build_conversation(num_turns=5)

    # step=11 < 6*2=12 → no changes
    result = await mw.on_llm_start(msgs, _ctx(step=11))
    assert result == msgs


@pytest.mark.asyncio
async def test_disabled_config_returns_unchanged() -> None:
    """When enabled=False, messages are returned unchanged regardless of age."""
    config = MicroCompactConfig(enabled=False, stale_after_steps=1)
    mw = MicroCompactMiddleware(config)
    msgs = _build_conversation(num_turns=10)

    result = await mw.on_llm_start(msgs, _ctx(step=100))
    assert result == msgs


@pytest.mark.asyncio
async def test_old_compactable_tool_result_cleared() -> None:
    """A tool result from a compactable tool older than stale_after_steps
    turns should have its content replaced with the cleared_message."""
    config = MicroCompactConfig(stale_after_steps=3)
    mw = MicroCompactMiddleware(config)

    # Build 8 turns with duckdb_sql — old ones should be cleared
    msgs = _build_conversation(num_turns=8, tool_name="duckdb_sql")
    result = await mw.on_llm_start(msgs, _ctx(step=20))

    cleared = config.cleared_message
    # The earliest tool results should be cleared
    tool_results = [m for m in result if isinstance(m, dict) and m.get("role") == "tool"]
    assert any(tr["content"] == cleared for tr in tool_results), (
        "Expected at least one old tool result to be cleared"
    )
    # The most recent tool results (within stale_after_steps turns) should NOT be cleared
    # Last tool result is turn 1 from end, should survive
    assert tool_results[-1]["content"] != cleared


@pytest.mark.asyncio
async def test_non_compactable_tool_not_cleared() -> None:
    """Tool results from tools NOT in compactable_tools are never cleared."""
    config = MicroCompactConfig(stale_after_steps=2)
    mw = MicroCompactMiddleware(config)

    msgs = _build_conversation(
        num_turns=8, tool_name="think", tool_result_content="I think..."
    )
    result = await mw.on_llm_start(msgs, _ctx(step=20))

    # No tool result should be cleared — "think" is not compactable
    tool_results = [m for m in result if isinstance(m, dict) and m.get("role") == "tool"]
    for tr in tool_results:
        assert tr["content"] == "I think..."


@pytest.mark.asyncio
async def test_recent_compactable_tool_not_cleared() -> None:
    """A compactable tool result that is recent (within stale_after_steps turns)
    should NOT be cleared."""
    config = MicroCompactConfig(stale_after_steps=10)
    mw = MicroCompactMiddleware(config)

    # Only 5 turns — all are within the stale threshold of 10
    # But we need step >= stale_after_steps * 2 = 20 to even run
    msgs = _build_conversation(num_turns=5, tool_name="duckdb_sql")
    result = await mw.on_llm_start(msgs, _ctx(step=20))

    # All tool results should survive — only 5 turns, threshold is 10
    tool_results = [m for m in result if isinstance(m, dict) and m.get("role") == "tool"]
    for tr in tool_results:
        assert tr["content"] != config.cleared_message


@pytest.mark.asyncio
async def test_already_cleared_not_double_cleared() -> None:
    """A tool result that already has the cleared_message as content should
    not be modified again (idempotent)."""
    config = MicroCompactConfig(stale_after_steps=2)
    mw = MicroCompactMiddleware(config)

    # Build messages manually with a pre-cleared result
    msgs: list[Message] = [
        _system("System prompt"),
        _assistant(tool_calls=[_tc("duckdb_sql", "tc_0")]),
        _tool_result("tc_0", config.cleared_message),  # already cleared
        _human("Round 0"),
        # 6 more turns to push tc_0 well past stale threshold
        *[m for i in range(1, 7)
          for m in (
              _assistant(tool_calls=[_tc("duckdb_sql", f"tc_{i}")]),
              _tool_result(f"tc_{i}", f"data_{i}"),
              _human(f"Round {i}"),
          )],
    ]

    result = await mw.on_llm_start(msgs, _ctx(step=20))

    # The already-cleared result should still be the cleared_message (unchanged)
    old_result = [m for m in result if isinstance(m, dict) and m.get("tool_call_id") == "tc_0"]
    assert len(old_result) == 1
    assert old_result[0]["content"] == config.cleared_message


@pytest.mark.asyncio
async def test_non_tool_messages_never_touched() -> None:
    """System, assistant, and human messages should never be modified."""
    config = MicroCompactConfig(stale_after_steps=2)
    mw = MicroCompactMiddleware(config)

    msgs = _build_conversation(num_turns=8, tool_name="duckdb_sql")
    result = await mw.on_llm_start(msgs, _ctx(step=20))

    # Check system, assistant, human messages are identical
    for orig, new in zip(msgs, result):
        if isinstance(orig, dict) and orig.get("role") != "tool":
            assert orig is new, f"Non-tool message was replaced: {orig}"


@pytest.mark.asyncio
async def test_input_messages_not_mutated() -> None:
    """The original message list and its elements must not be mutated."""
    config = MicroCompactConfig(stale_after_steps=2)
    mw = MicroCompactMiddleware(config)

    msgs = _build_conversation(num_turns=8, tool_name="duckdb_sql")
    # Save original contents for comparison
    original_contents = [
        (m.get("content") if isinstance(m, dict) else None) for m in msgs
    ]
    original_list_len = len(msgs)

    result = await mw.on_llm_start(msgs, _ctx(step=20))

    # The original list should not be mutated
    assert len(msgs) == original_list_len
    assert result is not msgs
    for i, msg in enumerate(msgs):
        if isinstance(msg, dict):
            assert msg.get("content") == original_contents[i], (
                f"Original message at index {i} was mutated"
            )


@pytest.mark.asyncio
async def test_mixed_tools_only_compactable_cleared() -> None:
    """When a conversation has both compactable and non-compactable tools,
    only the compactable ones are cleared."""
    config = MicroCompactConfig(stale_after_steps=2)
    mw = MicroCompactMiddleware(config)

    msgs: list[Message] = [
        _system("System"),
        # Old turns (will be stale)
        _assistant(tool_calls=[_tc("duckdb_sql", "tc_sql")]),
        _tool_result("tc_sql", "sql result data"),
        _human("ok"),
        _assistant(tool_calls=[_tc("think", "tc_think")]),
        _tool_result("tc_think", "thought content"),
        _human("ok"),
        # More recent turns to push the old ones past stale threshold
        *[m for i in range(6)
          for m in (
              _assistant(content=f"analysis {i}"),
              _human(f"feedback {i}"),
          )],
    ]

    result = await mw.on_llm_start(msgs, _ctx(step=20))

    # Find the sql result and think result
    sql_result = next(
        m for m in result if isinstance(m, dict) and m.get("tool_call_id") == "tc_sql"
    )
    think_result = next(
        m for m in result if isinstance(m, dict) and m.get("tool_call_id") == "tc_think"
    )

    assert sql_result["content"] == config.cleared_message
    assert think_result["content"] == "thought content"


@pytest.mark.asyncio
async def test_default_config_used_when_none() -> None:
    """When no config is passed, default MicroCompactConfig is used."""
    mw = MicroCompactMiddleware()
    # Just verify it doesn't crash and uses defaults
    msgs = _build_conversation(num_turns=2)
    result = await mw.on_llm_start(msgs, _ctx(step=0))
    assert result == msgs
