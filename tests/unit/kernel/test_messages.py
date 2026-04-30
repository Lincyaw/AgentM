"""Tests for kernel message data model.

Per CLAUDE.md testing philosophy, we test behavior (immutability contract,
helper round-trips), not framework guarantees (field defaults, dataclass
construction).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agentm.core.kernel.messages import (
    TextContent,
    ToolResultBlock,
    ToolResultMessage,
    text_message,
    tool_result,
)


def test_text_content_is_immutable() -> None:
    """The frozen contract is what guarantees safe sharing across handlers."""

    block = TextContent(type="text", text="hello")
    with pytest.raises(FrozenInstanceError):
        block.text = "mutated"  # type: ignore[misc]


def test_text_message_round_trip() -> None:
    """`text_message` produces a UserMessage carrying exactly the text given."""

    msg = text_message("hi there", timestamp=42.0)
    assert msg.role == "user"
    assert msg.timestamp == 42.0
    assert len(msg.content) == 1
    block = msg.content[0]
    assert isinstance(block, TextContent)
    assert block.text == "hi there"


def test_tool_result_helper_builds_tool_result_message() -> None:
    """`tool_result` must wrap a single text block in a ToolResultBlock and
    place it inside a ToolResultMessage with the right role."""

    msg = tool_result("call-123", "the answer is 42", is_error=False)
    assert isinstance(msg, ToolResultMessage)
    assert msg.role == "tool_result"
    assert len(msg.content) == 1
    block = msg.content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.tool_call_id == "call-123"
    assert block.is_error is False
    assert len(block.content) == 1
    inner = block.content[0]
    assert isinstance(inner, TextContent)
    assert inner.text == "the answer is 42"


def test_tool_result_helper_propagates_error_flag() -> None:
    """`is_error=True` must reach the inner ToolResultBlock so the loop's
    error-handling path treats it correctly."""

    msg = tool_result("call-x", "boom", is_error=True)
    block = msg.content[0]
    assert block.is_error is True
