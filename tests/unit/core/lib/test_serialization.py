from __future__ import annotations

from dataclasses import dataclass

from agentm.core.abi import (
    AssistantMessage,
    MessageEnd,
    TextContent,
    TextDelta,
    ToolCallBlock,
    ToolCallStart,
)
from agentm.core.lib import _to_jsonable


@dataclass
class _PlainDataclass:
    value: int


class _DictBacked:
    def __init__(self) -> None:
        self.name = "dict-backed"
        self.child = _PlainDataclass(3)


def test_to_jsonable_serializes_assistant_message_blocks() -> None:
    message = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="hi")],
        timestamp=1.0,
    )

    assert _to_jsonable(message)["content"] == [{"type": "text", "text": "hi"}]


def test_to_jsonable_serializes_tool_call_blocks() -> None:
    block = ToolCallBlock(type="tool_call", id="call-1", name="read", arguments={"path": "x"})

    assert _to_jsonable(block) == {
        "type": "tool_call",
        "id": "call-1",
        "name": "read",
        "arguments": {"path": "x"},
    }


def test_to_jsonable_serializes_stream_deltas() -> None:
    assert _to_jsonable(TextDelta(text="hello")) == {"text": "hello"}
    assert _to_jsonable(ToolCallStart(id="call-1", name="bash")) == {
        "id": "call-1",
        "name": "bash",
    }


def test_to_jsonable_serializes_message_end_and_dict_backed_objects() -> None:
    event = MessageEnd(
        message=AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="done")],
            timestamp=2.0,
        )
    )

    assert _to_jsonable(event)["message"]["role"] == "assistant"
    assert _to_jsonable(_DictBacked()) == {
        "name": "dict-backed",
        "child": {"value": 3},
    }
