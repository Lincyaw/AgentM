"""Tests for kernel→Anthropic message and tool serialization.

Each test targets a contract that, if broken, would either confuse the API
(wrong shape) or lose data the model needs to continue (e.g. tool result
packing, signature preservation).
"""

from __future__ import annotations

import base64

from agentm.core.kernel.messages import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
)
from agentm.core.kernel.tool import FunctionTool, ToolResult
from agentm.llm.anthropic import _to_anthropic_messages, _to_anthropic_tools


def _user(text: str) -> UserMessage:
    return UserMessage(
        role="user",
        content=[TextContent(type="text", text=text)],
        timestamp=0.0,
    )


def test_user_text_message_roundtrip() -> None:
    out = _to_anthropic_messages([_user("hello")])
    assert out == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


def test_assistant_mixed_text_and_tool_call() -> None:
    msg = AssistantMessage(
        role="assistant",
        content=[
            TextContent(type="text", text="thinking out loud"),
            ToolCallBlock(
                type="tool_call", id="t1", name="echo", arguments={"x": 1}
            ),
        ],
        timestamp=0.0,
        stop_reason="tool_use",
    )
    out = _to_anthropic_messages([msg])
    assert out == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking out loud"},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "echo",
                    "input": {"x": 1},
                },
            ],
        }
    ]


def test_assistant_thinking_block_preserves_signature() -> None:
    msg = AssistantMessage(
        role="assistant",
        content=[
            ThinkingBlock(type="thinking", text="reason", signature="sig-xyz"),
            TextContent(type="text", text="answer"),
        ],
        timestamp=0.0,
    )
    out = _to_anthropic_messages([msg])
    blocks = out[0]["content"]
    assert blocks[0] == {
        "type": "thinking",
        "thinking": "reason",
        "signature": "sig-xyz",
    }
    # Without signature, the field should be omitted.
    msg2 = AssistantMessage(
        role="assistant",
        content=[ThinkingBlock(type="thinking", text="r", signature=None)],
        timestamp=0.0,
    )
    out2 = _to_anthropic_messages([msg2])
    assert out2[0]["content"][0] == {"type": "thinking", "thinking": "r"}


def test_tool_result_becomes_user_role_message() -> None:
    tr = ToolResultMessage(
        role="tool_result",
        content=[
            ToolResultBlock(
                type="tool_result",
                tool_call_id="t1",
                content=[TextContent(type="text", text="ok")],
                is_error=False,
            )
        ],
        timestamp=0.0,
    )
    out = _to_anthropic_messages([tr])
    assert out == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "ok"}],
                    "is_error": False,
                }
            ],
        }
    ]


def test_adjacent_tool_results_are_packed_into_one_user_message() -> None:
    def make(tr_id: str, text: str) -> ToolResultMessage:
        return ToolResultMessage(
            role="tool_result",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_call_id=tr_id,
                    content=[TextContent(type="text", text=text)],
                )
            ],
            timestamp=0.0,
        )

    out = _to_anthropic_messages([make("a", "1"), make("b", "2")])
    assert len(out) == 1
    assert out[0]["role"] == "user"
    ids = [c["tool_use_id"] for c in out[0]["content"]]
    assert ids == ["a", "b"]


def test_user_message_then_tool_results_stay_separate() -> None:
    # A user-text message followed by a tool_result should NOT swallow the
    # results into the user message — they belong to a different turn.
    user = _user("hi")
    tr = ToolResultMessage(
        role="tool_result",
        content=[
            ToolResultBlock(
                type="tool_result",
                tool_call_id="x",
                content=[TextContent(type="text", text="r")],
            )
        ],
        timestamp=0.0,
    )
    out = _to_anthropic_messages([user, tr])
    assert len(out) == 2
    assert out[0]["content"][0]["type"] == "text"
    assert out[1]["content"][0]["type"] == "tool_result"


def test_image_content_is_base64_encoded() -> None:
    raw = b"\x89PNG\r\nfoo"
    msg = UserMessage(
        role="user",
        content=[ImageContent(type="image", data=raw, mime_type="image/png")],
        timestamp=0.0,
    )
    out = _to_anthropic_messages([msg])
    block = out[0]["content"][0]
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    assert block["source"]["data"] == base64.b64encode(raw).decode("ascii")


def test_tool_conversion_shape() -> None:
    async def _fn(args: dict[str, object]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="x")])

    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    tool = FunctionTool(
        name="echo",
        description="Echoes the input.",
        parameters=schema,
        fn=_fn,
    )
    out = _to_anthropic_tools([tool])
    assert out == [
        {
            "name": "echo",
            "description": "Echoes the input.",
            "input_schema": schema,
        }
    ]
