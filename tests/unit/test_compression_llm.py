"""Tests for LLM-based compression in Sub-Agent hooks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agentm.config.schema import CompressionConfig
from agentm.middleware.compression import (
    _DEFAULT_THRESHOLD_TOKENS,
    _summarize_messages,
    build_compression_hook,
    count_tokens,
    sub_agent_compression_hook,
)


def _make_messages(n: int, content_size: int = 100) -> list:
    """Create n messages with specified content size."""
    msgs = []
    for i in range(n):
        if i % 2 == 0:
            msgs.append(HumanMessage(content=f"Message {i}: " + "x" * content_size))
        else:
            msgs.append(AIMessage(content=f"Response {i}: " + "y" * content_size))
    return msgs


class TestSubAgentCompressionHook:
    def test_below_threshold_passes_through(self) -> None:
        """Messages below threshold pass through unchanged."""
        messages = _make_messages(3, content_size=10)
        result = sub_agent_compression_hook({"messages": messages})
        assert "messages" in result
        assert result["messages"] == messages
        assert "llm_input_messages" not in result

    @patch("agentm.middleware.compression._summarize_messages")
    def test_above_threshold_triggers_compression(
        self, mock_summarize: MagicMock
    ) -> None:
        """Messages above threshold trigger LLM compression."""
        mock_summarize.return_value = "Summary of conversation"
        messages = _make_messages(400, content_size=2000)
        assert count_tokens(messages) >= _DEFAULT_THRESHOLD_TOKENS

        result = sub_agent_compression_hook({"messages": messages})

        assert "llm_input_messages" in result
        assert "messages" not in result
        mock_summarize.assert_called_once()
        assert "[Compressed History Summary]" in result["llm_input_messages"][0].content
        assert result["llm_input_messages"][-2:] == messages[-2:]

    def test_empty_messages(self) -> None:
        """Empty messages pass through."""
        result = sub_agent_compression_hook({"messages": []})
        assert result["messages"] == []

    @patch(
        "agentm.middleware.compression.count_tokens",
        return_value=_DEFAULT_THRESHOLD_TOKENS + 1,
    )
    @patch("agentm.middleware.compression._summarize_messages")
    def test_few_messages_above_threshold_not_compressed(
        self, mock_summarize: MagicMock, mock_count: MagicMock
    ) -> None:
        """When message count <= preserve_n, messages pass through even above threshold."""
        messages = _make_messages(2, content_size=100)

        result = sub_agent_compression_hook({"messages": messages})

        assert "messages" in result
        assert result["messages"] == messages
        mock_summarize.assert_not_called()


class TestBuildCompressionHook:
    @patch("agentm.middleware.compression._summarize_messages")
    def test_respects_preserve_latest_n(self, mock_summarize: MagicMock) -> None:
        """Configurable preserve_latest_n is respected."""
        mock_summarize.return_value = "Summary"
        config = CompressionConfig(
            compression_threshold=0.01,
            compression_model="gpt-4o-mini",
            preserve_latest_n=3,
        )
        hook = build_compression_hook(config)
        messages = _make_messages(10, content_size=100)

        result = hook({"messages": messages})

        if "llm_input_messages" in result:
            assert result["llm_input_messages"][-3:] == messages[-3:]

    def test_below_configured_threshold(self) -> None:
        """Below configured threshold passes through."""
        config = CompressionConfig(
            compression_threshold=0.99,
            compression_model="gpt-4o-mini",
        )
        hook = build_compression_hook(config)
        messages = _make_messages(5, content_size=10)
        result = hook({"messages": messages})
        assert "messages" in result

    @patch("agentm.middleware.compression._summarize_messages")
    def test_few_messages_not_compressed(self, mock_summarize: MagicMock) -> None:
        """When message count <= preserve_n, hook passes through."""
        config = CompressionConfig(
            compression_threshold=0.01,
            compression_model="gpt-4o-mini",
            preserve_latest_n=5,
        )
        hook = build_compression_hook(config)
        messages = _make_messages(5, content_size=100)

        result = hook({"messages": messages})

        assert "messages" in result
        mock_summarize.assert_not_called()


class TestSummarizeMessages:
    @patch("langchain_openai.ChatOpenAI")
    def test_summarize_formats_messages(self, mock_llm_class: MagicMock) -> None:
        """_summarize_messages formats messages and calls LLM."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test summary")
        mock_llm_class.return_value = mock_llm

        messages = [
            HumanMessage(content="What is the CPU usage?"),
            AIMessage(content="CPU is at 85%"),
        ]

        result = _summarize_messages(messages)

        assert result == "Test summary"
        mock_llm.invoke.assert_called_once()

    @patch("langchain_openai.ChatOpenAI")
    def test_summarize_truncates_long_content(self, mock_llm_class: MagicMock) -> None:
        """Long message content is truncated in the summarization prompt."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Truncated summary")
        mock_llm_class.return_value = mock_llm

        messages = [HumanMessage(content="x" * 1000)]

        result = _summarize_messages(messages)

        assert result == "Truncated summary"
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_content = call_args[0].content
        assert "..." in prompt_content

    @patch("langchain_openai.ChatOpenAI")
    def test_summarize_handles_tool_calls(self, mock_llm_class: MagicMock) -> None:
        """Messages with tool_calls are formatted with tool names."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Tool summary")
        mock_llm_class.return_value = mock_llm

        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"name": "get_metrics", "args": {}, "id": "1"}]

        result = _summarize_messages([ai_msg])

        assert result == "Tool summary"
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_content = call_args[0].content
        assert "get_metrics" in prompt_content
