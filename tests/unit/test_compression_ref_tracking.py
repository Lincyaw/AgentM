"""Tests for compression event tracking via contextvars.

Bug prevented: Compression fires but no record is kept, so recall_history
has no way to know that context was lost and cannot retrieve original data.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agentm.config.schema import CompressionConfig
from agentm.middleware.compression import (
    _DEFAULT_THRESHOLD_TOKENS,
    build_compression_hook,
    clear_compression_events,
    get_compression_events,
    record_compression_event,
    sub_agent_compression_hook,
)


class TestCompressionEventAPI:
    """Low-level API: record, get, clear compression events."""

    def setup_method(self) -> None:
        clear_compression_events()

    def test_no_events_initially(self) -> None:
        assert get_compression_events() == []

    def test_record_and_retrieve(self) -> None:
        record_compression_event({"layer": "sub_agent", "reason": "test"})
        events = get_compression_events()
        assert len(events) == 1
        assert events[0]["layer"] == "sub_agent"
        assert events[0]["reason"] == "test"

    def test_multiple_events_accumulate(self) -> None:
        record_compression_event({"layer": "sub_agent", "step_count": 10})
        record_compression_event({"layer": "sub_agent", "step_count": 20})
        assert len(get_compression_events()) == 2

    def test_clear_removes_all(self) -> None:
        record_compression_event({"layer": "sub_agent", "reason": "a"})
        record_compression_event({"layer": "sub_agent", "reason": "b"})
        clear_compression_events()
        assert get_compression_events() == []

    def test_get_returns_copy(self) -> None:
        """Mutating the returned list must not affect internal state."""
        record_compression_event({"layer": "sub_agent"})
        events = get_compression_events()
        events.clear()
        assert len(get_compression_events()) == 1


class TestSubAgentHookRecordsEvent:
    """sub_agent_compression_hook records an event when compression fires."""

    def setup_method(self) -> None:
        clear_compression_events()

    @patch("agentm.middleware.compression._summarize_messages", return_value="Summary")
    @patch("agentm.middleware.compression.count_tokens", return_value=200_000)
    def test_records_event_on_compression(
        self, _mock_count: MagicMock, _mock_summarize: MagicMock
    ) -> None:
        messages = [
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
            HumanMessage(content="msg3"),
            AIMessage(content="msg4"),
        ]

        result = sub_agent_compression_hook({"messages": messages})

        assert "llm_input_messages" in result
        events = get_compression_events()
        assert len(events) == 1
        assert events[0]["layer"] == "sub_agent"
        assert events[0]["step_count"] == 2  # 4 messages - 2 preserved = 2 compressed
        assert "200000" in events[0]["reason"]

    def test_no_event_below_threshold(self) -> None:
        messages = [HumanMessage(content="short")]
        sub_agent_compression_hook({"messages": messages})
        assert get_compression_events() == []

    @patch(
        "agentm.middleware.compression.count_tokens",
        return_value=_DEFAULT_THRESHOLD_TOKENS + 1,
    )
    def test_no_event_when_few_messages(self, _mock_count: MagicMock) -> None:
        """When message count <= preserve_n, no compression and no event."""
        messages = [HumanMessage(content="a"), AIMessage(content="b")]
        sub_agent_compression_hook({"messages": messages})
        assert get_compression_events() == []


class TestBuildCompressionHookRecordsEvent:
    """build_compression_hook's returned hook records an event on compression."""

    def setup_method(self) -> None:
        clear_compression_events()

    @patch("agentm.middleware.compression._summarize_messages", return_value="Summary")
    @patch("agentm.middleware.compression.count_tokens", return_value=200_000)
    def test_configurable_hook_records_event(
        self, _mock_count: MagicMock, _mock_summarize: MagicMock
    ) -> None:
        config = CompressionConfig(
            compression_threshold=0.01,
            compression_model="gpt-4o-mini",
            preserve_latest_n=2,
        )
        hook = build_compression_hook(config)
        messages = [
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
            HumanMessage(content="msg3"),
            AIMessage(content="msg4"),
        ]

        result = hook({"messages": messages})

        assert "llm_input_messages" in result
        events = get_compression_events()
        assert len(events) == 1
        assert events[0]["layer"] == "sub_agent"
        assert events[0]["step_count"] == 2

    def test_configurable_hook_no_event_below_threshold(self) -> None:
        config = CompressionConfig(
            compression_threshold=0.99,
            compression_model="gpt-4o-mini",
        )
        hook = build_compression_hook(config)
        hook({"messages": [HumanMessage(content="short")]})
        assert get_compression_events() == []
