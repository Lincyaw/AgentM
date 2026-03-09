"""Tests for recall_history tool implementation.

Ref: designs/orchestrator.md § recall_history Tool

Bug prevented: recall_history stub returns hardcoded string instead of
searching checkpoint history -> orchestrator cannot recover compressed details
-> investigation quality degrades after compression.
"""

from __future__ import annotations

import inspect
from typing import Literal, get_args, get_origin, get_type_hints
from unittest.mock import MagicMock, patch

import pytest

from agentm.tools.orchestrator import create_orchestrator_tools


def _resolve_annotation(func, param_name: str):
    """Resolve a parameter's type annotation, handling deferred annotations."""
    hints = get_type_hints(func, include_extras=True)
    return hints.get(param_name)


def _extract_literal_values(annotation) -> set[str] | None:
    """Extract Literal values from an annotation."""
    from typing import Annotated

    origin = get_origin(annotation)
    if origin is Literal:
        return set(get_args(annotation))
    if origin is Annotated:
        inner = get_args(annotation)[0]
        if get_origin(inner) is Literal:
            return set(get_args(inner))
    return None


@pytest.fixture
def orch_tools():
    """Create orchestrator tools with mock dependencies."""
    mock_tm = MagicMock()
    mock_pool = MagicMock()
    return create_orchestrator_tools(mock_tm, mock_pool)


class TestRecallHistorySignature:
    """Verify recall_history signature matches design doc.

    Bug: scope parameter is untyped -> LLM passes arbitrary scope values
    -> tool silently ignores invalid scopes -> inconsistent behavior.
    """

    def test_recall_history_in_tools_dict(self, orch_tools):
        assert "recall_history" in orch_tools
        assert callable(orch_tools["recall_history"])

    def test_is_sync(self, orch_tools):
        """recall_history is synchronous (checkpoint reads are sync in LangGraph)."""
        assert not inspect.iscoroutinefunction(orch_tools["recall_history"])

    def test_has_query_parameter(self, orch_tools):
        sig = inspect.signature(orch_tools["recall_history"])
        assert "query" in sig.parameters

    def test_has_scope_parameter(self, orch_tools):
        sig = inspect.signature(orch_tools["recall_history"])
        assert "scope" in sig.parameters

    def test_scope_is_literal_with_correct_values(self, orch_tools):
        annotation = _resolve_annotation(orch_tools["recall_history"], "scope")
        values = _extract_literal_values(annotation)
        assert values is not None, "scope should be a Literal type"
        assert values == {"current_compression", "all_compressions"}

    def test_scope_defaults_to_current_compression(self, orch_tools):
        sig = inspect.signature(orch_tools["recall_history"])
        param = sig.parameters["scope"]
        assert param.default == "current_compression"


class TestSetGraphRef:
    """Verify _set_graph_ref is available for builder wiring.

    Bug: recall_history created outside factory -> no access to graph/config
    -> cannot traverse checkpoint history.
    """

    def test_set_graph_ref_in_tools_dict(self, orch_tools):
        assert "_set_graph_ref" in orch_tools
        assert callable(orch_tools["_set_graph_ref"])


class TestRecallHistoryWithoutGraphRef:
    """When graph reference has not been wired, recall_history must return
    a clear unavailable message instead of crashing.

    Bug: recall_history called before graph wiring -> AttributeError on None
    -> unhandled exception crashes the agent.
    """

    def test_returns_unavailable_message(self, orch_tools):
        result = orch_tools["recall_history"]("What was the CPU?")
        assert "not available" in result.lower()


class TestRecallHistoryNoCompression:
    """When graph is wired but no compression has occurred, recall_history
    should tell the agent that full history is still in context.

    Bug: recall_history traverses empty checkpoint history -> returns
    confusing empty result -> agent keeps retrying recall.
    """

    def test_no_compression_refs_returns_helpful_message(self, orch_tools):
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {"compression_refs": []}
        mock_graph.get_state.return_value = mock_state

        orch_tools["_set_graph_ref"](
            mock_graph, {"configurable": {"thread_id": "test"}}
        )
        result = orch_tools["recall_history"]("What was the CPU?")
        assert "no compression" in result.lower()


class TestRecallHistoryWithCompression:
    """When compression refs exist, recall_history must traverse checkpoint
    history and use LLM to extract relevant information.

    Bug: recall_history ignores checkpoint messages -> returns summary-level
    info instead of raw data values -> defeats purpose of recall.
    """

    @patch("agentm.tools.orchestrator.ChatOpenAI", create=True)
    def test_traverses_history_and_calls_llm(self, mock_llm_class, orch_tools):
        from langchain_core.messages import AIMessage, HumanMessage

        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="CPU was 85% — postgres 45%, java 12%"
        )
        mock_llm_class.return_value = mock_llm

        # Set up mock graph with compression refs and history
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {
            "compression_refs": [
                {"from_checkpoint_id": "s0", "to_checkpoint_id": "s5"}
            ]
        }
        mock_graph.get_state.return_value = mock_state

        # Mock checkpoint history with messages
        mock_snapshot = MagicMock()
        mock_snapshot.values = {
            "messages": [
                HumanMessage(content="Check CPU usage"),
                AIMessage(content="CPU is 85%, postgres 45%, java 12%"),
            ]
        }
        mock_graph.get_state_history.return_value = [mock_snapshot]

        orch_tools["_set_graph_ref"](
            mock_graph, {"configurable": {"thread_id": "test"}}
        )

        # Patch the lazy import inside recall_history
        with patch(
            "langchain_openai.ChatOpenAI", mock_llm_class
        ):
            result = orch_tools["recall_history"]("What was the CPU breakdown?")

        assert "85%" in result
        mock_llm.invoke.assert_called_once()

    def test_empty_history_returns_no_messages(self, orch_tools):
        """When checkpoint history exists but has no messages."""
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {
            "compression_refs": [
                {"from_checkpoint_id": "s0", "to_checkpoint_id": "s5"}
            ]
        }
        mock_graph.get_state.return_value = mock_state

        mock_snapshot = MagicMock()
        mock_snapshot.values = {"messages": []}
        mock_graph.get_state_history.return_value = [mock_snapshot]

        orch_tools["_set_graph_ref"](
            mock_graph, {"configurable": {"thread_id": "test"}}
        )
        result = orch_tools["recall_history"]("What was the CPU?")
        assert "no messages" in result.lower()

    def test_checkpoint_traversal_error_returns_error_message(self, orch_tools):
        """When checkpoint history access raises, return error string."""
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {
            "compression_refs": [
                {"from_checkpoint_id": "s0", "to_checkpoint_id": "s5"}
            ]
        }
        mock_graph.get_state.return_value = mock_state
        mock_graph.get_state_history.side_effect = RuntimeError("DB connection lost")

        orch_tools["_set_graph_ref"](
            mock_graph, {"configurable": {"thread_id": "test"}}
        )
        result = orch_tools["recall_history"]("What was the CPU?")
        assert "error" in result.lower()
        assert "DB connection lost" in result
