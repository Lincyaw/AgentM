"""Tests for harness middleware: LoopDetection, PreCompletion, ToolOutputOffload."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentm.middleware.loop_detection import (
    LoopDetectionMiddleware,
    build_loop_detection_hook,
)
from agentm.middleware.pre_completion import (
    PreCompletionChecklistMiddleware,
    build_pre_completion_hook,
)
from agentm.middleware.tool_output import (
    ToolOutputOffloadMiddleware,
    build_tool_output_offload_hook,
)


# =========================================================================
# Helpers
# =========================================================================


def _ai_with_tool_calls(calls: list[dict[str, object]]) -> AIMessage:
    """Create an AIMessage carrying tool_calls."""
    return AIMessage(content="", tool_calls=calls)  # type: ignore[arg-type]


def _ai_no_tools(content: str = "Done.") -> AIMessage:
    """Create an AIMessage with no tool_calls (agent finishing)."""
    return AIMessage(content=content)


def _tool_msg(content: str, tool_call_id: str = "tc1") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id)


# =========================================================================
# LoopDetectionMiddleware
# =========================================================================


class TestLoopDetectionMiddleware:
    """Verify doom-loop detection and warning injection."""

    def test_no_warning_below_threshold(self) -> None:
        """When call count < threshold, no warning is injected."""
        mw = LoopDetectionMiddleware(threshold=3, window_size=20)
        msgs = [
            _ai_with_tool_calls(
                [{"name": "read_file", "args": {"path": "/a.py"}, "id": f"tc{i}"}]
            )
            for i in range(2)
        ]
        result = mw.before_model({"messages": msgs})
        out_msgs = result["messages"]
        # No HumanMessage warning appended
        assert all(not isinstance(m, HumanMessage) for m in out_msgs)

    def test_warning_injected_at_threshold(self) -> None:
        """When call count >= threshold, a warning HumanMessage is appended."""
        mw = LoopDetectionMiddleware(threshold=3, window_size=20)
        msgs = [
            _ai_with_tool_calls(
                [{"name": "read_file", "args": {"path": "/a.py"}, "id": f"tc{i}"}]
            )
            for i in range(4)
        ]
        result = mw.before_model({"messages": msgs})
        out_msgs = result["messages"]
        warnings = [m for m in out_msgs if isinstance(m, HumanMessage)]
        assert len(warnings) == 1
        assert "LOOP DETECTION WARNING" in warnings[0].content
        assert "read_file" in warnings[0].content

    def test_window_size_limits_scope(self) -> None:
        """Only the last `window_size` AI messages are inspected."""
        mw = LoopDetectionMiddleware(threshold=3, window_size=2)
        # 4 identical calls, but window only sees the last 2
        msgs = [
            _ai_with_tool_calls(
                [{"name": "read_file", "args": {"path": "/a.py"}, "id": f"tc{i}"}]
            )
            for i in range(4)
        ]
        result = mw.before_model({"messages": msgs})
        out_msgs = result["messages"]
        # Only 2 within window -- below threshold of 3
        assert all(not isinstance(m, HumanMessage) for m in out_msgs)

    def test_hook_factory_returns_callable(self) -> None:
        """build_loop_detection_hook returns a working hook."""
        hook = build_loop_detection_hook(threshold=2, window_size=10)
        msgs = [
            _ai_with_tool_calls(
                [{"name": "grep", "args": {"q": "foo"}, "id": f"tc{i}"}]
            )
            for i in range(3)
        ]
        result = hook({"messages": msgs})
        warnings = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        assert len(warnings) == 1

    def test_different_args_not_counted_together(self) -> None:
        """Calls with different args are counted separately."""
        mw = LoopDetectionMiddleware(threshold=2, window_size=20)
        msgs = [
            _ai_with_tool_calls(
                [
                    {
                        "name": "read_file",
                        "args": {"path": f"/file{i}.py"},
                        "id": f"tc{i}",
                    }
                ]
            )
            for i in range(5)
        ]
        result = mw.before_model({"messages": msgs})
        out_msgs = result["messages"]
        # All args are different -- no loop
        assert all(not isinstance(m, HumanMessage) for m in out_msgs)

    def test_respects_llm_input_messages_key(self) -> None:
        """When state has llm_input_messages, hook uses and returns that key."""
        mw = LoopDetectionMiddleware(threshold=2, window_size=20)
        msgs = [
            _ai_with_tool_calls([{"name": "tool_x", "args": {}, "id": f"tc{i}"}])
            for i in range(3)
        ]
        result = mw.before_model({"llm_input_messages": msgs})
        assert "llm_input_messages" in result


# =========================================================================
# PreCompletionChecklistMiddleware
# =========================================================================


class TestPreCompletionChecklistMiddleware:
    """Verify pre-completion verification reminder."""

    def test_no_trigger_when_ai_has_tool_calls(self) -> None:
        """If last AI message has tool_calls, no reminder is injected."""
        mw = PreCompletionChecklistMiddleware()
        msgs = [
            HumanMessage(content="do stuff"),
            _ai_with_tool_calls([{"name": "grep", "args": {"q": "x"}, "id": "tc1"}]),
        ]
        result = mw.before_model({"messages": msgs})
        out_msgs = result["messages"]
        assert len(out_msgs) == len(msgs)

    def test_triggers_when_ai_has_no_tool_calls(self) -> None:
        """If last AI message has no tool_calls, a reminder is injected once."""
        mw = PreCompletionChecklistMiddleware()
        msgs = [
            HumanMessage(content="do stuff"),
            _ai_no_tools("I'm done."),
        ]
        result = mw.before_model({"messages": msgs})
        out_msgs = result["messages"]
        reminders = [
            m
            for m in out_msgs
            if isinstance(m, HumanMessage) and "verify" in m.content.lower()
        ]
        assert len(reminders) == 1
        assert "tested your solution" in reminders[0].content

    def test_fires_only_once(self) -> None:
        """After triggering once, subsequent calls do NOT inject again."""
        mw = PreCompletionChecklistMiddleware()
        msgs = [
            HumanMessage(content="do stuff"),
            _ai_no_tools("I'm done."),
        ]
        # First call -- triggers
        result1 = mw.before_model({"messages": msgs})
        reminders1 = [
            m
            for m in result1["messages"]
            if isinstance(m, HumanMessage) and "verify" in m.content.lower()
        ]
        assert len(reminders1) == 1

        # Second call -- should NOT trigger again
        result2 = mw.before_model({"messages": msgs})
        reminders2 = [
            m
            for m in result2["messages"]
            if isinstance(m, HumanMessage) and "verify" in m.content.lower()
        ]
        assert len(reminders2) == 0

    def test_custom_checklist(self) -> None:
        """Custom checklist text is used when provided."""
        custom = "Did you run the linter?"
        mw = PreCompletionChecklistMiddleware(checklist=custom)
        msgs = [_ai_no_tools("I'm done.")]
        result = mw.before_model({"messages": msgs})
        out_msgs = result["messages"]
        injected = [m for m in out_msgs if isinstance(m, HumanMessage)]
        assert len(injected) == 1
        assert injected[0].content == custom

    def test_hook_factory_returns_callable(self) -> None:
        """build_pre_completion_hook returns a working hook."""
        hook = build_pre_completion_hook(checklist="Check!")
        msgs = [_ai_no_tools("Done.")]
        result = hook({"messages": msgs})
        injected = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        assert len(injected) == 1
        assert injected[0].content == "Check!"

    def test_no_ai_messages_no_trigger(self) -> None:
        """If there are no AI messages at all, nothing is injected."""
        mw = PreCompletionChecklistMiddleware()
        msgs = [HumanMessage(content="hello")]
        result = mw.before_model({"messages": msgs})
        assert len(result["messages"]) == 1


# =========================================================================
# ToolOutputOffloadMiddleware
# =========================================================================


class TestToolOutputOffloadMiddleware:
    """Verify large tool output truncation."""

    def test_short_content_not_truncated(self) -> None:
        """Content within max_chars is left untouched."""
        mw = ToolOutputOffloadMiddleware(max_chars=100)
        msg = _tool_msg("short output")
        result = mw.before_model({"messages": [msg]})
        out_msgs = result["messages"]
        assert len(out_msgs) == 1
        assert out_msgs[0].content == "short output"

    def test_long_content_truncated(self) -> None:
        """Content exceeding max_chars is replaced with head + tail."""
        mw = ToolOutputOffloadMiddleware(max_chars=100, head_chars=30, tail_chars=20)
        long_content = "A" * 200
        msg = _tool_msg(long_content)
        result = mw.before_model({"messages": [msg]})
        out_msgs = result["messages"]
        truncated = out_msgs[0].content
        assert len(truncated) < len(long_content)
        assert truncated.startswith("A" * 30)
        assert truncated.endswith("A" * 20)

    def test_truncated_message_includes_original_length(self) -> None:
        """The truncation notice includes the original content length."""
        mw = ToolOutputOffloadMiddleware(max_chars=100, head_chars=30, tail_chars=20)
        long_content = "X" * 500
        msg = _tool_msg(long_content)
        result = mw.before_model({"messages": [msg]})
        truncated = result["messages"][0].content
        assert "original length: 500" in truncated
        assert "characters omitted" in truncated

    def test_non_tool_messages_untouched(self) -> None:
        """HumanMessage and AIMessage are never truncated."""
        mw = ToolOutputOffloadMiddleware(max_chars=10)
        msgs = [
            HumanMessage(content="A" * 100),
            _ai_no_tools("B" * 100),
        ]
        result = mw.before_model({"messages": msgs})
        assert result["messages"][0].content == "A" * 100
        assert result["messages"][1].content == "B" * 100

    def test_preserves_tool_call_id(self) -> None:
        """Truncated ToolMessage retains the original tool_call_id."""
        mw = ToolOutputOffloadMiddleware(max_chars=50, head_chars=10, tail_chars=10)
        msg = ToolMessage(
            content="Z" * 200,
            tool_call_id="my-call-42",
        )
        result = mw.before_model({"messages": [msg]})
        out_msg = result["messages"][0]
        assert out_msg.tool_call_id == "my-call-42"

    def test_hook_factory_returns_callable(self) -> None:
        """build_tool_output_offload_hook returns a working hook."""
        hook = build_tool_output_offload_hook(
            max_chars=50, head_chars=10, tail_chars=10
        )
        msg = _tool_msg("Q" * 200)
        result = hook({"messages": [msg]})
        assert "characters omitted" in result["messages"][0].content

    def test_respects_llm_input_messages_key(self) -> None:
        """When state has llm_input_messages, hook uses and returns that key."""
        mw = ToolOutputOffloadMiddleware(max_chars=50, head_chars=10, tail_chars=10)
        msg = _tool_msg("R" * 200)
        result = mw.before_model({"llm_input_messages": [msg]})
        assert "llm_input_messages" in result


# ---------------------------------------------------------------------------
# NodePipeline
# ---------------------------------------------------------------------------


class TestNodePipeline:
    """NodePipeline provides before + after lifecycle for Node-mode graphs."""

    def test_before_chains_middlewares(self) -> None:
        """before() applies all middleware before_model hooks in order."""
        from agentm.middleware import AgentMMiddleware, NodePipeline

        class AddMarker(AgentMMiddleware):
            def __init__(self, marker: str):
                self._marker = marker

            def before_model(self, state: dict) -> dict:
                msgs = list(state.get("messages", []))
                msgs.append(self._marker)
                return {"messages": msgs}

        pipeline = NodePipeline([AddMarker("A"), AddMarker("B")])
        result = pipeline.before({"messages": []})
        assert result["messages"] == ["A", "B"]

    def test_after_calls_aafter_model(self) -> None:
        """after() invokes aafter_model on all middleware with response merged."""
        import asyncio
        from agentm.middleware import AgentMMiddleware, NodePipeline

        captured: list[dict] = []

        class Recorder(AgentMMiddleware):
            async def aafter_model(self, state: dict, runtime=None):
                captured.append(state)
                return None

        pipeline = NodePipeline([Recorder()])
        fake_response = AIMessage(content="hello")
        asyncio.run(pipeline.after({"messages": []}, fake_response))
        assert len(captured) == 1
        assert captured[0]["response"] is fake_response

    def test_empty_pipeline_passthrough(self) -> None:
        """Empty pipeline returns state unchanged."""
        from agentm.middleware import NodePipeline

        pipeline = NodePipeline([])
        state = {"messages": ["test"]}
        result = pipeline.before(state)
        assert result["messages"] == ["test"]

    def test_before_preserves_llm_input_messages(self) -> None:
        """Pipeline propagates llm_input_messages through the chain."""
        from agentm.middleware import AgentMMiddleware, NodePipeline

        pipeline = NodePipeline([AgentMMiddleware()])
        state = {"llm_input_messages": ["rewritten"]}
        result = pipeline.before(state)
        assert result["llm_input_messages"] == ["rewritten"]


# ---------------------------------------------------------------------------
# TrajectoryMiddleware.aafter_model
# ---------------------------------------------------------------------------


class TestTrajectoryMiddlewareAfterModel:
    """TrajectoryMiddleware.aafter_model records tool_call and llm_end events."""

    def _make_middleware(self) -> tuple:
        """Create a TrajectoryMiddleware with a mock collector."""
        from unittest.mock import MagicMock
        from agentm.middleware.trajectory import TrajectoryMiddleware

        collector = MagicMock()
        mw = TrajectoryMiddleware(collector, ["orchestrator", "w1"], task_id="t1")
        return mw, collector

    def test_records_tool_calls(self) -> None:
        """aafter_model emits tool_call events for AI messages with tool_calls."""
        import asyncio

        mw, collector = self._make_middleware()
        response = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc1", "name": "query_logs", "args": {"service": "api"}},
            ],
        )
        asyncio.run(mw.aafter_model({"messages": [], "response": response}))
        collector.record_sync.assert_called_with(
            event_type="tool_call",
            agent_path=["orchestrator", "w1"],
            data={"tool_name": "query_logs", "args": {"service": "api"}},
            task_id="t1",
        )

    def test_records_llm_end(self) -> None:
        """aafter_model emits llm_end for AI messages with content but no tool_calls."""
        import asyncio

        mw, collector = self._make_middleware()
        response = AIMessage(content="Final answer here")
        asyncio.run(mw.aafter_model({"messages": [], "response": response}))
        collector.record_sync.assert_called_with(
            event_type="llm_end",
            agent_path=["orchestrator", "w1"],
            data={"content": "Final answer here"},
            task_id="t1",
        )

    def test_no_response_is_noop(self) -> None:
        """aafter_model does nothing when response is not in state."""
        import asyncio

        mw, collector = self._make_middleware()
        asyncio.run(mw.aafter_model({"messages": []}))
        collector.record_sync.assert_not_called()
