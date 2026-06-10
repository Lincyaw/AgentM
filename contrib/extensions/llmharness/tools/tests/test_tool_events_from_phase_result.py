"""Lock the PhaseResult -> ToolEvent translation.

External trainers feed the output of this helper into
:func:`extractor_process_reward` / :func:`auditor_process_reward` to score
a rollout. If pairing of tool_call <-> tool_result drifts, every reward
the trainer observes shifts silently.
"""

from __future__ import annotations

from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
)

from llmharness import tool_events_from_phase_result
from llmharness.replay.engine import PhaseResult


def _assistant_with_call(call_id: str, name: str, args: dict[str, object]) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[ToolCallBlock(type="tool_call", id=call_id, name=name, arguments=args)],
        timestamp=0.0,
    )


def _tool_result(call_id: str, text: str, *, is_error: bool = False) -> ToolResultMessage:
    return ToolResultMessage(
        role="tool_result",
        content=[
            ToolResultBlock(
                type="tool_result",
                tool_call_id=call_id,
                content=[TextContent(type="text", text=text)],
                is_error=is_error,
            )
        ],
        timestamp=0.0,
    )


def _phase_result(messages: list) -> PhaseResult:
    return PhaseResult(output=None, status="ok", error=None, latency_ms=0, messages=messages)


def test_two_calls_two_results_paired_by_id() -> None:
    messages = [
        UserMessage(role="user", content=[TextContent(type="text", text="go")], timestamp=0.0),
        _assistant_with_call("c-1", "upsert_node", {"id": "n1"}),
        _tool_result("c-1", "ok"),
        _assistant_with_call("c-2", "finalize_extraction", {}),
        _tool_result("c-2", "ok"),
    ]
    events = tool_events_from_phase_result(_phase_result(messages))
    assert len(events) == 2
    assert events[0]["tool_name"] == "upsert_node"
    assert events[0]["args"] == {"id": "n1"}
    assert events[0]["is_error"] is False
    assert events[0]["error_text"] is None
    assert events[1]["tool_name"] == "finalize_extraction"
    assert events[1]["is_error"] is False
