"""Trajectory middleware — records LLM input messages to the trajectory.

Emits ``llm_start`` trajectory events containing message summaries
before each LLM invocation.
"""

from __future__ import annotations

from typing import Any, Callable

from agentm.core.trajectory import TrajectoryCollector
from agentm.middleware import AgentMMiddleware


def build_llm_input_hook(
    trajectory: TrajectoryCollector,
    agent_path: list[str],
    task_id: str | None = None,
) -> Callable:
    """Build a pre_model_hook that records the messages sent to the LLM.

    Emits an ``llm_start`` trajectory event containing a summary of the
    messages about to be fed to the model.  Uses ``record_sync`` because
    pre_model_hook is synchronous.

    Each compiled subgraph is now created per-dispatch with the correct
    *agent_path* baked in, so no mutable override is needed.  The optional
    *task_id* is forwarded to ``trajectory.record_sync`` for correlation.
    """

    last_message_count = 0

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal last_message_count
        messages = state.get("llm_input_messages") or state.get("messages", [])
        total = len(messages)
        new_messages = messages[last_message_count:]
        last_message_count = total

        trajectory.record_sync(
            event_type="llm_start",
            agent_path=agent_path,
            data={
                "message_count": total,
                "new_message_count": len(new_messages),
                "messages": new_messages,
            },
            task_id=task_id,
        )
        # Pass through unchanged
        if "llm_input_messages" in state:
            return {"llm_input_messages": state["llm_input_messages"]}
        return {"messages": messages}

    return hook


class TrajectoryMiddleware(AgentMMiddleware):
    """Pre- and post-model middleware that records trajectory events.

    ``before_model``: emits ``llm_start`` with message summaries.
    ``aafter_model``: emits ``tool_call`` / ``llm_end`` from the LLM response.
    """

    def __init__(
        self,
        trajectory: TrajectoryCollector,
        agent_path: list[str],
        task_id: str | None = None,
    ) -> None:
        self._trajectory = trajectory
        self._agent_path = agent_path
        self._task_id = task_id
        self._hook = build_llm_input_hook(trajectory, agent_path, task_id=task_id)

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._hook(state)

    async def aafter_model(
        self, state: dict[str, Any], runtime: Any = None
    ) -> dict[str, Any] | None:
        """Record ``tool_call`` or ``llm_end`` events from the LLM response."""
        response = state.get("response")
        if response is None:
            return None

        ai_msg = response
        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        content = getattr(ai_msg, "content", "")

        if tool_calls:
            for tc in tool_calls:
                self._trajectory.record_sync(
                    event_type="tool_call",
                    agent_path=self._agent_path,
                    data={
                        "tool_name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    },
                    task_id=self._task_id,
                )
        elif content:
            self._trajectory.record_sync(
                event_type="llm_end",
                agent_path=self._agent_path,
                data={"content": str(content)},
                task_id=self._task_id,
            )
        return None
