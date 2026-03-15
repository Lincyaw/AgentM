"""Trajectory middleware — records LLM input messages to the trajectory.

Emits ``llm_start`` trajectory events containing message summaries
before each LLM invocation.
"""

from __future__ import annotations

from typing import Any, Callable

from agentm.core.trajectory import TrajectoryCollector
from agentm.middleware import AgentMMiddleware


def _full_message(msg: Any) -> dict[str, Any]:
    """Convert a LangChain message to a plain dict with full content.

    Extracts role, content, and — for AI messages — tool_calls with id/name/args.
    For tool messages, includes the tool name and tool_call_id.
    """
    role = getattr(msg, "type", "unknown")
    content = getattr(msg, "content", "")
    entry: dict[str, Any] = {"role": role, "content": content}
    if role == "ai":
        tc = getattr(msg, "tool_calls", None)
        if tc:
            entry["tool_calls"] = [
                {
                    "id": c.get("id", ""),
                    "name": c.get("name", ""),
                    "args": c.get("args", {}),
                }
                for c in tc
            ]
    elif role == "tool":
        entry["name"] = getattr(msg, "name", "")
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
    return entry


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

    def _truncate(text: str, limit: int) -> str:
        return text[:limit] + ("..." if len(text) > limit else "")

    def _summarize(msg: Any) -> dict[str, Any]:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        entry: dict[str, Any] = {"role": role}
        if role == "system":
            entry["content"] = _truncate(content, 500)
        elif role == "ai":
            tc = getattr(msg, "tool_calls", None)
            if tc:
                entry["tool_calls"] = [{"name": c.get("name", "")} for c in tc]
            if content:
                entry["content"] = _truncate(content, 300)
        elif role == "tool":
            entry["name"] = getattr(msg, "name", "")
            entry["content"] = _truncate(content, 200)
        elif role == "human":
            entry["content"] = _truncate(content, 300)
        else:
            entry["content"] = _truncate(content, 200)
        return entry

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("llm_input_messages") or state.get("messages", [])

        trajectory.record_sync(
            event_type="llm_start",
            agent_path=agent_path,
            data={
                "message_count": len(messages),
                "messages": [_summarize(msg) for msg in messages],
                "full_messages": [_full_message(msg) for msg in messages],
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
