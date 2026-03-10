"""Pre-model hook builders for Sub-Agent instruction injection and chaining."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage

from agentm.core.task_manager import TaskManager
from agentm.core.trajectory import TrajectoryCollector


def build_instruction_hook(task_manager: TaskManager, task_id: str) -> Callable:
    """Build a pre_model_hook that injects pending instructions as HumanMessages.

    The returned hook is called before each LLM invocation. It checks
    task_manager.consume_instructions(task_id) and prepends any pending
    instructions as HumanMessages to the LLM input.

    Ref: designs/sub-agent.md § Instruction Queue
    """

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        instructions = task_manager.consume_instructions(task_id)
        original_messages = state.get("messages", [])
        if not instructions:
            return {"messages": original_messages}
        injected = [HumanMessage(content=instr) for instr in instructions]
        return {"messages": [*injected, *original_messages]}

    return hook


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

    def _full_message(msg: Any) -> dict[str, Any]:
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


def build_combined_hook(
    instruction_hook: Callable, compression_hook: Callable
) -> Callable:
    """Chain instruction + compression hooks into a single pre_model_hook.

    Applies instruction_hook first (to inject messages), then compression_hook
    (to compress if needed). Both hooks follow the pre_model_hook protocol:
    receive state dict, return dict with 'messages' or 'llm_input_messages'.

    Ref: designs/orchestrator.md § Instruction Injection
    """

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        intermediate = instruction_hook(state)
        result = compression_hook(intermediate)
        if "llm_input_messages" in result:
            return {"llm_input_messages": result["llm_input_messages"]}
        return {"messages": result["messages"]}

    return hook
