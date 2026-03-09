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
) -> Callable:
    """Build a pre_model_hook that records the messages sent to the LLM.

    Emits an ``llm_start`` trajectory event containing a summary of the
    messages about to be fed to the model.  Uses ``record_sync`` because
    pre_model_hook is synchronous.
    """

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("llm_input_messages") or state.get("messages", [])
        summary: list[dict[str, Any]] = []
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            entry: dict[str, Any] = {"role": role}
            if role == "system":
                entry["content"] = content[:500] + ("..." if len(content) > 500 else "")
            elif role == "ai":
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    entry["tool_calls"] = [{"name": c.get("name", "")} for c in tc]
                if content:
                    entry["content"] = content[:300] + (
                        "..." if len(content) > 300 else ""
                    )
            elif role == "tool":
                entry["name"] = getattr(msg, "name", "")
                entry["content"] = content[:200] + ("..." if len(content) > 200 else "")
            elif role == "human":
                entry["content"] = content[:300] + ("..." if len(content) > 300 else "")
            else:
                entry["content"] = content[:200] + ("..." if len(content) > 200 else "")
            summary.append(entry)

        trajectory.record_sync(
            event_type="llm_start",
            agent_path=agent_path,
            data={
                "message_count": len(messages),
                "messages": summary,
            },
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
