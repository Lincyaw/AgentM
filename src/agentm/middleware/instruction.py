"""Instruction injection middleware — injects pending instructions as HumanMessages.

Ref: designs/sub-agent.md § Instruction Queue
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage

from agentm.core.task_manager import TaskManager
from agentm.middleware import AgentMMiddleware


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


class InstructionInjectionMiddleware(AgentMMiddleware):
    """Pre-model middleware that injects pending instructions as HumanMessages."""

    def __init__(self, task_manager: TaskManager, task_id: str) -> None:
        self._hook = build_instruction_hook(task_manager, task_id)

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._hook(state)
