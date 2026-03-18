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


class InstructionInjectionMiddleware(AgentMMiddleware):
    """Pre-model middleware that injects pending instructions as HumanMessages."""

    def __init__(self, task_manager: TaskManager, task_id: str) -> None:
        self._hook = build_instruction_hook(task_manager, task_id)

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._hook(state)
