"""Pre-model hook builders for Sub-Agent instruction injection and chaining.

All functions are stubs — raise NotImplementedError.
"""

from __future__ import annotations

from typing import Callable

from agentm.core.task_manager import TaskManager


def build_instruction_hook(task_manager: TaskManager, task_id: str) -> Callable:
    """Build a pre_model_hook that injects pending instructions as HumanMessages.

    The returned hook is called before each LLM invocation. It checks
    task_manager.consume_instructions(task_id) and prepends any pending
    instructions as HumanMessages to the LLM input.

    Ref: designs/sub-agent.md § Instruction Queue
    """
    raise NotImplementedError


def build_combined_hook(instruction_hook: Callable, compression_hook: Callable) -> Callable:
    """Chain instruction + compression hooks into a single pre_model_hook.

    Applies instruction_hook first (to inject messages), then compression_hook
    (to compress if needed). Both hooks follow the pre_model_hook protocol:
    receive state dict, return dict with 'messages' or 'llm_input_messages'.

    Ref: designs/orchestrator.md § Instruction Injection
    """
    raise NotImplementedError
