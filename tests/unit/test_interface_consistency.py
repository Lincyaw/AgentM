"""Tests for cross-module interface consistency.

These tests verify that interfaces between modules are compatible —
types, return values, and function signatures agree across module boundaries.

Bug prevented: Module A expects Module B to accept type X, but Module B's
signature says type Y → runtime TypeError when the modules interact.
"""

from __future__ import annotations

import inspect
from typing import Literal, get_args, get_origin, get_type_hints


def _extract_task_type_values(func) -> set[str]:
    """Extract the Literal values of the task_type parameter from a function.

    Handles `from __future__ import annotations` by using get_type_hints().
    """
    hints = get_type_hints(func, include_extras=True)
    annotation = hints.get("task_type")
    assert annotation is not None, f"{func.__qualname__} missing task_type annotation"

    origin = get_origin(annotation)
    if origin is Literal:
        return set(get_args(annotation))

    from typing import Annotated
    if origin is Annotated:
        inner = get_args(annotation)[0]
        if get_origin(inner) is Literal:
            return set(get_args(inner))

    raise AssertionError(f"{func.__qualname__} task_type is not Literal: {annotation}")


class TestTaskTypeLiteralConsistency:
    """Ref: designs/sub-agent.md § Task Types, designs/orchestrator.md § spawn_worker

    task_type Literal values must be identical across modules that use them:
    - agents/sub_agent.py create_sub_agent
    - core/task_manager.py TaskManager.submit

    Bug: one module has {"scout", "verify", "deep_analyze"} but another uses
    {"scout", "verify", "analyze"} → spawn_worker passes "deep_analyze"
    to create_sub_agent which doesn't recognize it.
    """

    def test_create_sub_agent_and_task_manager_submit_agree(self):
        from agentm.agents.sub_agent import create_sub_agent
        from agentm.core.task_manager import TaskManager

        sub_agent_values = _extract_task_type_values(create_sub_agent)
        submit_values = _extract_task_type_values(TaskManager.submit)
        assert sub_agent_values == submit_values, (
            f"task_type Literal mismatch:\n"
            f"  create_sub_agent: {sub_agent_values}\n"
            f"  TaskManager.submit: {submit_values}"
        )


class TestOrchestratorCreationImports:
    """Ref: designs/orchestrator.md § Orchestrator Creation

    The orchestrator module must export the expected factory functions.

    Bug: function renamed or moved → AgentSystemBuilder can't find it →
    system startup fails.
    """

    def test_build_orchestrator_prompt_exists(self):
        from agentm.agents.orchestrator import build_orchestrator_prompt
        assert callable(build_orchestrator_prompt)

    def test_create_orchestrator_exists(self):
        from agentm.agents.orchestrator import create_orchestrator
        assert callable(create_orchestrator)


class TestHooksModuleExports:
    """Ref: designs/sub-agent.md § Instruction Queue, designs/orchestrator.md § Instruction Injection

    The hooks module must export both hook builders.

    Bug: build_combined_hook missing → Sub-Agents can't chain instruction
    injection with compression → one of the two hooks silently doesn't fire.
    """

    def test_build_instruction_hook_exists(self):
        from agentm.agents.hooks import build_instruction_hook
        assert callable(build_instruction_hook)

    def test_build_combined_hook_exists(self):
        from agentm.agents.hooks import build_combined_hook
        assert callable(build_combined_hook)


class TestCompressionRefLayerValues:
    """Ref: designs/orchestrator.md § Compression Architecture

    CompressionRef.layer must be Literal["sub_agent", "orchestrator"].
    This is enforced by the dataclass field type annotation.

    Bug: layer accepts any string → recall_history can't filter by layer →
    returns data from wrong compression scope.
    """

    def test_layer_annotation_is_literal(self):
        from agentm.models.data import CompressionRef
        annotation = CompressionRef.__dataclass_fields__["layer"].type
        # The annotation should be Literal["sub_agent", "orchestrator"]
        # With from __future__ import annotations, it's stored as a string
        assert "sub_agent" in str(annotation) and "orchestrator" in str(annotation), (
            f"CompressionRef.layer annotation should be Literal['sub_agent', 'orchestrator'], "
            f"got: {annotation}"
        )
