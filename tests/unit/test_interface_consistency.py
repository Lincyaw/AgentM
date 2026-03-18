"""Tests for cross-module interface consistency.

These tests verify that interfaces between modules are compatible —
types, return values, and function signatures agree across module boundaries.

Bug prevented: Module A expects Module B to accept type X, but Module B's
signature says type Y → runtime TypeError when the modules interact.
"""

from __future__ import annotations

from typing import get_type_hints
from unittest.mock import AsyncMock

import pytest

from agentm.tools.orchestrator import create_orchestrator_tools


@pytest.fixture
def dispatch_agent_func():
    """Get dispatch_agent function from factory for testing."""
    mock_tm = AsyncMock()
    tools = create_orchestrator_tools(mock_tm, agent_pool=None)
    return tools["dispatch_agent"]


class TestTaskTypeLiteralConsistency:
    """Ref: designs/sub-agent.md § Task Types, designs/orchestrator.md § dispatch_agent

    task_type is str (widened from Literal) so that memory-extraction task types
    (collect, analyze, extract, refine) flow through alongside RCA types.
    The ANSWER_SCHEMA registry is the single source of truth for valid values.

    Bug prevented: ANSWER_SCHEMA missing a task_type that workers use →
    KeyError at runtime when creating the structured output model.
    """

    def test_answer_schema_covers_rca_task_types(self):
        """ANSWER_SCHEMA must include the three RCA worker task types."""
        from agentm.models.answer_schemas import ANSWER_SCHEMA
        from agentm.scenarios import discover

        discover()
        for task_type in ("scout", "verify", "deep_analyze"):
            assert task_type in ANSWER_SCHEMA, (
                f"ANSWER_SCHEMA missing RCA task type: {task_type!r}"
            )

    def test_answer_schema_covers_memory_extraction_task_types(self):
        """ANSWER_SCHEMA must include the four memory-extraction worker task types."""
        from agentm.models.answer_schemas import ANSWER_SCHEMA
        from agentm.scenarios import discover

        discover()
        for task_type in ("collect", "analyze", "extract", "refine"):
            assert task_type in ANSWER_SCHEMA, (
                f"ANSWER_SCHEMA missing memory-extraction task type: {task_type!r}"
            )

    def test_task_type_is_str_annotation(self, dispatch_agent_func):
        """dispatch_agent task_type must be annotated as str (not Literal)."""
        hints = get_type_hints(dispatch_agent_func, include_extras=True)
        annotation = hints.get("task_type")
        assert annotation is str, (
            f"dispatch_agent task_type should be str, got: {annotation}"
        )

    def test_dispatch_agent_and_task_manager_submit_agree(self, dispatch_agent_func):
        """Both dispatch_agent and TaskManager.submit must use str task_type."""
        from agentm.core.task_manager import TaskManager

        dispatch_hints = get_type_hints(dispatch_agent_func, include_extras=True)
        submit_hints = get_type_hints(TaskManager.submit, include_extras=True)

        dispatch_annotation = dispatch_hints.get("task_type")
        submit_annotation = submit_hints.get("task_type")

        assert dispatch_annotation is str, (
            f"dispatch_agent task_type should be str, got: {dispatch_annotation}"
        )
        assert submit_annotation is str, (
            f"TaskManager.submit task_type should be str, got: {submit_annotation}"
        )


class TestOrchestratorCreationImports:
    """Ref: designs/orchestrator.md § Orchestrator Creation

    The node orchestrator module must export the expected factory function.

    Bug: function renamed or moved → builder can't find it →
    system startup fails.
    """

    def test_create_node_orchestrator_exists(self):
        from agentm.agents.node.orchestrator import create_node_orchestrator

        assert callable(create_node_orchestrator)


class TestHooksModuleExports:
    """Ref: designs/sub-agent.md § Instruction Queue, designs/orchestrator.md § Instruction Injection

    The hooks module must export the instruction hook builder.
    Hook chaining is handled by compose_middleware / NodePipeline.

    Bug: build_instruction_hook missing → Sub-Agents can't inject
    instructions → pending instructions silently dropped.
    """

    def test_build_instruction_hook_exists(self):
        from agentm.middleware.instruction import build_instruction_hook

        assert callable(build_instruction_hook)


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
