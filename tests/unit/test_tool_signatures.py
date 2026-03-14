"""Tests for tool signature contracts against design documents.

Ref: designs/orchestrator.md § Orchestrator Tools
Ref: designs/sub-agent.md § Task Types

These tests verify that tool function signatures match the design doc's
normative definitions. They use introspection to check parameter names,
types, and defaults — NOT to test that stubs work.

Bug prevented: tool signature drifts from design → LLM receives wrong parameter
schema → tool calls fail at runtime or silently accept invalid arguments.
"""

from __future__ import annotations

import inspect
from typing import Literal, get_args, get_origin, get_type_hints
from unittest.mock import AsyncMock

import pytest

from agentm.tools.orchestrator import create_orchestrator_tools


def _resolve_annotation(func, param_name: str):
    """Resolve a parameter's type annotation, handling `from __future__ import annotations`.

    When `from __future__ import annotations` is active, annotations are strings.
    get_type_hints() evaluates them back to real types.
    """
    hints = get_type_hints(func, include_extras=True)
    return hints.get(param_name)


def _extract_literal_values(annotation) -> set[str] | None:
    """Extract Literal values from an annotation, handling Annotated wrappers."""
    origin = get_origin(annotation)

    # Direct Literal
    if origin is Literal:
        return set(get_args(annotation))

    # Annotated[Literal[...], ...] — unwrap
    from typing import Annotated

    if origin is Annotated:
        inner = get_args(annotation)[0]
        if get_origin(inner) is Literal:
            return set(get_args(inner))

    return None


@pytest.fixture
def orch_tools():
    """Create orchestrator tool functions via factory for signature testing."""
    mock_tm = AsyncMock()
    tools = create_orchestrator_tools(mock_tm, agent_pool=None)

    # Also load RCA-specific tools for testing their signatures
    from agentm.scenarios.rca.tools import create_rca_tools

    rca_tools = create_rca_tools(trajectory=None)
    tools.update(rca_tools)

    return tools


class TestDispatchAgentSignature:
    """Ref: designs/orchestrator.md § dispatch_agent

    Bug: missing task_type parameter → LLM cannot specify scout/verify/deep_analyze,
    all tasks dispatch as default → no task specialization.
    """

    def test_is_async(self, orch_tools):
        assert inspect.iscoroutinefunction(orch_tools["dispatch_agent"])

    def test_has_task_type_parameter(self, orch_tools):
        sig = inspect.signature(orch_tools["dispatch_agent"])
        assert "task_type" in sig.parameters

    def test_task_type_is_str(self, orch_tools):
        """task_type is str so memory-extraction types (collect, analyze, etc.) flow through."""
        annotation = _resolve_annotation(orch_tools["dispatch_agent"], "task_type")
        assert annotation is str, f"task_type should be str, got: {annotation}"

    def test_task_type_defaults_to_scout(self, orch_tools):
        sig = inspect.signature(orch_tools["dispatch_agent"])
        param = sig.parameters["task_type"]
        assert param.default == "scout"

    def test_has_tool_call_id_parameter(self, orch_tools):
        """Tool must accept tool_call_id for ToolMessage routing."""
        sig = inspect.signature(orch_tools["dispatch_agent"])
        assert "tool_call_id" in sig.parameters


class TestCheckTasksSignature:
    """Ref: designs/orchestrator.md § check_tasks

    Bug: check_tasks is sync → blocks the event loop while waiting for Sub-Agents,
    preventing other agents from running concurrently.
    """

    def test_is_async(self, orch_tools):
        assert inspect.iscoroutinefunction(orch_tools["check_tasks"])

    def test_has_wait_seconds_parameter(self, orch_tools):
        sig = inspect.signature(orch_tools["check_tasks"])
        assert "wait_seconds" in sig.parameters

    def test_has_tool_call_id_parameter(self, orch_tools):
        sig = inspect.signature(orch_tools["check_tasks"])
        assert "tool_call_id" in sig.parameters


class TestUpdateHypothesisSignature:
    """Ref: designs/orchestrator.md § update_hypothesis

    Bug: status parameter is Optional[str] instead of Literal → LLM can pass
    any string as status → bypasses state machine validation.
    """

    def test_is_async(self, orch_tools):
        assert inspect.iscoroutinefunction(orch_tools["update_hypothesis"])

    def test_description_is_required(self, orch_tools):
        """Description must NOT be Optional — every hypothesis needs a description.

        Bug: Optional description → LLM omits it → Notebook has undescribed hypothesis.
        """
        sig = inspect.signature(orch_tools["update_hypothesis"])
        param = sig.parameters["description"]
        # Required means no default value
        assert param.default is inspect.Parameter.empty

    def test_status_is_literal_matching_enum(self, orch_tools):
        """Status must be Literal with values matching HypothesisStatus enum.

        Bug: Literal values drift from enum → LLM passes valid Literal
        that the enum rejects downstream.
        """
        from agentm.scenarios.rca.enums import HypothesisStatus

        annotation = _resolve_annotation(orch_tools["update_hypothesis"], "status")
        literal_values = _extract_literal_values(annotation)
        assert literal_values is not None, "status should be a Literal type"
        enum_values = {member.value for member in HypothesisStatus}
        assert literal_values == enum_values, (
            f"Literal drift detected.\n"
            f"  Missing from Literal: {enum_values - literal_values}\n"
            f"  Extra in Literal: {literal_values - enum_values}"
        )

    def test_has_tool_call_id_parameter(self, orch_tools):
        sig = inspect.signature(orch_tools["update_hypothesis"])
        assert "tool_call_id" in sig.parameters


class TestBuildWorkerSubgraphSignature:
    """Ref: designs/sub-agent.md § Task Types

    Bug: build_worker_subgraph ignores task_type → all workers use the same
    generic prompt regardless of whether they're scouting or verifying.
    """

    def test_has_task_type_parameter(self):
        from agentm.agents.node.worker import build_worker_subgraph

        sig = inspect.signature(build_worker_subgraph)
        assert "task_type" in sig.parameters

    def test_task_type_is_str(self):
        """task_type is str so memory-extraction types flow through alongside RCA types."""
        from agentm.agents.node.worker import build_worker_subgraph

        annotation = _resolve_annotation(build_worker_subgraph, "task_type")
        assert annotation is str, f"task_type should be str, got: {annotation}"

    def test_task_type_defaults_to_scout(self):
        from agentm.agents.node.worker import build_worker_subgraph

        sig = inspect.signature(build_worker_subgraph)
        param = sig.parameters["task_type"]
        assert param.default == "scout"


class TestTaskManagerSubmitSignature:
    """Ref: designs/orchestrator.md § TaskManager

    Bug: TaskManager.submit() missing task_type → cannot specialize
    Sub-Agent behavior based on task purpose.
    """

    def test_is_async(self):
        from agentm.core.task_manager import TaskManager

        assert inspect.iscoroutinefunction(TaskManager.submit)

    def test_has_task_type_parameter(self):
        from agentm.core.task_manager import TaskManager

        sig = inspect.signature(TaskManager.submit)
        assert "task_type" in sig.parameters

    def test_task_type_is_str(self):
        """task_type is str (widened from Literal) to support multiple scenario types."""
        from agentm.core.task_manager import TaskManager

        annotation = _resolve_annotation(TaskManager.submit, "task_type")
        assert annotation is str, f"task_type should be str, got: {annotation}"


class TestTaskManagerMethodContracts:
    """Ref: designs/orchestrator.md § TaskManager, Instruction Injection

    Bug: TaskManager missing consume_instructions → instruction hook has nothing
    to dequeue → inject_instruction silently drops messages.
    """

    def test_has_consume_instructions_method(self):
        from agentm.core.task_manager import TaskManager

        assert hasattr(TaskManager, "consume_instructions")
        assert callable(TaskManager.consume_instructions)

    def test_has_get_task_method(self):
        from agentm.core.task_manager import TaskManager

        assert hasattr(TaskManager, "get_task")
        assert callable(TaskManager.get_task)

    def test_has_execute_agent_method(self):
        from agentm.core.task_manager import TaskManager

        assert hasattr(TaskManager, "_execute_agent")
        assert inspect.iscoroutinefunction(TaskManager._execute_agent)


class TestCompressionHookSignature:
    """Ref: designs/sub-agent.md § Context Compression

    Bug: build_compression_hook accepts untyped config → caller passes wrong
    config shape → compression silently misconfigured at runtime.
    """

    def test_build_compression_hook_accepts_compression_config(self):
        from agentm.middleware.compression import build_compression_hook

        sig = inspect.signature(build_compression_hook)
        param = sig.parameters["config"]
        # Should accept CompressionConfig, not Any
        assert param.annotation is not inspect.Parameter.empty
        # The annotation should reference CompressionConfig
        annotation_name = getattr(param.annotation, "__name__", str(param.annotation))
        assert "CompressionConfig" in annotation_name

    def test_build_compression_hook_returns_callable(self):
        from agentm.middleware.compression import build_compression_hook

        sig = inspect.signature(build_compression_hook)
        assert sig.return_annotation is not inspect.Parameter.empty


class TestInjectInstructionSignature:
    """inject_instruction should be async to properly delegate to TaskManager.inject().

    Bug: sync inject_instruction bypasses TaskManager.inject() status validation,
    allowing injection into completed/failed tasks.
    """

    def test_is_async(self, orch_tools):
        assert inspect.iscoroutinefunction(orch_tools["inject_instruction"])

    def test_has_task_id_parameter(self, orch_tools):
        sig = inspect.signature(orch_tools["inject_instruction"])
        assert "task_id" in sig.parameters

    def test_has_instruction_parameter(self, orch_tools):
        sig = inspect.signature(orch_tools["inject_instruction"])
        assert "instruction" in sig.parameters


class TestAbortTaskSignature:
    """abort_task should be async to properly delegate to TaskManager.abort().

    Bug: sync abort_task bypasses TaskManager.abort() status validation and
    trajectory recording, silently corrupting task state.
    """

    def test_is_async(self, orch_tools):
        assert inspect.iscoroutinefunction(orch_tools["abort_task"])

    def test_has_task_id_parameter(self, orch_tools):
        sig = inspect.signature(orch_tools["abort_task"])
        assert "task_id" in sig.parameters

    def test_has_reason_parameter(self, orch_tools):
        sig = inspect.signature(orch_tools["abort_task"])
        assert "reason" in sig.parameters
