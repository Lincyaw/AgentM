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

import pytest

from agentm.tools.orchestrator import create_orchestrator_tools
from agentm.harness.runtime import AgentRuntime


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


class _MockWorkerFactory:
    """Minimal WorkerFactory for signature testing."""

    def create_worker(self, agent_id: str, task_type: str):
        return None


@pytest.fixture
def orch_tools():
    """Create orchestrator tool functions via factory for signature testing."""
    runtime = AgentRuntime()
    factory = _MockWorkerFactory()
    tools = create_orchestrator_tools(runtime, factory)

    # Also load RCA-specific tools from the Scenario protocol
    from agentm.harness.scenario import SetupContext
    from agentm.scenarios.rca.scenario import RCAScenario

    wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
    for tool in wiring.orchestrator_tools:
        tools[tool.name] = tool.func

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
        """task_type is str so trajectory-analysis types flow through alongside RCA types."""
        annotation = _resolve_annotation(orch_tools["dispatch_agent"], "task_type")
        assert annotation is str, f"task_type should be str, got: {annotation}"

    def test_task_type_is_required(self, orch_tools):
        """task_type has no default — each scenario defines its own task types."""
        sig = inspect.signature(orch_tools["dispatch_agent"])
        param = sig.parameters["task_type"]
        assert param.default is inspect.Parameter.empty

    def test_has_metadata_parameter(self, orch_tools):
        """metadata replaces hypothesis_id — scenario-agnostic extension point."""
        sig = inspect.signature(orch_tools["dispatch_agent"])
        assert "metadata" in sig.parameters

    def test_no_tool_call_id_parameter(self, orch_tools):
        """tool_call_id removed — SimpleAgentLoop handles ToolMessage wrapping."""
        sig = inspect.signature(orch_tools["dispatch_agent"])
        assert "tool_call_id" not in sig.parameters


class TestCheckTasksSignature:
    """Ref: designs/orchestrator.md § check_tasks

    Bug: check_tasks is sync → blocks the event loop while waiting for Sub-Agents,
    preventing other agents from running concurrently.
    """

    def test_is_async(self, orch_tools):
        assert inspect.iscoroutinefunction(orch_tools["check_tasks"])

    def test_no_wait_seconds_parameter(self, orch_tools):
        """wait_seconds removed — SmartWaitStrategy controls timing."""
        sig = inspect.signature(orch_tools["check_tasks"])
        assert "wait_seconds" not in sig.parameters

    def test_no_tool_call_id_parameter(self, orch_tools):
        """tool_call_id removed — SimpleAgentLoop handles ToolMessage wrapping."""
        sig = inspect.signature(orch_tools["check_tasks"])
        assert "tool_call_id" not in sig.parameters


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

    def test_no_tool_call_id_parameter(self, orch_tools):
        """tool_call_id removed — SimpleAgentLoop handles ToolMessage wrapping."""
        sig = inspect.signature(orch_tools["update_hypothesis"])
        assert "tool_call_id" not in sig.parameters


class TestWorkerLoopFactorySignature:
    """WorkerLoopFactory.create_worker must accept agent_id and task_type."""

    def test_has_task_type_parameter(self):
        from agentm.harness.worker_factory import WorkerLoopFactory

        sig = inspect.signature(WorkerLoopFactory.create_worker)
        assert "task_type" in sig.parameters

    def test_has_agent_id_parameter(self):
        from agentm.harness.worker_factory import WorkerLoopFactory

        sig = inspect.signature(WorkerLoopFactory.create_worker)
        assert "agent_id" in sig.parameters


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
