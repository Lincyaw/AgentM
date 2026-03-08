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


class TestSpawnWorkerSignature:
    """Ref: designs/orchestrator.md § spawn_worker

    Bug: missing task_type parameter → LLM cannot specify scout/verify/deep_analyze,
    all tasks dispatch as default → no task specialization.
    """

    def test_is_async(self):
        from agentm.tools.orchestrator import spawn_worker
        assert inspect.iscoroutinefunction(spawn_worker)

    def test_has_task_type_parameter(self):
        from agentm.tools.orchestrator import spawn_worker
        sig = inspect.signature(spawn_worker)
        assert "task_type" in sig.parameters

    def test_has_instructions_parameter(self):
        from agentm.tools.orchestrator import spawn_worker
        sig = inspect.signature(spawn_worker)
        assert "instructions" in sig.parameters


class TestWaitForWorkersSignature:
    """Ref: designs/orchestrator.md § wait_for_workers

    Bug: wait_for_workers is sync → blocks the event loop while waiting for Sub-Agents,
    preventing other agents from running concurrently.
    """

    def test_is_async(self):
        from agentm.tools.orchestrator import wait_for_workers
        assert inspect.iscoroutinefunction(wait_for_workers)

    def test_has_timeout_seconds_parameter(self):
        from agentm.tools.orchestrator import wait_for_workers
        sig = inspect.signature(wait_for_workers)
        assert "timeout_seconds" in sig.parameters

    def test_timeout_seconds_defaults_to_30(self):
        from agentm.tools.orchestrator import wait_for_workers
        sig = inspect.signature(wait_for_workers)
        param = sig.parameters["timeout_seconds"]
        assert param.default == 30


class TestUpdateHypothesisSignature:
    """Ref: designs/orchestrator.md § update_hypothesis

    Bug: status parameter is Optional[str] instead of constrained →
    LLM can pass any string as status → bypasses state machine validation.
    """

    def test_is_sync(self):
        from agentm.tools.orchestrator import update_hypothesis
        assert not inspect.iscoroutinefunction(update_hypothesis)

    def test_description_is_required(self):
        """Description must NOT be Optional — every hypothesis needs a description."""
        from agentm.tools.orchestrator import update_hypothesis
        sig = inspect.signature(update_hypothesis)
        param = sig.parameters["description"]
        assert param.default is inspect.Parameter.empty

    def test_status_defaults_to_formed(self):
        from agentm.tools.orchestrator import update_hypothesis
        sig = inspect.signature(update_hypothesis)
        param = sig.parameters["status"]
        assert param.default == "formed"


class TestRemoveHypothesisSignature:
    """Ref: designs/orchestrator.md § remove_hypothesis"""

    def test_is_sync(self):
        from agentm.tools.orchestrator import remove_hypothesis
        assert not inspect.iscoroutinefunction(remove_hypothesis)

    def test_has_id_parameter(self):
        from agentm.tools.orchestrator import remove_hypothesis
        sig = inspect.signature(remove_hypothesis)
        assert "id" in sig.parameters


class TestCreateSubAgentSignature:
    """Ref: designs/sub-agent.md § Task Types

    Bug: create_sub_agent ignores task_type → all Sub-Agents use the same
    generic prompt regardless of whether they're scouting or verifying.
    """

    def test_has_task_type_parameter(self):
        from agentm.agents.sub_agent import create_sub_agent
        sig = inspect.signature(create_sub_agent)
        assert "task_type" in sig.parameters

    def test_task_type_is_literal_with_correct_values(self):
        from agentm.agents.sub_agent import create_sub_agent
        annotation = _resolve_annotation(create_sub_agent, "task_type")
        values = _extract_literal_values(annotation)
        assert values is not None, "task_type should be a Literal type"
        assert values == {"scout", "verify", "deep_analyze"}

    def test_task_type_defaults_to_scout(self):
        from agentm.agents.sub_agent import create_sub_agent
        sig = inspect.signature(create_sub_agent)
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

    def test_task_type_is_literal_with_correct_values(self):
        from agentm.core.task_manager import TaskManager
        annotation = _resolve_annotation(TaskManager.submit, "task_type")
        values = _extract_literal_values(annotation)
        assert values is not None, "task_type should be a Literal type"
        assert values == {"scout", "verify", "deep_analyze"}


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
        from agentm.core.compression import build_compression_hook
        sig = inspect.signature(build_compression_hook)
        param = sig.parameters["config"]
        # Should accept CompressionConfig, not Any
        assert param.annotation is not inspect.Parameter.empty
        # The annotation should reference CompressionConfig
        annotation_name = getattr(param.annotation, "__name__", str(param.annotation))
        assert "CompressionConfig" in annotation_name

    def test_build_compression_hook_returns_callable(self):
        from agentm.core.compression import build_compression_hook
        sig = inspect.signature(build_compression_hook)
        assert sig.return_annotation is not inspect.Parameter.empty
