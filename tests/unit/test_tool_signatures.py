"""Focused contract tests for orchestrator tool signatures."""

from __future__ import annotations

import inspect
from typing import Literal, get_args, get_origin, get_type_hints

import pytest

from agentm.harness.runtime import AgentRuntime
from agentm.tools.orchestrator import create_orchestrator_tools


def _resolve_annotation(func, param_name: str):
    return get_type_hints(func, include_extras=True).get(param_name)


def _extract_literal_values(annotation) -> set[str] | None:
    origin = get_origin(annotation)
    if origin is Literal:
        return set(get_args(annotation))
    return None


class _MockWorkerFactory:
    def create_worker(
        self,
        agent_id: str,
        task_type: str,
        *,
        task_id: str | None = None,
    ):
        return None


@pytest.fixture
def orch_tools():
    runtime = AgentRuntime()
    tools = create_orchestrator_tools(runtime, _MockWorkerFactory())

    from agentm.harness.scenario import SetupContext
    from agentm.scenarios.rca.scenario import RCAScenario

    wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
    for tool in wiring.orchestrator_tools:
        tools[tool.name] = tool.func
    return tools


def test_dispatch_agent_contract_has_required_task_type_and_metadata(orch_tools) -> None:
    func = orch_tools["dispatch_agent"]
    sig = inspect.signature(func)
    assert inspect.iscoroutinefunction(func)
    assert "task_type" in sig.parameters
    assert sig.parameters["task_type"].default is inspect.Parameter.empty
    assert _resolve_annotation(func, "task_type") is str
    assert "metadata" in sig.parameters


def test_check_tasks_contract_is_async_and_no_legacy_wait_param(orch_tools) -> None:
    func = orch_tools["check_tasks"]
    sig = inspect.signature(func)
    assert inspect.iscoroutinefunction(func)
    assert "wait_seconds" not in sig.parameters


def test_update_hypothesis_status_literal_matches_store_contract(orch_tools) -> None:
    from agentm.scenarios.rca.hypothesis_store import HypothesisStore

    func = orch_tools["update_hypothesis"]
    literal_values = _extract_literal_values(_resolve_annotation(func, "status"))
    assert literal_values == HypothesisStore._VALID_STATUSES


def test_worker_factory_create_worker_requires_agent_id_and_task_type() -> None:
    from agentm.harness.worker_factory import WorkerLoopFactory

    sig = inspect.signature(WorkerLoopFactory.create_worker)
    assert "agent_id" in sig.parameters
    assert "task_type" in sig.parameters
    assert "task_id" in sig.parameters


def test_inject_and_abort_tool_contracts_are_async_and_have_required_ids(orch_tools) -> None:
    inject = orch_tools["inject_instruction"]
    abort = orch_tools["abort_task"]

    assert inspect.iscoroutinefunction(inject)
    assert "task_id" in inspect.signature(inject).parameters
    assert "instruction" in inspect.signature(inject).parameters

    assert inspect.iscoroutinefunction(abort)
    assert "task_id" in inspect.signature(abort).parameters
    assert "reason" in inspect.signature(abort).parameters
