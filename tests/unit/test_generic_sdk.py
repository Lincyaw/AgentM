"""Tests for the generic SDK architecture: strategy protocol, strategy registry,
middleware composition, storage backend, composite backend, task result, and
generic builder.

Ref: designs/generic-state-wrapper.md § Generic SDK Redesign
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentm.backends.composite import CompositeBackend
from agentm.backends.filesystem import FilesystemBackend
from agentm.core.backend import StorageBackend
from agentm.core.strategy import ReasoningStrategy
from agentm.core.strategy_registry import (
    get_strategy,
    list_strategies,
    register_strategy,
)
from agentm.middleware import AgentMMiddleware, compose_middleware
from agentm.middleware.budget import BudgetMiddleware
from agentm.middleware.dedup import DedupMiddleware
from agentm.models.state import BaseExecutorState, S
from agentm.models.task_result import TaskResult
from agentm.scenarios import discover

# Ensure scenario registrations are loaded for all tests in this module.
discover()


# ---------------------------------------------------------------------------
# Strategy Protocol
# ---------------------------------------------------------------------------


class TestReasoningStrategyProtocol:
    """ReasoningStrategy is runtime_checkable and satisfied by concrete strategies.

    Bug prevented: a strategy that doesn't implement all methods silently
    accepted → crashes at runtime during execution.
    """

    def test_hypothesis_driven_satisfies_protocol(self):
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        strategy = HypothesisDrivenStrategy()
        assert isinstance(strategy, ReasoningStrategy)

    def test_memory_extraction_satisfies_protocol(self):
        from agentm.scenarios.memory_extraction.strategy import MemoryExtractionStrategy

        strategy = MemoryExtractionStrategy()
        assert isinstance(strategy, ReasoningStrategy)

    def test_hypothesis_driven_name(self):
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        assert HypothesisDrivenStrategy().name == "hypothesis_driven"

    def test_memory_extraction_name(self):
        from agentm.scenarios.memory_extraction.strategy import MemoryExtractionStrategy

        assert MemoryExtractionStrategy().name == "memory_extraction"


# ---------------------------------------------------------------------------
# Strategy Registry
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    """Strategy registry maps system-type strings to strategy instances.

    Bug prevented: typo in system_type silently returns None → NoneType
    attribute errors deep in the execution pipeline.
    """

    def test_get_hypothesis_driven(self):
        strategy = get_strategy("hypothesis_driven")
        assert strategy.name == "hypothesis_driven"

    def test_get_memory_extraction(self):
        strategy = get_strategy("memory_extraction")
        assert strategy.name == "memory_extraction"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="No strategy registered"):
            get_strategy("unknown_type")

    def test_list_strategies_returns_registered(self):
        names = list_strategies()
        assert "hypothesis_driven" in names
        assert "memory_extraction" in names

    def test_register_custom_strategy(self):
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        register_strategy("custom_test", HypothesisDrivenStrategy())
        try:
            strategy = get_strategy("custom_test")
            assert strategy.name == "hypothesis_driven"
        finally:
            # Clean up
            from agentm.core.strategy_registry import _STRATEGY_INSTANCES

            _STRATEGY_INSTANCES.pop("custom_test", None)


# ---------------------------------------------------------------------------
# Hypothesis-Driven Strategy
# ---------------------------------------------------------------------------


class TestHypothesisDrivenStrategy:
    """HypothesisDrivenStrategy produces correct initial state and phase graph.

    Bug prevented: missing field in initial state → KeyError during graph execution.
    """

    def test_initial_state_has_required_fields(self):
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        strategy = HypothesisDrivenStrategy()
        state = strategy.initial_state("t1", "test task")
        assert state["task_id"] == "t1"
        assert state["task_description"] == "test task"
        assert state["current_phase"] == "exploration"
        assert state["notebook"] is not None
        assert state["compression_refs"] == []

    def test_phase_definitions_has_four_phases(self):
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        phases = HypothesisDrivenStrategy().phase_definitions()
        assert set(phases.keys()) == {
            "exploration",
            "generation",
            "verification",
            "confirmation",
        }

    def test_format_context_empty_notebook(self):
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        strategy = HypothesisDrivenStrategy()
        state = strategy.initial_state("t1", "test")
        result = strategy.format_context(state)
        assert "Diagnostic Notebook" in result

    def test_format_context_no_notebook(self):
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        strategy = HypothesisDrivenStrategy()
        state: dict[str, Any] = {
            "messages": [],
            "task_id": "t1",
            "task_description": "t",
            "current_phase": "x",
        }
        result = strategy.format_context(state)
        assert "no data collected" in result

    def test_should_terminate_false_initially(self):
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        strategy = HypothesisDrivenStrategy()
        state = strategy.initial_state("t1", "test")
        assert strategy.should_terminate(state) is False

    def test_answer_schemas_has_rca_types(self):
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        schemas = HypothesisDrivenStrategy().get_answer_schemas()
        assert "scout" in schemas
        assert "deep_analyze" in schemas
        assert "verify" in schemas

    def test_state_schema_returns_correct_type(self):
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy
        from agentm.scenarios.rca.state import HypothesisDrivenState

        assert HypothesisDrivenStrategy().state_schema() is HypothesisDrivenState


# ---------------------------------------------------------------------------
# Memory-Extraction Strategy
# ---------------------------------------------------------------------------


class TestMemoryExtractionStrategy:
    """MemoryExtractionStrategy produces correct initial state and phase graph."""

    def test_initial_state_has_required_fields(self):
        from agentm.scenarios.memory_extraction.strategy import MemoryExtractionStrategy

        strategy = MemoryExtractionStrategy()
        state = strategy.initial_state("t1", "extract")
        assert state["task_id"] == "t1"
        assert state["current_phase"] == "collect"
        assert state["source_trajectories"] == []
        assert state["extracted_patterns"] == []

    def test_phase_definitions_has_four_phases(self):
        from agentm.scenarios.memory_extraction.strategy import MemoryExtractionStrategy

        phases = MemoryExtractionStrategy().phase_definitions()
        assert set(phases.keys()) == {"collect", "analyze", "extract", "refine"}

    def test_answer_schemas_has_extraction_types(self):
        from agentm.scenarios.memory_extraction.strategy import MemoryExtractionStrategy

        schemas = MemoryExtractionStrategy().get_answer_schemas()
        assert "collect" in schemas
        assert "analyze" in schemas
        assert "extract" in schemas
        assert "refine" in schemas


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class TestMiddlewareBase:
    """AgentMMiddleware base class and compose_middleware produce valid hooks.

    Bug prevented: compose_middleware passes wrong state format between hooks →
    missing 'messages' or 'llm_input_messages' key.
    """

    def test_base_middleware_passthrough(self):
        mw = AgentMMiddleware()
        state = {"messages": ["hello"]}
        hook = mw.to_pre_model_hook()
        result = hook(state)
        assert result["messages"] == ["hello"]

    def test_compose_empty_list(self):
        hook = compose_middleware([])
        state = {"messages": ["test"]}
        result = hook(state)
        assert result == state

    def test_compose_preserves_llm_input_messages(self):
        mw = AgentMMiddleware()
        hook = compose_middleware([mw])
        state = {"llm_input_messages": ["rewritten"]}
        result = hook(state)
        assert result["llm_input_messages"] == ["rewritten"]

    def test_compose_chains_multiple(self):
        class AddMarker(AgentMMiddleware):
            def __init__(self, marker: str):
                self._marker = marker

            def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
                msgs = list(state.get("messages", []))
                msgs.append(self._marker)
                return {"messages": msgs}

        hook = compose_middleware([AddMarker("A"), AddMarker("B")])
        result = hook({"messages": []})
        assert result["messages"] == ["A", "B"]


class TestBudgetMiddleware:
    """BudgetMiddleware injects warnings when step budget is low."""

    def _ai_with_tools(self, name: str = "query_sql") -> Any:
        """Create a minimal AI message with a tool_call."""
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": name, "args": {}}],
        )

    def test_no_warning_when_budget_plentiful(self):
        mw = BudgetMiddleware(max_steps=20)
        state = {"messages": []}
        result = mw.before_model(state)
        assert result == {"messages": []}

    def test_think_is_counted_as_step(self):
        """Think tool calls must count toward the budget (prevents think-loops)."""
        mw = BudgetMiddleware(max_steps=5)
        think_msgs = [self._ai_with_tools("think") for _ in range(4)]
        result = mw.before_model({"messages": think_msgs})
        # 4/5 used → 1 remaining ≤ 3 → should inject WARNING
        last = result["messages"][-1]
        assert "WARNING" in last.content
        assert "1/5" in last.content

    def test_exhausted_property_false_initially(self):
        mw = BudgetMiddleware(max_steps=10)
        assert mw.exhausted is False

    def test_exhausted_property_true_when_budget_depleted(self):
        mw = BudgetMiddleware(max_steps=3)
        msgs = [self._ai_with_tools() for _ in range(3)]
        mw.before_model({"messages": msgs})
        assert mw.exhausted is True

    def test_exhausted_urgency_message_content(self):
        mw = BudgetMiddleware(max_steps=2)
        msgs = [self._ai_with_tools() for _ in range(2)]
        result = mw.before_model({"messages": msgs})
        last = result["messages"][-1]
        assert "BUDGET EXHAUSTED" in last.content
        assert "do NOT call any tool including think" in last.content

    def test_middleware_is_agentm_middleware(self):
        assert isinstance(BudgetMiddleware(10), AgentMMiddleware)


class TestDedupMiddleware:
    """DedupMiddleware wraps the existing DedupTracker hook."""

    def test_tracker_accessible(self):
        mw = DedupMiddleware()
        assert mw.tracker is not None
        assert mw.tracker.size == 0

    def test_middleware_is_agentm_middleware(self):
        assert isinstance(DedupMiddleware(), AgentMMiddleware)


# ---------------------------------------------------------------------------
# StorageBackend
# ---------------------------------------------------------------------------


class TestStorageBackendProtocol:
    """StorageBackend is runtime_checkable and satisfied by FilesystemBackend.

    Bug prevented: a backend that claims to implement the protocol but
    misses a method → runtime crash during knowledge I/O.
    """

    def test_filesystem_satisfies_protocol(self):
        backend = FilesystemBackend()
        assert isinstance(backend, StorageBackend)


# ---------------------------------------------------------------------------
# FilesystemBackend
# ---------------------------------------------------------------------------


class TestFilesystemBackend:
    """FilesystemBackend reads/writes files relative to root_dir."""

    def test_write_and_read(self, tmp_path: Path):
        backend = FilesystemBackend(tmp_path)
        backend.write("test.txt", "hello\nworld\n")
        content = backend.read("test.txt")
        assert content == "hello\nworld\n"

    def test_read_with_offset_and_limit(self, tmp_path: Path):
        backend = FilesystemBackend(tmp_path)
        backend.write("lines.txt", "a\nb\nc\nd\n")
        content = backend.read("lines.txt", offset=1, limit=2)
        assert content == "b\nc\n"

    def test_ls(self, tmp_path: Path):
        backend = FilesystemBackend(tmp_path)
        backend.write("a.txt", "")
        backend.write("b.txt", "")
        entries = backend.ls(".")
        assert "a.txt" in entries
        assert "b.txt" in entries

    def test_exists(self, tmp_path: Path):
        backend = FilesystemBackend(tmp_path)
        assert not backend.exists("nope.txt")
        backend.write("yes.txt", "")
        assert backend.exists("yes.txt")

    def test_mkdir(self, tmp_path: Path):
        backend = FilesystemBackend(tmp_path)
        backend.mkdir("sub/dir")
        assert (tmp_path / "sub" / "dir").is_dir()

    def test_glob(self, tmp_path: Path):
        backend = FilesystemBackend(tmp_path)
        backend.write("a.py", "")
        backend.write("b.txt", "")
        matches = backend.glob("*.py")
        assert any("a.py" in m for m in matches)
        assert not any("b.txt" in m for m in matches)

    def test_grep(self, tmp_path: Path):
        backend = FilesystemBackend(tmp_path)
        backend.write("code.py", "def hello():\n    pass\n")
        results = backend.grep("hello")
        assert len(results) == 1
        assert results[0]["line"] == 1

    def test_path_traversal_blocked(self, tmp_path: Path):
        backend = FilesystemBackend(tmp_path)
        with pytest.raises(ValueError, match="outside root"):
            backend.read("../../etc/passwd")


# ---------------------------------------------------------------------------
# CompositeBackend
# ---------------------------------------------------------------------------


class TestCompositeBackend:
    """CompositeBackend routes operations by path prefix.

    Bug prevented: prefix matching is too greedy (/know matches /knowledge)
    or too loose (default backend never reached).
    """

    def test_routes_to_mounted_backend(self, tmp_path: Path):
        default_dir = tmp_path / "default"
        default_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        default = FilesystemBackend(default_dir)
        knowledge = FilesystemBackend(knowledge_dir)

        composite = CompositeBackend(default=default)
        composite.mount("/knowledge", knowledge)

        composite.write("/knowledge/entry.json", '{"key": "value"}')

        # Should be in the knowledge backend's root
        assert (knowledge_dir / "entry.json").exists()
        assert not (default_dir / "knowledge" / "entry.json").exists()

    def test_falls_through_to_default(self, tmp_path: Path):
        default_dir = tmp_path / "default"
        default_dir.mkdir()

        default = FilesystemBackend(default_dir)
        composite = CompositeBackend(default=default)

        composite.write("other.txt", "content")
        assert (default_dir / "other.txt").exists()

    def test_longest_prefix_wins(self, tmp_path: Path):
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        dir_b = tmp_path / "b"
        dir_b.mkdir()

        composite = CompositeBackend(default=FilesystemBackend(tmp_path))
        composite.mount("/data", FilesystemBackend(dir_a))
        composite.mount("/data/special", FilesystemBackend(dir_b))

        composite.write("/data/special/file.txt", "special")
        assert (dir_b / "file.txt").exists()
        assert not (dir_a / "special" / "file.txt").exists()


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------


class TestTaskResult:
    """TaskResult[R] is a frozen generic dataclass for sub-agent results.

    Bug prevented: mutable result dict shared between tasks → one task's
    modification corrupts another's result.
    """

    def test_frozen(self):
        result = TaskResult(task_id="t1", agent_id="a1", status="completed")
        with pytest.raises(AttributeError):
            result.status = "failed"  # type: ignore[misc]

    def test_typed_result(self):
        result = TaskResult[dict](
            task_id="t1",
            agent_id="a1",
            status="completed",
            result={"findings": "root cause found"},
        )
        assert result.result == {"findings": "root cause found"}

    def test_none_result_by_default(self):
        result = TaskResult(task_id="t1", agent_id="a1", status="failed")
        assert result.result is None

    def test_metadata_independent(self):
        r1 = TaskResult(task_id="t1", agent_id="a1", status="completed")
        r2 = TaskResult(task_id="t2", agent_id="a2", status="completed")
        # frozen=True + default_factory means each gets its own dict
        assert r1.metadata is not r2.metadata


# ---------------------------------------------------------------------------
# TypeVar S
# ---------------------------------------------------------------------------


class TestStateTypeVar:
    """S = TypeVar('S', bound=BaseExecutorState) is available for generic use.

    Bug prevented: TypeVar not exported → framework components can't be
    parameterized over user-defined state types.
    """

    def test_typevar_bound(self):
        assert S.__bound__ is BaseExecutorState

    def test_base_executor_state_current_phase_is_str(self):
        # current_phase changed from Phase enum to str in Phase 1
        annotations = BaseExecutorState.__annotations__
        # With from __future__ import annotations, the annotation is a string
        assert "current_phase" in annotations


# ---------------------------------------------------------------------------
# GenericAgentSystemBuilder
# ---------------------------------------------------------------------------


class TestGenericAgentSystemBuilder:
    """GenericAgentSystemBuilder requires strategy and scenario.

    Bug prevented: builder missing strategy → None attribute access during
    graph compilation.
    """

    def test_build_without_strategy_raises(self):
        from agentm.builder import GenericAgentSystemBuilder
        from agentm.scenarios.rca.state import HypothesisDrivenState

        builder = GenericAgentSystemBuilder(HypothesisDrivenState)
        with pytest.raises(ValueError, match="Strategy is required"):
            builder.build()

    def test_build_without_scenario_raises(self):
        from agentm.builder import GenericAgentSystemBuilder
        from agentm.scenarios.rca.state import HypothesisDrivenState
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        builder = GenericAgentSystemBuilder(HypothesisDrivenState)
        builder.with_strategy(HypothesisDrivenStrategy())
        with pytest.raises(ValueError, match="Scenario config is required"):
            builder.build()

    def test_fluent_api_returns_self(self):
        from agentm.builder import GenericAgentSystemBuilder
        from agentm.scenarios.rca.state import HypothesisDrivenState
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        builder = GenericAgentSystemBuilder(HypothesisDrivenState)
        result = builder.with_strategy(HypothesisDrivenStrategy())
        assert result is builder
