"""Tests for the generic SDK architecture: scenario registry,
middleware composition, storage backend, composite backend, task result, and
generic builder.

Ref: designs/generic-state-wrapper.md, designs/sdk-consistency.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentm.backends.composite import CompositeBackend
from agentm.backends.filesystem import FilesystemBackend
from agentm.core.backend import StorageBackend
from agentm.models.state import BaseExecutorState

from agentm.scenarios import discover

# Ensure scenario registrations are loaded for all tests in this module.
discover()


# ---------------------------------------------------------------------------
# Scenario Registry (replaces old Strategy Registry)
# ---------------------------------------------------------------------------


class TestScenarioRegistry:
    """ScenarioRegistry maps scenario names to Scenario instances.

    Bug prevented: typo in scenario_name silently returns None -> NoneType
    attribute errors deep in the execution pipeline.
    """

    def test_get_hypothesis_driven(self):
        from agentm.harness.scenario import get_scenario

        scenario = get_scenario("hypothesis_driven")
        assert scenario.name == "hypothesis_driven"

    def test_get_trajectory_analysis(self):
        from agentm.harness.scenario import get_scenario

        scenario = get_scenario("trajectory_analysis")
        assert scenario.name == "trajectory_analysis"

    def test_get_general_purpose(self):
        from agentm.harness.scenario import get_scenario

        scenario = get_scenario("general_purpose")
        assert scenario.name == "general_purpose"

    def test_unknown_type_raises(self):
        from agentm.harness.scenario import get_scenario

        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("unknown_type_xyz")

    def test_list_scenarios_returns_registered(self):
        from agentm.harness.scenario import list_scenarios

        names = list_scenarios()
        assert "hypothesis_driven" in names
        assert "trajectory_analysis" in names
        assert "general_purpose" in names


# ---------------------------------------------------------------------------
# Hypothesis-Driven Strategy
# ---------------------------------------------------------------------------


class TestRCAScenario:
    """RCAScenario setup produces correct wiring."""

    def test_setup_returns_orchestrator_tools(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        tool_names = {t.name for t in wiring.orchestrator_tools}
        assert "update_hypothesis" in tool_names
        assert "remove_hypothesis" in tool_names

    def test_setup_returns_worker_tools(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert len(wiring.worker_tools) > 0

    def test_setup_returns_answer_schemas(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert "scout" in wiring.answer_schemas
        assert "deep_analyze" in wiring.answer_schemas
        assert "verify" in wiring.answer_schemas

    def test_setup_returns_output_schema(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert wiring.output_schema is not None

    def test_format_context_is_callable(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        result = wiring.format_context()
        assert isinstance(result, str)

    def test_hooks_configured(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert wiring.hooks.think_stall_enabled is True
        assert wiring.hooks.think_stall_limit == 3


# ---------------------------------------------------------------------------
# Trajectory-Analysis Scenario
# ---------------------------------------------------------------------------


class TestTrajectoryAnalysisScenario:
    """TrajectoryAnalysisScenario setup produces correct wiring."""

    def test_setup_returns_answer_schemas(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.trajectory_analysis.scenario import TrajectoryAnalysisScenario

        wiring = TrajectoryAnalysisScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert "analyze" in wiring.answer_schemas

    def test_setup_returns_output_schema(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.trajectory_analysis.scenario import TrajectoryAnalysisScenario

        wiring = TrajectoryAnalysisScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert wiring.output_schema is not None

    def test_hooks_are_default(self):
        from agentm.harness.scenario import SetupContext
        from agentm.models.data import OrchestratorHooks
        from agentm.scenarios.trajectory_analysis.scenario import TrajectoryAnalysisScenario

        wiring = TrajectoryAnalysisScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        default_hooks = OrchestratorHooks()
        assert wiring.hooks.think_stall_enabled == default_hooks.think_stall_enabled


# ---------------------------------------------------------------------------
# StorageBackend
# ---------------------------------------------------------------------------


class TestStorageBackendProtocol:
    """StorageBackend is runtime_checkable and satisfied by FilesystemBackend.

    Bug prevented: a backend that claims to implement the protocol but
    misses a method -> runtime crash during knowledge I/O.
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
# build_agent_system -- unknown scenario raises ValueError
# ---------------------------------------------------------------------------


class TestBuildAgentSystemValidation:
    """build_agent_system raises on unknown scenario names.

    Bug prevented: typo in scenario_name silently creates a broken system.
    """

    def test_unknown_scenario_raises(self):
        from agentm.builder import build_agent_system
        from agentm.config.schema import (
            OrchestratorConfig,
            ScenarioConfig,
            SystemTypeConfig,
        )

        config = ScenarioConfig(
            system=SystemTypeConfig(type="nonexistent"),
            orchestrator=OrchestratorConfig(model="gpt-4o", temperature=0.7, tools=[]),
            agents={},
        )
        with pytest.raises(ValueError, match="Unknown scenario"):
            build_agent_system("nonexistent_scenario_xyz", config)
