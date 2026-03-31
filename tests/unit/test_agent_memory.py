"""Tests for the Agent Memory Scope module.

Ref: designs/agent-memory.md

Tests memory directory resolution, prompt construction (with and without
MEMORY.md content), entry truncation, middleware injection, and dirty-flag
caching.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agentm.harness.agent_memory import (
    AgentMemoryConfig,
    MemoryMiddleware,
    MemoryScope,
    get_memory_dir,
    load_agent_memory_prompt,
)
from agentm.harness.types import LoopContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    tmp_path: Path,
    *,
    agent_identity: str = "test-agent",
    scenario_name: str = "test-scenario",
    scopes: list[MemoryScope] | None = None,
    max_prompt_entries: int = 50,
) -> AgentMemoryConfig:
    return AgentMemoryConfig(
        enabled=True,
        memory_root=str(tmp_path),
        scopes=scopes or [MemoryScope.AGENT, MemoryScope.SCENARIO, MemoryScope.PROJECT],
        max_prompt_entries=max_prompt_entries,
        agent_identity=agent_identity,
        scenario_name=scenario_name,
    )


def _write_memory(base_dir: Path, content: str) -> None:
    """Write a MEMORY.md file under the given directory."""
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "MEMORY.md").write_text(content, encoding="utf-8")


def _make_ctx() -> LoopContext:
    return LoopContext(
        agent_id="test-agent",
        step=0,
        max_steps=30,
        tool_call_count=0,
        metadata={},
    )


# ---------------------------------------------------------------------------
# get_memory_dir — correct path for each scope
# ---------------------------------------------------------------------------


class TestGetMemoryDir:
    """Bug: wrong directory mapping causes one scope to overwrite another."""

    def test_agent_scope_path(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, agent_identity="my-agent")
        result = get_memory_dir(config, MemoryScope.AGENT)
        assert result == Path(str(tmp_path)) / "agent" / "my-agent"

    def test_scenario_scope_path(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, scenario_name="rca")
        result = get_memory_dir(config, MemoryScope.SCENARIO)
        assert result == Path(str(tmp_path)) / "scenario" / "rca"

    def test_project_scope_path(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        result = get_memory_dir(config, MemoryScope.PROJECT)
        assert result == Path(str(tmp_path)) / "project"


# ---------------------------------------------------------------------------
# load_agent_memory_prompt — empty when no MEMORY.md exists
# ---------------------------------------------------------------------------


class TestLoadPromptEmpty:
    """Bug: returning non-empty prompt for missing files wastes context window."""

    def test_no_memory_files_returns_empty(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert load_agent_memory_prompt(config) == ""

    def test_empty_memory_files_returns_empty(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        # Create empty MEMORY.md files
        for scope in (MemoryScope.AGENT, MemoryScope.SCENARIO, MemoryScope.PROJECT):
            _write_memory(get_memory_dir(config, scope), "")
        assert load_agent_memory_prompt(config) == ""

    def test_memory_with_only_header_no_entries_returns_empty(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _write_memory(
            get_memory_dir(config, MemoryScope.AGENT),
            "# Agent Memory: test-agent\n\nNo entries yet.\n",
        )
        assert load_agent_memory_prompt(config) == ""


# ---------------------------------------------------------------------------
# load_agent_memory_prompt — correct XML format with content
# ---------------------------------------------------------------------------


class TestLoadPromptFormat:
    """Bug: malformed XML tags cause LLM to misinterpret memory blocks."""

    def test_single_scope_has_xml_tags(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, scopes=[MemoryScope.AGENT])
        _write_memory(
            get_memory_dir(config, MemoryScope.AGENT),
            "- [Check Pools](notes/pools.md) -- always check pool metrics\n",
        )
        prompt = load_agent_memory_prompt(config)
        assert prompt.startswith("<agent_memory>")
        assert prompt.endswith("</agent_memory>")
        assert '<memory scope="agent" name="test-agent">' in prompt
        assert "always check pool metrics" in prompt

    def test_multi_scope_ordering(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _write_memory(
            get_memory_dir(config, MemoryScope.PROJECT),
            "- project-level entry\n",
        )
        _write_memory(
            get_memory_dir(config, MemoryScope.SCENARIO),
            "- scenario-level entry\n",
        )
        _write_memory(
            get_memory_dir(config, MemoryScope.AGENT),
            "- agent-level entry\n",
        )
        prompt = load_agent_memory_prompt(config)

        # PROJECT should appear before SCENARIO, SCENARIO before AGENT
        proj_pos = prompt.index('scope="project"')
        scen_pos = prompt.index('scope="scenario"')
        agent_pos = prompt.index('scope="agent"')
        assert proj_pos < scen_pos < agent_pos

    def test_scenario_tag_includes_name(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, scenario_name="rca", scopes=[MemoryScope.SCENARIO])
        _write_memory(
            get_memory_dir(config, MemoryScope.SCENARIO),
            "- cascading failures pattern\n",
        )
        prompt = load_agent_memory_prompt(config)
        assert '<memory scope="scenario" name="rca">' in prompt


# ---------------------------------------------------------------------------
# load_agent_memory_prompt — truncation at max_prompt_entries
# ---------------------------------------------------------------------------


class TestLoadPromptTruncation:
    """Bug: unbounded entries blow up prompt size and exceed context window."""

    def test_entries_truncated_at_max(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, max_prompt_entries=3, scopes=[MemoryScope.AGENT])
        lines = "\n".join(f"- entry {i}" for i in range(10))
        _write_memory(get_memory_dir(config, MemoryScope.AGENT), lines)

        prompt = load_agent_memory_prompt(config)
        # Should contain exactly 3 entries
        assert prompt.count("- entry") == 3
        assert "- entry 0" in prompt
        assert "- entry 2" in prompt
        assert "- entry 3" not in prompt

    def test_truncation_across_scopes(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, max_prompt_entries=4)

        # 3 project entries + 3 scenario entries = 6 total, but max is 4
        proj_lines = "\n".join(f"- proj-entry-{i}" for i in range(3))
        scen_lines = "\n".join(f"- scen-entry-{i}" for i in range(3))
        _write_memory(get_memory_dir(config, MemoryScope.PROJECT), proj_lines)
        _write_memory(get_memory_dir(config, MemoryScope.SCENARIO), scen_lines)

        prompt = load_agent_memory_prompt(config)
        # PROJECT gets 3, SCENARIO gets remaining 1
        assert prompt.count("- proj-entry") == 3
        assert prompt.count("- scen-entry") == 1


# ---------------------------------------------------------------------------
# MemoryMiddleware.on_llm_start — prompt injection
# ---------------------------------------------------------------------------


class TestMemoryMiddlewareInjection:
    """Bug: memory not appended to system message -> agent ignores past learnings."""

    @pytest.mark.asyncio
    async def test_appends_to_existing_system_message(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, scopes=[MemoryScope.AGENT])
        _write_memory(
            get_memory_dir(config, MemoryScope.AGENT),
            "- always check pools\n",
        )
        mw = MemoryMiddleware(config)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": "Hello"},
        ]

        result = await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]

        assert len(result) == 2
        system_msg = result[0]
        assert isinstance(system_msg, dict)
        assert "You are a helpful assistant." in system_msg["content"]  # type: ignore[operator]
        assert "<agent_memory>" in system_msg["content"]  # type: ignore[operator]
        assert "always check pools" in system_msg["content"]  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_inserts_system_message_when_none_exists(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, scopes=[MemoryScope.AGENT])
        _write_memory(
            get_memory_dir(config, MemoryScope.AGENT),
            "- always check pools\n",
        )
        mw = MemoryMiddleware(config)
        messages: list[dict[str, object]] = [
            {"role": "human", "content": "Hello"},
        ]

        result = await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]

        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert result[0]["role"] == "system"
        assert "<agent_memory>" in result[0]["content"]  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_no_injection_when_no_memory(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        mw = MemoryMiddleware(config)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        result = await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]

        assert len(result) == 1
        assert result[0] is messages[0]  # unchanged


# ---------------------------------------------------------------------------
# MemoryMiddleware — dirty flag caching
# ---------------------------------------------------------------------------


class TestMemoryMiddlewareDirtyFlag:
    """Bug: reading MEMORY.md on every LLM call wastes I/O; stale cache misses updates."""

    def test_initially_dirty(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        mw = MemoryMiddleware(config)
        assert mw.dirty is True

    @pytest.mark.asyncio
    async def test_clean_after_first_call(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, scopes=[MemoryScope.AGENT])
        _write_memory(
            get_memory_dir(config, MemoryScope.AGENT),
            "- some entry\n",
        )
        mw = MemoryMiddleware(config)
        messages: list[dict[str, object]] = [{"role": "system", "content": "sys"}]
        await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]

        assert mw.dirty is False

    @pytest.mark.asyncio
    async def test_mark_dirty_triggers_reload(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, scopes=[MemoryScope.AGENT])
        agent_dir = get_memory_dir(config, MemoryScope.AGENT)
        _write_memory(agent_dir, "- original entry\n")

        mw = MemoryMiddleware(config)
        messages: list[dict[str, object]] = [{"role": "system", "content": "sys"}]

        # First call loads original
        result1 = await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]
        assert "original entry" in result1[0]["content"]  # type: ignore[index,operator]
        assert mw.dirty is False

        # Update the file and mark dirty
        _write_memory(agent_dir, "- updated entry\n")
        mw.mark_dirty()
        assert mw.dirty is True

        # Second call should pick up the new content
        result2 = await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]
        assert "updated entry" in result2[0]["content"]  # type: ignore[index,operator]
        assert "original entry" not in result2[0]["content"]  # type: ignore[index,operator]

    @pytest.mark.asyncio
    async def test_cached_prompt_reused_when_clean(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, scopes=[MemoryScope.AGENT])
        agent_dir = get_memory_dir(config, MemoryScope.AGENT)
        _write_memory(agent_dir, "- cached entry\n")

        mw = MemoryMiddleware(config)
        messages: list[dict[str, object]] = [{"role": "system", "content": "sys"}]

        # First call
        await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]

        # Modify file but do NOT mark dirty
        _write_memory(agent_dir, "- new entry that should not appear\n")

        # Second call should still use cached (old) content
        result = await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]
        assert "cached entry" in result[0]["content"]  # type: ignore[index,operator]
        assert "new entry that should not appear" not in result[0]["content"]  # type: ignore[index,operator]
