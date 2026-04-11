"""Focused regression tests for agent memory loading and injection."""
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


def test_get_memory_dir_maps_scope_to_expected_path(tmp_path: Path) -> None:
    config = _make_config(tmp_path, agent_identity="my-agent", scenario_name="rca")
    assert get_memory_dir(config, MemoryScope.AGENT) == tmp_path / "agent" / "my-agent"
    assert get_memory_dir(config, MemoryScope.SCENARIO) == tmp_path / "scenario" / "rca"
    assert get_memory_dir(config, MemoryScope.PROJECT) == tmp_path / "project"


def test_load_prompt_empty_when_no_entries(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_memory(get_memory_dir(config, MemoryScope.AGENT), "# Agent Memory: test-agent\n")
    assert load_agent_memory_prompt(config) == ""


def test_load_prompt_preserves_scope_order_and_entry_budget(tmp_path: Path) -> None:
    config = _make_config(tmp_path, max_prompt_entries=4)
    _write_memory(get_memory_dir(config, MemoryScope.PROJECT), "- p1\n- p2\n- p3\n")
    _write_memory(get_memory_dir(config, MemoryScope.SCENARIO), "- s1\n- s2\n")
    prompt = load_agent_memory_prompt(config)
    assert prompt.startswith("<agent_memory>")
    assert prompt.count("- p") == 3
    assert prompt.count("- s") == 1
    assert prompt.index('scope="project"') < prompt.index('scope="scenario"')


@pytest.mark.asyncio
async def test_middleware_injects_memory_into_existing_system_message(tmp_path: Path) -> None:
    config = _make_config(tmp_path, scopes=[MemoryScope.AGENT])
    _write_memory(get_memory_dir(config, MemoryScope.AGENT), "- always check pools\n")
    mw = MemoryMiddleware(config)
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "human", "content": "Hello"},
    ]
    result = await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]
    assert "<agent_memory>" in result[0]["content"]  # type: ignore[index,operator]
    assert "always check pools" in result[0]["content"]  # type: ignore[index,operator]


@pytest.mark.asyncio
async def test_middleware_creates_system_message_when_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path, scopes=[MemoryScope.AGENT])
    _write_memory(get_memory_dir(config, MemoryScope.AGENT), "- cache hint\n")
    mw = MemoryMiddleware(config)
    result = await mw.on_llm_start([{"role": "human", "content": "Hello"}], _make_ctx())  # type: ignore[arg-type]
    assert result[0]["role"] == "system"
    assert "<agent_memory>" in result[0]["content"]  # type: ignore[index,operator]


@pytest.mark.asyncio
async def test_mark_dirty_forces_prompt_reload(tmp_path: Path) -> None:
    config = _make_config(tmp_path, scopes=[MemoryScope.AGENT])
    agent_dir = get_memory_dir(config, MemoryScope.AGENT)
    _write_memory(agent_dir, "- original entry\n")
    mw = MemoryMiddleware(config)
    messages: list[dict[str, object]] = [{"role": "system", "content": "sys"}]

    first = await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]
    assert "original entry" in first[0]["content"]  # type: ignore[index,operator]
    _write_memory(agent_dir, "- updated entry\n")
    mw.mark_dirty()

    second = await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]
    assert "updated entry" in second[0]["content"]  # type: ignore[index,operator]
    assert "original entry" not in second[0]["content"]  # type: ignore[index,operator]


@pytest.mark.asyncio
async def test_clean_cache_reuses_previous_prompt_until_dirty(tmp_path: Path) -> None:
    config = _make_config(tmp_path, scopes=[MemoryScope.AGENT])
    agent_dir = get_memory_dir(config, MemoryScope.AGENT)
    _write_memory(agent_dir, "- cached entry\n")
    mw = MemoryMiddleware(config)
    messages: list[dict[str, object]] = [{"role": "system", "content": "sys"}]

    await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]
    _write_memory(agent_dir, "- new entry should not appear\n")
    reused = await mw.on_llm_start(messages, _make_ctx())  # type: ignore[arg-type]
    assert "cached entry" in reused[0]["content"]  # type: ignore[index,operator]
    assert "new entry should not appear" not in reused[0]["content"]  # type: ignore[index,operator]
