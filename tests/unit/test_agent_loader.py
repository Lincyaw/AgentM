"""Focused regression tests for declarative agent definition loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm.config.agent_loader import AgentDefinition, load_agent_definitions, parse_agent_markdown


def _write_md(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_markdown_maps_frontmatter_and_body(tmp_path: Path) -> None:
    md = _write_md(
        tmp_path / "agent_loader_test.md",
        """\
---
name: scout
description: Scout agent
task_type: read
model: gpt-4
tools: [search, read_file]
max_steps: 30
---
You are a scout agent.
""",
    )
    defn = parse_agent_markdown(md)
    assert defn.name == "scout"
    assert defn.task_type == "read"
    assert defn.tools == ["search", "read_file"]
    assert "You are a scout agent." in defn.system_prompt


def test_parse_markdown_missing_name_raises(tmp_path: Path) -> None:
    md = _write_md(tmp_path / "bad.md", "---\ndescription: x\n---\nBody")
    with pytest.raises(ValueError, match="missing required 'name'"):
        parse_agent_markdown(md)


def test_load_agent_definitions_loads_all_md_files(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    _write_md(agents_dir / "a.md", "---\nname: alpha\n---\nA")
    _write_md(agents_dir / "b.md", "---\nname: beta\n---\nB")
    result = load_agent_definitions(tmp_path)
    assert set(result) == {"alpha", "beta"}


def test_load_agent_definitions_returns_empty_when_missing_dir(tmp_path: Path) -> None:
    assert load_agent_definitions(tmp_path) == {}


def test_agent_definition_is_frozen() -> None:
    defn = AgentDefinition(name="test")
    with pytest.raises(AttributeError):
        defn.name = "changed"  # type: ignore[misc]
