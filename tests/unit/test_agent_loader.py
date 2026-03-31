"""Tests for declarative agent definition loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm.config.agent_loader import (
    AgentDefinition,
    load_agent_definitions,
    parse_agent_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_md(path: Path, content: str) -> Path:
    """Write content to a .md file and return its path."""
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# parse_agent_markdown — happy path
# ---------------------------------------------------------------------------

class TestParseAgentMarkdown:
    """Tests for parse_agent_markdown."""

    def test_parses_frontmatter_and_body(self, tmp_path: Path) -> None:
        """Frontmatter fields are mapped and body becomes system_prompt."""
        md = _write_md(
            tmp_path / "scout.md",
            """\
---
name: scout
description: Scout agent
task_type: read
model: gpt-4
temperature: 0.5
tools: [search, read_file]
max_steps: 30
timeout: 60
---
You are a scout agent.

## Instructions
Search for clues.
""",
        )

        defn = parse_agent_markdown(md)

        assert defn.name == "scout"
        assert defn.description == "Scout agent"
        assert defn.task_type == "read"
        assert defn.model == "gpt-4"
        assert defn.temperature == 0.5
        assert defn.tools == ["search", "read_file"]
        assert defn.max_steps == 30
        assert defn.timeout == 60
        assert "You are a scout agent." in defn.system_prompt
        assert "## Instructions" in defn.system_prompt
        assert defn.source_file == md

    def test_missing_name_raises_value_error(self, tmp_path: Path) -> None:
        """Frontmatter without 'name' must raise ValueError."""
        md = _write_md(
            tmp_path / "bad.md",
            """\
---
description: No name here
---
Body text.
""",
        )

        with pytest.raises(ValueError, match="missing required 'name'"):
            parse_agent_markdown(md)

    def test_unknown_frontmatter_keys_ignored(self, tmp_path: Path) -> None:
        """Unknown keys in frontmatter are silently dropped."""
        md = _write_md(
            tmp_path / "agent.md",
            """\
---
name: tester
unknown_key: should_be_ignored
another_unknown: 42
---
Prompt body.
""",
        )

        defn = parse_agent_markdown(md)

        assert defn.name == "tester"
        assert defn.system_prompt == "Prompt body."
        # Verify no extra attributes leaked through
        assert not hasattr(defn, "unknown_key")
        assert not hasattr(defn, "another_unknown")

    def test_default_values_for_optional_fields(self, tmp_path: Path) -> None:
        """When only 'name' is provided, all other fields have correct defaults."""
        md = _write_md(
            tmp_path / "minimal.md",
            """\
---
name: minimal-agent
---
""",
        )

        defn = parse_agent_markdown(md)

        assert defn.name == "minimal-agent"
        assert defn.description == ""
        assert defn.task_type is None
        assert defn.system_prompt == ""
        assert defn.model == ""
        assert defn.temperature == 0.0
        assert defn.tools == []
        assert defn.disallowed_tools == []
        assert defn.include_think_tool is True
        assert defn.max_steps == 20
        assert defn.timeout == 120
        assert defn.tool_call_budget is None
        assert defn.skills == []
        assert defn.tool_settings == {}
        assert defn.source_file == md

    def test_all_fields_from_frontmatter(self, tmp_path: Path) -> None:
        """Every AgentDefinition field can be set via frontmatter."""
        md = _write_md(
            tmp_path / "full.md",
            """\
---
name: full-agent
description: Full featured
task_type: analyze
model: claude-3
temperature: 0.8
tools: [tool_a]
disallowed_tools: [tool_b]
include_think_tool: false
max_steps: 50
timeout: 300
tool_call_budget: 100
skills: [skill/advanced]
tool_settings:
  tool_a:
    param1: value1
---
Full prompt.
""",
        )

        defn = parse_agent_markdown(md)

        assert defn.name == "full-agent"
        assert defn.description == "Full featured"
        assert defn.task_type == "analyze"
        assert defn.model == "claude-3"
        assert defn.temperature == 0.8
        assert defn.tools == ["tool_a"]
        assert defn.disallowed_tools == ["tool_b"]
        assert defn.include_think_tool is False
        assert defn.max_steps == 50
        assert defn.timeout == 300
        assert defn.tool_call_budget == 100
        assert defn.skills == ["skill/advanced"]
        assert defn.tool_settings == {"tool_a": {"param1": "value1"}}
        assert defn.system_prompt == "Full prompt."


# ---------------------------------------------------------------------------
# load_agent_definitions
# ---------------------------------------------------------------------------

class TestLoadAgentDefinitions:
    """Tests for load_agent_definitions."""

    def test_loads_multiple_files_from_agents_dir(self, tmp_path: Path) -> None:
        """All .md files in agents/ directory are loaded."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        _write_md(
            agents_dir / "alpha.md",
            """\
---
name: alpha
description: First agent
---
Alpha prompt.
""",
        )
        _write_md(
            agents_dir / "beta.md",
            """\
---
name: beta
description: Second agent
---
Beta prompt.
""",
        )

        result = load_agent_definitions(tmp_path)

        assert len(result) == 2
        assert "alpha" in result
        assert "beta" in result
        assert result["alpha"].description == "First agent"
        assert result["beta"].system_prompt == "Beta prompt."

    def test_returns_empty_dict_when_agents_dir_missing(
        self, tmp_path: Path
    ) -> None:
        """When agents/ directory does not exist, return empty dict."""
        result = load_agent_definitions(tmp_path)

        assert result == {}


# ---------------------------------------------------------------------------
# AgentDefinition immutability
# ---------------------------------------------------------------------------

class TestAgentDefinitionImmutability:
    """Verify AgentDefinition is frozen (immutable)."""

    def test_frozen_raises_on_attribute_set(self) -> None:
        """Attempting to set an attribute on a frozen dataclass raises."""
        defn = AgentDefinition(name="test")

        with pytest.raises(AttributeError):
            defn.name = "changed"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            defn.temperature = 1.0  # type: ignore[misc]
