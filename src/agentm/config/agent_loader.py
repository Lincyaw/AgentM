"""Declarative agent definition loader.

Loads AgentDefinition from Markdown frontmatter files. Each file
describes a single agent: YAML frontmatter for metadata, body for
system prompt.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm.utils.frontmatter import parse_frontmatter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDefinition:
    """Complete, immutable definition of a single agent."""

    # --- Identity ---
    name: str
    description: str = ""

    # --- Behavior ---
    task_type: str | None = None
    system_prompt: str = ""

    # --- Model ---
    model: str = ""
    temperature: float = 0.0

    # --- Tools ---
    tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    include_think_tool: bool = True

    # --- Execution ---
    max_steps: int = 20
    timeout: int = 120
    tool_call_budget: int | None = None

    # --- Advanced ---
    skills: list[str] = field(default_factory=list)
    tool_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- Source metadata ---
    source_file: Path | None = None


# Fields accepted from frontmatter (used to filter unknown keys).
_AGENT_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(AgentDefinition) if f.name != "source_file"
)


def parse_agent_markdown(path: Path) -> AgentDefinition:
    """Parse a Markdown file with YAML frontmatter into an AgentDefinition.

    The file format is:
        ---
        name: my-agent
        description: ...
        tools: [a, b]
        ---
        System prompt body text...

    Frontmatter must contain ``name``. Unknown keys are silently ignored.
    The body after the closing ``---`` becomes ``system_prompt``.

    Args:
        path: Path to the ``.md`` file.

    Returns:
        A frozen AgentDefinition instance.

    Raises:
        ValueError: If ``name`` is missing from frontmatter.
        yaml.YAMLError: If frontmatter is not valid YAML.
    """
    text = path.read_text(encoding="utf-8")
    raw, body = parse_frontmatter(text)

    if "name" not in raw:
        raise ValueError(
            f"Agent definition file is missing required 'name' field: {path}"
        )

    # Warn about unknown frontmatter keys (typo detection).
    unknown_keys = set(raw.keys()) - _AGENT_FIELDS
    if unknown_keys:
        logger.warning(
            "Agent '%s' has unknown frontmatter keys (ignored): %s",
            raw.get("name", path.stem),
            sorted(unknown_keys),
        )

    # Filter to known fields only (lenient: unknown keys dropped with warning).
    known: dict[str, Any] = {k: v for k, v in raw.items() if k in _AGENT_FIELDS}

    known["system_prompt"] = body.strip()
    known["source_file"] = path

    return AgentDefinition(**known)


def load_agent_definitions(scenario_dir: Path) -> dict[str, AgentDefinition]:
    """Load all agent definitions from ``scenario_dir/agents/*.md``.

    Args:
        scenario_dir: Root directory of a scenario.

    Returns:
        A dict mapping agent name to its definition.
        Returns an empty dict if the ``agents/`` subdirectory does not exist.
    """
    agents_dir = scenario_dir / "agents"
    if not agents_dir.is_dir():
        return {}

    definitions: dict[str, AgentDefinition] = {}
    for md_file in sorted(agents_dir.glob("*.md")):
        defn = parse_agent_markdown(md_file)
        definitions[defn.name] = defn

    return definitions


