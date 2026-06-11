"""Markdown frontmatter parsers shared by Claude Code compatibility atoms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.core.abi import SkillRecord
from agentm.core.lib import parse_frontmatter

@dataclass(frozen=True, slots=True)
class AgentRecord:
    name: str
    description: str
    file_path: str
    tools: str
    model: str

@dataclass(frozen=True, slots=True)
class CommandRecord:
    skill: SkillRecord
    body: str

def _read_markdown(md_path: Path) -> tuple[dict[str, Any], str] | None:
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return parse_frontmatter(text)

def parse_md_skill_records(dir: Path) -> list[SkillRecord]:
    """Parse ``*.md`` files in ``dir`` into model-invocable skill records."""

    if not dir.is_dir():
        return []
    records: list[SkillRecord] = []
    seen: set[str] = set()
    for md_path in sorted(dir.glob("*.md")):
        parsed = _read_markdown(md_path)
        if parsed is None:
            continue
        metadata, _body = parsed
        raw_name = metadata.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else md_path.stem
        if name in seen:
            continue
        raw_description = metadata.get("description")
        description = (
            raw_description.strip()
            if isinstance(raw_description, str) and raw_description.strip()
            else f"Claude command from {md_path}"
        )
        seen.add(name)
        records.append(
            SkillRecord(
                name=name,
                description=description,
                file_path=str(md_path.resolve()),
                base_dir=str(md_path.parent.resolve()),
                disable_model_invocation=False,
                source="claude-command",
            )
        )
    return records

def parse_md_command_records(dir: Path) -> list[CommandRecord]:
    """Parse command markdown files into slash-command bodies plus skills."""

    skills_by_path = {record.file_path: record for record in parse_md_skill_records(dir)}
    records: list[CommandRecord] = []
    for resolved_path, skill in skills_by_path.items():
        parsed = _read_markdown(Path(resolved_path))
        if parsed is None:
            continue
        _metadata, body = parsed
        if body.strip():
            records.append(CommandRecord(skill=skill, body=body))
    return records

def parse_md_agent_records(dir: Path) -> list[AgentRecord]:
    """Parse Claude Code persona markdown files from one agent directory."""

    if not dir.is_dir():
        return []
    records: list[AgentRecord] = []
    seen: set[str] = set()
    for md_path in sorted(dir.glob("*.md")):
        parsed = _read_markdown(md_path)
        if parsed is None:
            continue
        metadata, _body = parsed
        raw_name = metadata.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else md_path.stem
        if name in seen:
            continue
        raw_description = metadata.get("description")
        if not isinstance(raw_description, str) or not raw_description.strip():
            continue
        raw_tools = metadata.get("tools", "")
        raw_model = metadata.get("model", "")
        seen.add(name)
        records.append(
            AgentRecord(
                name=name,
                description=raw_description.strip(),
                file_path=str(md_path.resolve()),
                tools=raw_tools.strip() if isinstance(raw_tools, str) else "",
                model=raw_model.strip() if isinstance(raw_model, str) else "",
            )
        )
    return records

__all__ = (
    "AgentRecord",
    "CommandRecord",
    "parse_md_agent_records",
    "parse_md_command_records",
    "parse_md_skill_records",
)
