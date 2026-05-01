"""Skill discovery and prompt formatting."""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentm.core.frontmatter import parse_frontmatter

_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_NAME_PATTERN = r"^[a-z0-9-]+$"


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    level: Literal["warning", "collision"]
    message: str
    path: str


@dataclass(frozen=True, slots=True)
class SkillRecord:
    name: str
    description: str
    file_path: str
    base_dir: str
    disable_model_invocation: bool
    source: str


def _normalize_path(raw_path: str, cwd: str) -> str:
    expanded = raw_path.strip()
    if expanded == "~":
        return str(Path.home())
    if expanded.startswith("~/"):
        return str(Path.home() / expanded[2:])
    if expanded.startswith("~"):
        return str(Path.home() / expanded[1:])
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(cwd, expanded))


def _validate_name(name: str, parent_dir_name: str) -> list[str]:
    issues: list[str] = []
    if name != parent_dir_name:
        issues.append(
            f'name "{name}" does not match parent directory "{parent_dir_name}"'
        )
    if len(name) > _MAX_NAME_LENGTH:
        issues.append(f"name exceeds {_MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.match(_NAME_PATTERN, name):
        issues.append(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
        )
    if name.startswith("-") or name.endswith("-"):
        issues.append("name must not start or end with a hyphen")
    if "--" in name:
        issues.append("name must not contain consecutive hyphens")
    return issues


def _validate_description(description: str | None) -> list[str]:
    issues: list[str] = []
    if description is None or description.strip() == "":
        issues.append("description is required")
    elif len(description) > _MAX_DESCRIPTION_LENGTH:
        issues.append(
            f"description exceeds {_MAX_DESCRIPTION_LENGTH} characters ({len(description)})"
        )
    return issues


def _parse_skill_file(
    file_path: str,
    source: str,
) -> tuple[SkillRecord | None, list[SkillDiagnostic]]:
    diagnostics: list[SkillDiagnostic] = []
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except OSError as exc:
        return None, [
            SkillDiagnostic(level="warning", message=str(exc), path=file_path)
        ]

    metadata, _body = parse_frontmatter(text)
    skill_dir = str(Path(file_path).parent)
    parent_dir_name = Path(skill_dir).name

    description_value = metadata.get("description")
    description = description_value if isinstance(description_value, str) else None
    for issue in _validate_description(description):
        diagnostics.append(
            SkillDiagnostic(level="warning", message=issue, path=file_path)
        )

    raw_name = metadata.get("name")
    name = raw_name if isinstance(raw_name, str) and raw_name else parent_dir_name
    for issue in _validate_name(name, parent_dir_name):
        diagnostics.append(
            SkillDiagnostic(level="warning", message=issue, path=file_path)
        )

    disable_value = metadata.get("disable-model-invocation")
    disable_model_invocation = disable_value is True
    if disable_value is not None and not isinstance(disable_value, bool):
        diagnostics.append(
            SkillDiagnostic(
                level="warning",
                message="disable-model-invocation must be a boolean",
                path=file_path,
            )
        )

    if description is None or description.strip() == "":
        return None, diagnostics

    return (
        SkillRecord(
            name=name,
            description=description,
            file_path=file_path,
            base_dir=skill_dir,
            disable_model_invocation=disable_model_invocation,
            source=source,
        ),
        diagnostics,
    )


def _load_skills_from_dir(
    directory: str,
    source: str,
    *,
    include_root_files: bool,
) -> tuple[list[SkillRecord], list[SkillDiagnostic]]:
    skills: list[SkillRecord] = []
    diagnostics: list[SkillDiagnostic] = []
    if not os.path.isdir(directory):
        return skills, diagnostics

    visited: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(
        directory,
        topdown=True,
        followlinks=True,
    ):
        real_dir = os.path.realpath(dirpath)
        if real_dir in visited:
            dirnames[:] = []
            continue
        visited.add(real_dir)

        dirnames[:] = sorted(dirnames)
        filenames = sorted(filenames)

        skill_path = os.path.join(dirpath, "SKILL.md")
        if "SKILL.md" in filenames and os.path.isfile(skill_path):
            skill, skill_diags = _parse_skill_file(os.path.abspath(skill_path), source)
            if skill is not None:
                skills.append(skill)
            diagnostics.extend(skill_diags)
            dirnames[:] = []
            continue

        if (
            not include_root_files
            or os.path.abspath(dirpath) != os.path.abspath(directory)
        ):
            continue

        for filename in filenames:
            if not filename.endswith(".md"):
                continue
            file_path = os.path.join(dirpath, filename)
            if not os.path.isfile(file_path):
                continue
            skill, skill_diags = _parse_skill_file(os.path.abspath(file_path), source)
            if skill is not None:
                skills.append(skill)
            diagnostics.extend(skill_diags)

    return skills, diagnostics


def load_skills(
    *,
    cwd: str,
    agent_dir: str,
    skill_paths: list[str] | tuple[str, ...] = (),
    include_defaults: bool = True,
) -> tuple[list[SkillRecord], list[SkillDiagnostic]]:
    discovered: list[SkillRecord] = []
    diagnostics: list[SkillDiagnostic] = []
    seen_real_files: set[str] = set()
    seen_names: dict[str, str] = {}

    def add_batch(
        batch_skills: list[SkillRecord], batch_diagnostics: list[SkillDiagnostic]
    ) -> None:
        diagnostics.extend(batch_diagnostics)
        for skill in batch_skills:
            real_path = os.path.realpath(skill.file_path)
            if real_path in seen_real_files:
                continue
            seen_real_files.add(real_path)
            existing_path = seen_names.get(skill.name)
            if existing_path is not None:
                diagnostics.append(
                    SkillDiagnostic(
                        level="collision",
                        message=(
                            f"skill name collision: {skill.name!r} already loaded from "
                            f"{existing_path}"
                        ),
                        path=skill.file_path,
                    )
                )
                continue
            seen_names[skill.name] = skill.file_path
            discovered.append(skill)

    if include_defaults:
        add_batch(
            *_load_skills_from_dir(
                os.path.join(agent_dir, "skills"),
                "user",
                include_root_files=True,
            )
        )
        add_batch(
            *_load_skills_from_dir(
                os.path.join(cwd, ".agentm", "skills"),
                "project",
                include_root_files=True,
            )
        )

    for raw_path in skill_paths:
        resolved_path = _normalize_path(raw_path, cwd)
        if os.path.isdir(resolved_path):
            add_batch(
                *_load_skills_from_dir(
                    resolved_path,
                    "path",
                    include_root_files=True,
                )
            )
            continue
        if os.path.isfile(resolved_path):
            skill, skill_diags = _parse_skill_file(
                os.path.abspath(resolved_path),
                "path",
            )
            add_batch([skill] if skill is not None else [], skill_diags)

    return discovered, diagnostics


def format_skills_for_prompt(skills: list[SkillRecord]) -> str:
    visible_skills = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible_skills:
        return ""

    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in visible_skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{html.escape(skill.name, quote=True)}</name>")
        lines.append(
            f"    <description>{html.escape(skill.description, quote=True)}</description>"
        )
        lines.append(
            f"    <location>{html.escape(skill.file_path, quote=True)}</location>"
        )
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)
