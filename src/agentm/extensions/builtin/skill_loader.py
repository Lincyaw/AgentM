"""Builtin skill loader atom.

Discovers SKILL.md files (default agent dir, project dirs supplied by the
:class:`agentm.core.abi.project_layout.ProjectLayout`, explicit paths, peer
contributions via ``ResourcesDiscoverEvent``) and injects an
``<available_skills>`` index into the system prompt.

The skill discovery + prompt formatting engine is inlined below; it
previously lived in ``core/_internal/skills.py`` and was reached via
``api.skills``. Now that the only consumer is this atom, the indirection
adds nothing.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any

from agentm.core.abi import (
    BeforeAgentStartEvent,
    DiagnosticEvent,
    ExtensionAPI,
    FunctionTool,
    ResourcesDiscoverEvent,
    SessionReadyEvent,
    SkillDiagnostic,
    SkillRecord,
    TextContent,
    ToolResult,
)
from pydantic import BaseModel, Field

from agentm.core.lib import agentm_home_dir, parse_frontmatter
from agentm.core.lib import pydantic_to_tool_schema
from agentm.extensions import ExtensionManifest

class SkillLoaderConfig(BaseModel):
    skill_paths: list[str] = []
    include_defaults: bool = True
    inherit_claude: bool | None = None

MANIFEST = ExtensionManifest(
    name="skill_loader",
    description="Discover SKILL.md files and inject an <available_skills> index.",
    registers=(
        "tool:load_skill",
        "event:before_agent_start",
        "event:resources_discover",
        "event:session_ready",
    ),
    config_schema=SkillLoaderConfig,
    requires=(),  # Leaf atom: consumes resource-discovery responses from any peer.
)

# === Skills engine (inlined; previously core/_internal/skills.py) ==========

DEFAULT_MAX_NAME_LENGTH = 64
DEFAULT_MAX_DESCRIPTION_LENGTH = 1024
_NAME_PATTERN = r"^[a-z0-9-]+$"

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

def _validate_name(
    name: str, parent_dir_name: str, *, max_name_length: int
) -> list[str]:
    issues: list[str] = []
    if name != parent_dir_name:
        issues.append(
            f'name "{name}" does not match parent directory "{parent_dir_name}"'
        )
    if len(name) > max_name_length:
        issues.append(f"name exceeds {max_name_length} characters ({len(name)})")
    if not re.match(_NAME_PATTERN, name):
        issues.append(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
        )
    if name.startswith("-") or name.endswith("-"):
        issues.append("name must not start or end with a hyphen")
    if "--" in name:
        issues.append("name must not contain consecutive hyphens")
    return issues

def _validate_description(
    description: str | None, *, max_description_length: int
) -> list[str]:
    issues: list[str] = []
    if description is None or description.strip() == "":
        issues.append("description is required")
    elif len(description) > max_description_length:
        issues.append(
            f"description exceeds {max_description_length} characters "
            f"({len(description)})"
        )
    return issues

def _parse_skill_file(
    file_path: str,
    source: str,
    *,
    max_name_length: int,
    max_description_length: int,
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
    for issue in _validate_description(
        description, max_description_length=max_description_length
    ):
        diagnostics.append(
            SkillDiagnostic(level="warning", message=issue, path=file_path)
        )

    raw_name = metadata.get("name")
    name = raw_name if isinstance(raw_name, str) and raw_name else parent_dir_name
    for issue in _validate_name(
        name, parent_dir_name, max_name_length=max_name_length
    ):
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
    max_name_length: int,
    max_description_length: int,
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
            skill, skill_diags = _parse_skill_file(
                os.path.abspath(skill_path),
                source,
                max_name_length=max_name_length,
                max_description_length=max_description_length,
            )
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
            skill, skill_diags = _parse_skill_file(
                os.path.abspath(file_path),
                source,
                max_name_length=max_name_length,
                max_description_length=max_description_length,
            )
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
    project_skill_dirs: list[str] | tuple[str, ...] | None = None,
    max_name_length: int = DEFAULT_MAX_NAME_LENGTH,
    max_description_length: int = DEFAULT_MAX_DESCRIPTION_LENGTH,
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
                max_name_length=max_name_length,
                max_description_length=max_description_length,
            )
        )
        # Project-scope skill directories are policy, not kernel — they MUST be
        # supplied by the harness via ``ProjectLayout.skills_dirs()``. The
        # kernel no longer hard-codes ``<cwd>/.agentm/skills``; minimal/no-layout
        # sessions simply have no project skills.
        project_dirs: tuple[str, ...] = (
            tuple(project_skill_dirs) if project_skill_dirs is not None else ()
        )
        for project_dir in project_dirs:
            add_batch(
                *_load_skills_from_dir(
                    project_dir,
                    "project",
                    include_root_files=True,
                    max_name_length=max_name_length,
                    max_description_length=max_description_length,
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
                    max_name_length=max_name_length,
                    max_description_length=max_description_length,
                )
            )
            continue
        if os.path.isfile(resolved_path):
            skill, skill_diags = _parse_skill_file(
                os.path.abspath(resolved_path),
                "path",
                max_name_length=max_name_length,
                max_description_length=max_description_length,
            )
            add_batch([skill] if skill is not None else [], skill_diags)

    return discovered, diagnostics

def format_skills_for_prompt(skills: list[SkillRecord]) -> str:
    visible_skills = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible_skills:
        return ""

    lines = [
        "\n\n# Skills",
        "",
        "You have access to a skill system that extends your capabilities with "
        "specialized, up-to-date instructions for specific domains.",
        "Each skill below has a description of when it applies. When you receive "
        "a task that matches a skill's description, you MUST call the `load_skill` "
        "tool with that skill's name to read the full instructions BEFORE "
        "responding to the task.",
        "Do not answer from your own knowledge when a relevant skill is available "
        "— the skill may contain project-specific conventions, current procedures, "
        "or constraints you are not aware of.",
        "If no skill matches, proceed normally without loading any.",
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

# Tool schemas (Pydantic -> JSON Schema via pydantic_to_tool_schema)
# ---------------------------------------------------------------------------

class _LoadSkillParams(BaseModel):
    name: str = Field(description="Skill name from <available_skills>.")

# === Atom install ==========================================================

async def install(api: ExtensionAPI, config: SkillLoaderConfig) -> None:
    include_defaults = config.include_defaults
    # ``inherit_claude`` defaults to ``include_defaults`` — when callers turn
    # off the standard agentm defaults (e.g. test isolation) they should not
    # silently pick up the real user's ``~/.claude/skills`` either.
    inherit_claude = config.inherit_claude if config.inherit_claude is not None else include_defaults
    configured_skill_paths = list(config.skill_paths)
    cached_prompt_block = ""
    skills_by_name: dict[str, SkillRecord] = {}

    async def _populate(_: SessionReadyEvent) -> None:
        nonlocal cached_prompt_block
        discovered_paths = list(configured_skill_paths)
        if inherit_claude:
            # Auto-pick up Claude Code skill directories so users can reuse
            # the same `.claude/skills/<name>/SKILL.md` layout. Non-existent
            # paths are silently ignored by ``load_skills``.
            discovered_paths.append(str(Path.home() / ".claude" / "skills"))
            discovered_paths.append(str(Path(api.cwd) / ".claude" / "skills"))
            # Also walk Claude Code's installed-plugin skills:
            # ``~/.claude/plugins/cache/<source>/<plugin>/<version>/skills/``.
            # Plugins are how the bulk of Claude Code's library ships
            # (autoharness, workbuddy, codex, …) — ignoring them would
            # leave agentm with only the user's personal ``~/.claude/skills``.
            plugin_cache = Path.home() / ".claude" / "plugins" / "cache"
            if plugin_cache.is_dir():
                for source_dir in plugin_cache.iterdir():
                    if not source_dir.is_dir():
                        continue
                    for plugin_dir in source_dir.iterdir():
                        if not plugin_dir.is_dir():
                            continue
                        for version_dir in plugin_dir.iterdir():
                            if not version_dir.is_dir():
                                continue
                            skills_dir = version_dir / "skills"
                            if skills_dir.is_dir():
                                discovered_paths.append(str(skills_dir))
        response_owners: dict[int, str] = {}

        class _ResourceResponseObserver:
            def on_emit_start(self, channel: str, event: Any) -> None:
                del channel, event

            def on_handler_done(
                self,
                channel: str,
                handler: Any,
                event: Any,
                value: Any,
                err: BaseException | None,
                duration_ns: int,
                owner: str | None = None,
            ) -> None:
                del handler, event, err, duration_ns
                if channel == ResourcesDiscoverEvent.CHANNEL and isinstance(value, dict):
                    response_owners[id(value)] = owner or "<unknown>"

            def on_emit_end(
                self, channel: str, event: Any, results: list[Any]
            ) -> None:
                del channel, event, results

        unsubscribe = api.add_observer(_ResourceResponseObserver())
        try:
            responses = await api.events.emit(
                ResourcesDiscoverEvent.CHANNEL,
                ResourcesDiscoverEvent(cwd=api.cwd, reason="startup"),
            )
        finally:
            unsubscribe()
        contributed_skills: list[SkillRecord] = []
        allowed_response_keys = {"skill_paths", "extra_skills"}
        for response in responses:
            if not isinstance(response, dict):
                continue
            origin = response_owners.get(id(response), "<unknown>")
            for key in sorted(set(response) - allowed_response_keys):
                await api.events.emit(
                    DiagnosticEvent.CHANNEL,
                    DiagnosticEvent(
                        level="warning",
                        source="skill_loader",
                        message=(
                            f"ignored unknown ResourcesDiscoverEvent response key {key!r} "
                            f"from {origin}"
                        ),
                    ),
                )
            extra_paths = response.get("skill_paths")
            if isinstance(extra_paths, list):
                discovered_paths.extend(str(path) for path in extra_paths)
            extra_skills = response.get("extra_skills")
            if isinstance(extra_skills, list):
                for entry in extra_skills:
                    if isinstance(entry, SkillRecord):
                        contributed_skills.append(entry)

        # Resolve project-scope skill dirs from the harness-supplied layout
        # (previously the SkillsService injected this; now the atom does it
        # directly).
        layout = api.get_project_layout()
        project_dirs = tuple(str(p) for p in layout.skills_dirs())

        skills, _diagnostics = load_skills(
            cwd=api.cwd,
            agent_dir=str(agentm_home_dir()),
            skill_paths=tuple(discovered_paths),
            include_defaults=include_defaults,
            project_skill_dirs=project_dirs,
        )
        # Append peer-contributed records last so they don't shadow disk-based
        # skills with the same name.
        seen_names = {skill.name for skill in skills}
        for record in contributed_skills:
            if record.name in seen_names:
                continue
            seen_names.add(record.name)
            skills.append(record)
        cached_prompt_block = format_skills_for_prompt(skills)
        skills_by_name.clear()
        for skill in skills:
            skills_by_name[skill.name] = skill

    def _inject(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_prompt_block:
            return None
        updated = f"{event.system or ''}{cached_prompt_block}"
        event.system = updated
        return {"system": updated}

    def _resolve_sibling(name: str) -> Path | None:
        """Check if *name* is a sibling file of any registered skill."""
        filename = name if name.endswith(".md") else f"{name}.md"
        for record in skills_by_name.values():
            candidate = Path(record.base_dir) / filename
            if candidate.is_file():
                return candidate
        return None

    async def _load_skill(args: dict[str, Any]) -> Any:
        name = str(args.get("name", "")).strip()
        if not name:
            return ToolResult(
                content=[TextContent(type="text", text="error: name is required")],
                is_error=True,
            )
        record = skills_by_name.get(name)
        target: Path | None
        if record is not None:
            target = Path(record.file_path)
        else:
            target = _resolve_sibling(name)
        if target is None:
            available = ", ".join(sorted(skills_by_name)) or "(none)"
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"error: skill {name!r} not found. Available: {available}",
                )],
                is_error=True,
            )
        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"error reading skill: {exc}")],
                is_error=True,
            )
        return ToolResult(content=[TextContent(type="text", text=content)])

    api.register_tool(
        FunctionTool(
            name="load_skill",
            description=(
                "Load the full content of a skill by name. "
                "Use this to read detailed instructions from <available_skills>."
            ),
            parameters=pydantic_to_tool_schema(_LoadSkillParams),
            fn=_load_skill,
        )
    )

    api.on(SessionReadyEvent.CHANNEL, _populate)
    api.on(BeforeAgentStartEvent.CHANNEL, _inject)
