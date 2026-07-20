"""Builtin ``skill_loader`` atom.

Discovers SKILL.md files (default agent dir, explicit paths, peer
contributions via ``ResourcesDiscoverEvent``) and injects an
``<available_skills>`` index into the system prompt.  Provides the
``load_skill`` tool for on-demand reading.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    BeforeRunEvent,
    EventBusObserver,
    FunctionTool,
    Handler,
    JsonValue,
    ResourcesDiscoverEvent,
    SessionReadyEvent,
    ToolResult,
)
from agentm.core.lib import (
    error_result,
    expand_path,
    expand_path_from_cwd,
    parse_frontmatter,
    pydantic_to_tool_schema,
    text_result,
)
from agentm.extensions import ExtensionManifest


def _agentm_home_dir() -> Path:
    home = os.environ.get("AGENTM_HOME")
    return expand_path(home) if home else Path.home() / ".agentm"


# ---------------------------------------------------------------------------
# Skill records
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Skill discovery engine
# ---------------------------------------------------------------------------

_MAX_NAME_LEN = 64
_MAX_DESC_LEN = 1024
_NAME_RE = re.compile(r"^[a-z0-9-]+$")


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

    metadata, _ = parse_frontmatter(text)
    skill_dir = str(Path(file_path).parent)
    parent_dir_name = Path(skill_dir).name

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        return None, diagnostics

    raw_name = metadata.get("name")
    name = raw_name if isinstance(raw_name, str) and raw_name else parent_dir_name
    if not _NAME_RE.match(name) or len(name) > _MAX_NAME_LEN:
        diagnostics.append(
            SkillDiagnostic(
                level="warning",
                message=f"invalid skill name: {name!r}",
                path=file_path,
            )
        )

    disable = metadata.get("disable-model-invocation") is True

    return (
        SkillRecord(
            name=name,
            description=description,
            file_path=file_path,
            base_dir=skill_dir,
            disable_model_invocation=disable,
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
        directory, topdown=True, followlinks=True
    ):
        real_dir = os.path.realpath(dirpath)
        if real_dir in visited:
            dirnames[:] = []
            continue
        visited.add(real_dir)
        dirnames[:] = sorted(dirnames)

        skill_path = os.path.join(dirpath, "SKILL.md")
        if "SKILL.md" in filenames and os.path.isfile(skill_path):
            skill, diags = _parse_skill_file(os.path.abspath(skill_path), source)
            if skill is not None:
                skills.append(skill)
            diagnostics.extend(diags)
            dirnames[:] = []
            continue

        if not include_root_files or os.path.abspath(dirpath) != os.path.abspath(
            directory
        ):
            continue
        for filename in sorted(filenames):
            if not filename.endswith(".md"):
                continue
            fp = os.path.join(dirpath, filename)
            if not os.path.isfile(fp):
                continue
            skill, diags = _parse_skill_file(os.path.abspath(fp), source)
            if skill is not None:
                skills.append(skill)
            diagnostics.extend(diags)

    return skills, diagnostics


def _load_skills(
    *,
    cwd: str,
    agent_dir: str,
    skill_paths: tuple[str, ...] = (),
    include_defaults: bool = True,
) -> list[SkillRecord]:
    discovered: list[SkillRecord] = []
    seen_files: set[str] = set()
    seen_names: set[str] = set()

    def add_batch(batch: list[SkillRecord]) -> None:
        for skill in batch:
            real = os.path.realpath(skill.file_path)
            if real in seen_files or skill.name in seen_names:
                continue
            seen_files.add(real)
            seen_names.add(skill.name)
            discovered.append(skill)

    if include_defaults:
        skills, _ = _load_skills_from_dir(
            os.path.join(agent_dir, "skills"),
            "user",
            include_root_files=True,
        )
        add_batch(skills)

    for raw in skill_paths:
        resolved = str(expand_path_from_cwd(raw, cwd))
        if os.path.isdir(resolved):
            skills, _ = _load_skills_from_dir(resolved, "path", include_root_files=True)
            add_batch(skills)
        elif os.path.isfile(resolved):
            skill, _ = _parse_skill_file(os.path.abspath(resolved), "path")
            if skill is not None:
                add_batch([skill])

    return discovered


def _format_skills_for_prompt(skills: list[SkillRecord]) -> str:
    visible = [s for s in skills if not s.disable_model_invocation]
    if not visible:
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
        "",
        "<available_skills>",
    ]
    for skill in visible:
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


# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------


class SkillLoaderConfig(BaseModel):
    skill_paths: list[str] = []
    include_defaults: bool = True
    inherit_claude: bool | None = None


MANIFEST = ExtensionManifest(
    name="skill_loader",
    description="Discover SKILL.md files and inject an <available_skills> index.",
    registers=(
        "tool:load_skill",
        "event:before_run",
        "event:resources_discover",
        "event:session_ready",
    ),
    config_schema=SkillLoaderConfig,
    requires=(),
    priority=AtomInstallPriority.CONTEXT,
)

_RESOURCE_RESPONSE_KEYS = frozenset({"skill_paths", "extra_skills"})


class _LoadSkillParams(BaseModel):
    name: str = Field(description="Skill name from <available_skills>.")


class _ResourceResponseObserver(EventBusObserver):
    def __init__(self) -> None:
        self.response_owners: dict[int, str] = {}

    def on_handler_done(
        self,
        channel: str,
        handler: Handler,
        event: object,
        result: object,
        error: BaseException | None,
        duration_ns: int,
        dispatch_id: str,
        owner: str | None = None,
    ) -> None:
        del handler, event, error, duration_ns, dispatch_id
        if channel == ResourcesDiscoverEvent.CHANNEL and isinstance(result, dict):
            self.response_owners[id(result)] = owner or "<unknown>"


class _SkillLoaderRuntime:
    def __init__(self, api: AtomAPI, config: SkillLoaderConfig) -> None:
        self._api = api
        self._include_defaults = config.include_defaults
        self._inherit_claude = (
            config.inherit_claude
            if config.inherit_claude is not None
            else self._include_defaults
        )
        self._configured_paths = list(config.skill_paths)
        self._cached_prompt_block = ""
        self._skills_by_name: dict[str, SkillRecord] = {}

    def install(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="load_skill",
                description=(
                    "Load the full content of a skill by name. "
                    "Use this to read detailed instructions from "
                    "<available_skills>."
                ),
                parameters=pydantic_to_tool_schema(_LoadSkillParams),
                fn=self.load_skill,
            )
        )
        self._api.on(SessionReadyEvent.CHANNEL, self.populate)
        self._api.on(BeforeRunEvent.CHANNEL, self.inject)

    async def populate(self, _: SessionReadyEvent) -> None:
        discovered_paths = self._discovery_paths()
        responses, owners = await self._discover_peer_resources()
        self._merge_resource_responses(responses, owners, discovered_paths)
        skills = _load_skills(
            cwd=self._api.ctx.cwd,
            agent_dir=str(_agentm_home_dir()),
            skill_paths=tuple(discovered_paths),
            include_defaults=self._include_defaults,
        )
        self._cached_prompt_block = _format_skills_for_prompt(skills)
        self._skills_by_name = {s.name: s for s in skills}

    def inject(self, event: BeforeRunEvent) -> dict[str, str] | None:
        if not self._cached_prompt_block:
            return None
        return {"system": f"{event.system or ''}{self._cached_prompt_block}"}

    async def load_skill(self, args: dict[str, JsonValue]) -> ToolResult:
        name = str(args.get("name", "")).strip()
        if not name:
            return error_result("error: name is required")
        target = self._skill_target(name)
        if target is None:
            available = ", ".join(sorted(self._skills_by_name)) or "(none)"
            return error_result(
                f"error: skill {name!r} not found. Available: {available}"
            )
        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            return error_result(f"error reading skill: {exc}")
        return text_result(content)

    def _discovery_paths(self) -> list[str]:
        paths = list(self._configured_paths)
        if not self._inherit_claude:
            return paths
        paths.append(str(Path.home() / ".claude" / "skills"))
        paths.append(str(expand_path_from_cwd(".claude/skills", self._api.ctx.cwd)))
        paths.extend(self._claude_plugin_skill_dirs())
        return paths

    @staticmethod
    def _claude_plugin_skill_dirs() -> list[str]:
        plugin_cache = Path.home() / ".claude" / "plugins" / "cache"
        if not plugin_cache.is_dir():
            return []
        paths: list[str] = []
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
                        paths.append(str(skills_dir))
        return paths

    async def _discover_peer_resources(self) -> tuple[list[object], dict[int, str]]:
        observer = _ResourceResponseObserver()
        unsubscribe = self._api.bus.add_observer(observer)
        try:
            responses = await self._api.bus.emit(
                ResourcesDiscoverEvent.CHANNEL,
                ResourcesDiscoverEvent(cwd=self._api.ctx.cwd, reason="startup"),
            )
        finally:
            unsubscribe()
        return responses, observer.response_owners

    @staticmethod
    def _merge_resource_responses(
        responses: list[object],
        owners: dict[int, str],
        discovered_paths: list[str],
    ) -> None:
        for response in responses:
            if not isinstance(response, dict):
                continue
            extra_paths = response.get("skill_paths")
            if isinstance(extra_paths, list):
                discovered_paths.extend(str(p) for p in extra_paths)

    def _skill_target(self, name: str) -> Path | None:
        record = self._skills_by_name.get(name)
        if record is not None:
            return Path(record.file_path)
        return self._resolve_sibling(name)

    def _resolve_sibling(self, name: str) -> Path | None:
        filename = name if name.endswith(".md") else f"{name}.md"
        for record in self._skills_by_name.values():
            candidate = Path(record.base_dir) / filename
            if candidate.is_file():
                return candidate
        return None


async def install(api: AtomAPI, config: SkillLoaderConfig) -> None:
    _SkillLoaderRuntime(api, config).install()
