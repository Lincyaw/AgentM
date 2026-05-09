"""Service facades exposed on ``ExtensionAPI``.

Each service is a thin pass-through over the corresponding ``core._internal``
module. Atoms call these instead of reaching into ``_internal`` directly,
so the §11 import contract can forbid ``agentm.core._internal`` outright.

The Protocols here are the stable atom-facing surface. The default
implementations live in this same module and simply delegate to the
internal callables (no policy added). Sessions can later substitute richer
implementations (per-session config, sandboxing, etc.) without atoms
changing shape.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agentm.core.abi import AgentMessage, Tool
from agentm.core.abi.compaction import (
    CompactionResult,
    CompactionSettings,
    ContextUsageEstimate,
)
from agentm.core.abi.project_layout import ProjectLayout
from agentm.core.abi.prompt_template import PromptTemplateRecord
from agentm.core.abi.skill import SkillDiagnostic, SkillRecord


# --- Skills service --------------------------------------------------------


@runtime_checkable
class SkillsService(Protocol):
    def load_skills(
        self,
        *,
        cwd: str,
        agent_dir: str,
        skill_paths: list[str] | tuple[str, ...] = (),
        include_defaults: bool = True,
    ) -> tuple[list[SkillRecord], list[SkillDiagnostic]]: ...

    def format_skills_for_prompt(self, skills: list[SkillRecord]) -> str: ...


class _DefaultSkillsService:
    def __init__(self, layout: ProjectLayout | None = None) -> None:
        self._layout = layout

    def load_skills(
        self,
        *,
        cwd: str,
        agent_dir: str,
        skill_paths: list[str] | tuple[str, ...] = (),
        include_defaults: bool = True,
    ) -> tuple[list[SkillRecord], list[SkillDiagnostic]]:
        from agentm.core._internal.skills import load_skills as _impl

        project_dirs: tuple[str, ...] | None
        if self._layout is None:
            project_dirs = None
        else:
            project_dirs = tuple(str(p) for p in self._layout.skills_dirs())
        return _impl(
            cwd=cwd,
            agent_dir=agent_dir,
            skill_paths=skill_paths,
            include_defaults=include_defaults,
            project_skill_dirs=project_dirs,
        )

    def format_skills_for_prompt(self, skills: list[SkillRecord]) -> str:
        from agentm.core._internal.skills import (
            format_skills_for_prompt as _impl,
        )

        return _impl(skills)


# --- PromptTemplates service ----------------------------------------------


@runtime_checkable
class PromptTemplatesService(Protocol):
    def load_prompt_templates(
        self,
        *,
        cwd: str,
        agent_dir: str,
        prompt_paths: list[str] | tuple[str, ...] = (),
        include_defaults: bool = True,
    ) -> list[PromptTemplateRecord]: ...

    def expand_prompt_template(
        self,
        text: str,
        templates: list[PromptTemplateRecord],
    ) -> str | None: ...

    # In-memory prompt registry (issue #76). Atoms register named prompt
    # bodies at install time; engine code retrieves them by name. Decoupled
    # from the on-disk slash-command template flow above so kernel callers
    # don't have to coordinate filesystem state.
    def register_prompt(self, name: str, body: str) -> None: ...

    def get_prompt(self, name: str) -> str | None: ...


class _DefaultPromptTemplatesService:
    def __init__(self, layout: ProjectLayout | None = None) -> None:
        self._layout = layout
        self._registry: dict[str, str] = {}

    def load_prompt_templates(
        self,
        *,
        cwd: str,
        agent_dir: str,
        prompt_paths: list[str] | tuple[str, ...] = (),
        include_defaults: bool = True,
    ) -> list[PromptTemplateRecord]:
        from agentm.core._internal.prompt_templates import (
            load_prompt_templates as _impl,
        )

        project_dirs: tuple[str, ...] | None
        if self._layout is None:
            project_dirs = None
        else:
            project_dirs = tuple(str(p) for p in self._layout.prompts_dirs())
        return _impl(
            cwd=cwd,
            agent_dir=agent_dir,
            prompt_paths=prompt_paths,
            include_defaults=include_defaults,
            project_prompt_dirs=project_dirs,
        )

    def expand_prompt_template(
        self,
        text: str,
        templates: list[PromptTemplateRecord],
    ) -> str | None:
        from agentm.core._internal.prompt_templates import (
            expand_prompt_template as _impl,
        )

        return _impl(text, templates)

    def register_prompt(self, name: str, body: str) -> None:
        self._registry[name] = body

    def get_prompt(self, name: str) -> str | None:
        return self._registry.get(name)


# --- Catalog service -------------------------------------------------------


@runtime_checkable
class CatalogService(Protocol):
    def list_versions(
        self, name: str, root: Path | None = None
    ) -> list[str]: ...

    def current_version(
        self, path: str, root: Path | None = None
    ) -> str: ...

    def get_source_at(
        self, path: str, version: str, root: Path | None = None
    ) -> bytes: ...

    def get_manifest_at(
        self, name: str, version: str, root: Path | None = None
    ) -> dict[str, Any]: ...

    def runs_for(
        self,
        fingerprint: dict[str, Any] | str,
        root: Path | None = None,
    ) -> list[str]: ...

    def compute_atom_hash(self, source: str) -> str: ...

    def compute_active_set_fingerprint(
        self,
        loaded: dict[str, str],
        scenario: str | None,
        core_hash: str | None,
    ) -> dict[str, Any]: ...

    def read_atom_decisions(
        self,
        name: str,
        version_key: str,
        root: Path | None = None,
    ) -> list[dict[str, Any]]: ...

    def append_atom_decision(
        self,
        name: str,
        version_key: str,
        record: dict[str, Any],
        root: Path | None = None,
    ) -> None: ...


class _DefaultCatalogService:
    def list_versions(
        self, name: str, root: Path | None = None
    ) -> list[str]:
        from agentm.core._internal.catalog import list_versions as _impl

        return _impl(name, root)

    def current_version(
        self, path: str, root: Path | None = None
    ) -> str:
        from agentm.core._internal.catalog import current_version as _impl

        return _impl(path, root)

    def get_source_at(
        self, path: str, version: str, root: Path | None = None
    ) -> bytes:
        from agentm.core._internal.catalog import get_source_at as _impl

        return _impl(path, version, root)

    def get_manifest_at(
        self, name: str, version: str, root: Path | None = None
    ) -> dict[str, Any]:
        from agentm.core._internal.catalog import get_manifest_at as _impl

        return _impl(name, version, root)

    def runs_for(
        self,
        fingerprint: dict[str, Any] | str,
        root: Path | None = None,
    ) -> list[str]:
        from agentm.core._internal.catalog import runs_for as _impl

        return _impl(fingerprint, root)

    def compute_atom_hash(self, source: str) -> str:
        from agentm.core._internal.catalog import compute_atom_hash as _impl

        return _impl(source)

    def compute_active_set_fingerprint(
        self,
        loaded: dict[str, str],
        scenario: str | None,
        core_hash: str | None,
    ) -> dict[str, Any]:
        from agentm.core._internal.catalog import (
            compute_active_set_fingerprint as _impl,
        )

        return _impl(loaded, scenario, core_hash)

    def read_atom_decisions(
        self,
        name: str,
        version_key: str,
        root: Path | None = None,
    ) -> list[dict[str, Any]]:
        import json

        from agentm.harness.catalog import _layout

        path = _layout.atom_decisions_path(name, version_key, root=root)
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            out.append(json.loads(line))
        return out

    def append_atom_decision(
        self,
        name: str,
        version_key: str,
        record: dict[str, Any],
        root: Path | None = None,
    ) -> None:
        import json

        from agentm.harness.catalog import _layout

        path = _layout.atom_decisions_path(name, version_key, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


# --- Compaction service ----------------------------------------------------

Summarizer = Callable[[str, str, int], Awaitable[str]]


@runtime_checkable
class CompactionService(Protocol):
    def estimate_context_tokens(
        self, messages: list[AgentMessage]
    ) -> ContextUsageEstimate: ...

    def should_compact(
        self,
        context_tokens: int,
        context_window: int,
        settings: CompactionSettings,
    ) -> bool: ...

    def prepare_compaction(
        self,
        path_entries: list[Any],
        settings: CompactionSettings,
        current_messages: list[AgentMessage] | None = None,
        tools: list[Tool] | None = None,
    ) -> Any | None: ...

    async def compact(
        self,
        preparation: Any,
        summarizer: Summarizer,
        summarization_prompt: str,
        custom_instructions: str | None = None,
        prompts: Any | None = None,
    ) -> CompactionResult: ...


class _DefaultCompactionService:
    def estimate_context_tokens(
        self, messages: list[AgentMessage]
    ) -> ContextUsageEstimate:
        from agentm.core._internal.compaction import (
            estimate_context_tokens as _impl,
        )

        return _impl(messages)

    def should_compact(
        self,
        context_tokens: int,
        context_window: int,
        settings: CompactionSettings,
    ) -> bool:
        from agentm.core._internal.compaction import should_compact as _impl

        return _impl(context_tokens, context_window, settings)

    def prepare_compaction(
        self,
        path_entries: list[Any],
        settings: CompactionSettings,
        current_messages: list[AgentMessage] | None = None,
        tools: list[Tool] | None = None,
    ) -> Any | None:
        from agentm.core._internal.compaction import (
            prepare_compaction as _impl,
        )

        return _impl(path_entries, settings, current_messages, tools)

    async def compact(
        self,
        preparation: Any,
        summarizer: Summarizer,
        summarization_prompt: str,
        custom_instructions: str | None = None,
        prompts: Any | None = None,
    ) -> CompactionResult:
        from agentm.core._internal.compaction import compact as _impl

        return await _impl(
            preparation,
            summarizer,
            summarization_prompt,
            custom_instructions,
            prompts,
        )


# --- Default builders ------------------------------------------------------


def default_skills_service(
    layout: ProjectLayout | None = None,
) -> SkillsService:
    return _DefaultSkillsService(layout)


def default_prompt_templates_service(
    layout: ProjectLayout | None = None,
) -> PromptTemplatesService:
    return _DefaultPromptTemplatesService(layout)


def default_catalog_service() -> CatalogService:
    return _DefaultCatalogService()


def default_compaction_service() -> CompactionService:
    return _DefaultCompactionService()


def default_project_layout(cwd: str) -> ProjectLayout:
    """Return the harness's default :class:`ProjectLayout` for ``cwd``."""

    from agentm.harness.catalog import default_project_layout as _impl

    return _impl(cwd)


__all__ = [
    "CatalogService",
    "CompactionService",
    "ProjectLayout",
    "PromptTemplatesService",
    "SkillsService",
    "Summarizer",
    "default_catalog_service",
    "default_compaction_service",
    "default_project_layout",
    "default_prompt_templates_service",
    "default_skills_service",
]
