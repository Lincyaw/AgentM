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

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agentm.core.abi.project_layout import ProjectLayout
from agentm.core.abi.prompt_template import PromptTemplateRecord


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

    # In-memory prompt registry. Atoms register named prompt bodies at
    # install time; engine code retrieves them by name. Decoupled from the
    # on-disk slash-command template flow above so kernel callers don't have
    # to coordinate filesystem state.
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

# --- Default builders ------------------------------------------------------


def default_prompt_templates_service(
    layout: ProjectLayout | None = None,
) -> PromptTemplatesService:
    return _DefaultPromptTemplatesService(layout)


def default_catalog_service() -> CatalogService:
    return _DefaultCatalogService()


def default_project_layout(cwd: str) -> ProjectLayout:
    """Return the harness's default :class:`ProjectLayout` for ``cwd``."""

    from agentm.harness.catalog import default_project_layout as _impl

    return _impl(cwd)


__all__ = [
    "CatalogService",
    "ProjectLayout",
    "PromptTemplatesService",
    "default_catalog_service",
    "default_project_layout",
    "default_prompt_templates_service",
]
