"""Claude Code installed-plugin resource discovery.

Reads ``~/.claude/plugins/installed_plugins.json`` and contributes each active
plugin's resource directories through the typed ``resources_discover`` event.
Project-scoped plugins are filtered by ``projectPath`` so they do not affect
unrelated repositories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentm.core.abi.events import DiagnosticEvent
from agentm.extensions import ExtensionManifest
from agentm.core.abi.events import ResourcesDiscoverEvent
from agentm.core.abi.extension import ExtensionAPI


_RECOGNIZED_SCOPES: frozenset[str] = frozenset({"user", "project"})


class PluginsConfig(BaseModel):
    model_config = {"extra": "allow"}

    registry_path: str | None = None
    exclude: list[str] = []


MANIFEST = ExtensionManifest(
    name="plugins",
    description=(
        "Expose installed Claude Code plugin skill, agent, and command "
        "directories through resource discovery."
    ),
    registers=("event:resources_discover",),
    config_schema=PluginsConfig,
    tier=2,
)


def _default_registry_path() -> Path:
    return Path.home() / ".claude" / "plugins" / "installed_plugins.json"


def _resolve_install_paths(
    registry_path: Path,
    cwd: str,
    exclude: set[str],
) -> tuple[list[Path], list[str]]:
    """Return (install_paths, unknown_scope_keys).

    Unknown scopes are silently dropped from the path list but their
    ``<plugin>@<marketplace>`` keys are returned so the caller can surface
    a diagnostic.
    """

    if not registry_path.is_file():
        return [], []
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], []
    cwd_resolved = str(Path(cwd).resolve())
    result: list[Path] = []
    unknown_scopes: list[str] = []
    for key, entries in (data.get("plugins") or {}).items():
        if key in exclude or not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            scope = entry.get("scope")
            install_path = entry.get("installPath")
            if not isinstance(install_path, str):
                continue
            if scope == "project":
                project_path = entry.get("projectPath")
                if not isinstance(project_path, str):
                    continue
                if str(Path(project_path).resolve()) != cwd_resolved:
                    continue
            elif scope != "user":
                if isinstance(scope, str) and scope not in _RECOGNIZED_SCOPES:
                    unknown_scopes.append(f"{key}:{scope}")
                continue
            path = Path(install_path)
            if path.is_dir():
                result.append(path)
    return result, unknown_scopes


def install(api: ExtensionAPI, config: PluginsConfig) -> None:
    registry_path = Path(config.registry_path) if config.registry_path else _default_registry_path()
    exclude = set(config.exclude)

    async def _on_discover(_event: ResourcesDiscoverEvent) -> dict[str, Any] | None:
        install_paths, unknown_scopes = _resolve_install_paths(
            registry_path, api.cwd, exclude
        )
        for scope_key in unknown_scopes:
            await api.events.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="warning",
                    source="cc.plugins",
                    message=(
                        f"ignored installed plugin entry with unrecognized "
                        f"scope: {scope_key} (expected 'user' or 'project')"
                    ),
                ),
            )
        if not install_paths:
            return None
        resource_dirs: dict[str, list[str]] = {
            "skill_paths": [],
            "agent_paths": [],
            "command_paths": [],
        }
        for install_path in install_paths:
            for key, child in (
                ("skill_paths", "skills"),
                ("agent_paths", "agents"),
                ("command_paths", "commands"),
            ):
                path = install_path / child
                if path.is_dir():
                    resource_dirs[key].append(str(path))
        return {key: value for key, value in resource_dirs.items() if value} or None

    api.on(ResourcesDiscoverEvent.CHANNEL, _on_discover)


__all__ = ("MANIFEST", "install")
