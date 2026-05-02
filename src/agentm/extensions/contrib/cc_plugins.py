"""Builtin atom that surfaces installed Claude Code plugins to the agent.

Reads ``~/.claude/plugins/installed_plugins.json`` and, for each activated
plugin, contributes the plugin's ``skills/``, ``agents/``, and
``commands/`` directories to peer atoms via the ``resources_discover``
event:

* ``skill_loader`` picks up extra ``skill_paths`` already (existing hook).
* ``cc_agents`` was extended to honor ``agent_paths`` from the same
  event in lockstep with this atom.
* ``cc_commands`` already accepts ``command_paths`` via configuration;
  we do not feed it through ``resources_discover`` (that atom collects its
  inputs at install time, not on the event), so plugin commands need a
  separate hook — see §3 of this docstring.

Scope filter: ``installed_plugins.json`` lists each entry as either
``"user"`` (always active) or ``"project"`` (only active when the
session's ``cwd`` matches the recorded ``projectPath``). We respect that
so a project-scoped plugin doesn't pollute unrelated repositories.

Section 3 — commands: ``cc_commands`` is install-time only, so to
include plugin command markdown files in the ``<available_skills>`` block
we contribute them as ``extra_skills`` (skill_loader's other input), the
same trick ``cc_commands`` itself uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentm.core.abi.skill import SkillRecord
from agentm.core.lib.frontmatter import parse_frontmatter
from agentm.extensions import ExtensionManifest
from agentm.harness.events import ResourcesDiscoverEvent
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="cc_plugins",
    description=(
        "Expose installed Claude Code plugins (~/.claude/plugins/) to the "
        "agent: their skills, agents, and commands appear in the system "
        "prompt the same way ~/.claude/{skills,agents,commands} do."
    ),
    registers=("event:resources_discover",),
    config_schema={
        "type": "object",
        "properties": {
            "registry_path": {
                "type": "string",
                "description": (
                    "Override path to ``installed_plugins.json``. Defaults "
                    "to ``~/.claude/plugins/installed_plugins.json``."
                ),
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Plugin keys to skip (matches the JSON's "
                    "``<plugin>@<marketplace>`` form)."
                ),
            },
        },
        "additionalProperties": True,
    },
)


def _default_registry_path() -> Path:
    return Path.home() / ".claude" / "plugins" / "installed_plugins.json"


def _resolve_install_paths(
    registry_path: Path,
    cwd: str,
    exclude: set[str],
) -> list[Path]:
    """Return the install directories of every activated plugin relevant
    to ``cwd``. Project-scoped plugins are filtered by ``projectPath``."""

    if not registry_path.is_file():
        return []
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cwd_resolved = str(Path(cwd).resolve())
    result: list[Path] = []
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
                # Unknown scope — be conservative and skip.
                continue
            p = Path(install_path)
            if p.is_dir():
                result.append(p)
    return result


def _command_skill_records(install_paths: list[Path]) -> list[SkillRecord]:
    """Read plugin ``commands/*.md`` and emit ``SkillRecord`` entries.

    ``cc_commands`` does this for ``~/.claude/commands/``; we mirror it
    for plugin commands so they reach the model through the same
    ``<available_skills>`` block.
    """

    out: list[SkillRecord] = []
    seen: set[str] = set()
    for install in install_paths:
        cmd_dir = install / "commands"
        if not cmd_dir.is_dir():
            continue
        for md_path in sorted(cmd_dir.glob("*.md")):
            try:
                text = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            metadata, _body = parse_frontmatter(text)
            name = metadata.get("name") if isinstance(metadata, dict) else None
            if not isinstance(name, str) or not name.strip():
                name = md_path.stem
            name = name.strip()
            if name in seen:
                continue
            description = (
                metadata.get("description")
                if isinstance(metadata, dict)
                else None
            )
            if not isinstance(description, str) or not description.strip():
                # Without a description the model has no reason to pick
                # this command — drop it to keep the prompt budget tight.
                continue
            seen.add(name)
            out.append(
                SkillRecord(
                    name=name,
                    description=description.strip(),
                    file_path=str(md_path.resolve()),
                    base_dir=str(md_path.parent.resolve()),
                    disable_model_invocation=False,
                    source="plugin",
                )
            )
    return out


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    registry_path = Path(
        config.get("registry_path") or _default_registry_path()
    )
    exclude = set(config.get("exclude") or ())

    def _on_discover(_event: ResourcesDiscoverEvent) -> dict[str, Any] | None:
        install_paths = _resolve_install_paths(registry_path, api.cwd, exclude)
        if not install_paths:
            return None
        skill_paths: list[str] = []
        agent_paths: list[str] = []
        for p in install_paths:
            sd = p / "skills"
            if sd.is_dir():
                skill_paths.append(str(sd))
            ad = p / "agents"
            if ad.is_dir():
                agent_paths.append(str(ad))
        extra_skills = _command_skill_records(install_paths)
        payload: dict[str, Any] = {}
        if skill_paths:
            payload["skill_paths"] = skill_paths
        if agent_paths:
            payload["agent_paths"] = agent_paths
        if extra_skills:
            payload["extra_skills"] = extra_skills
        return payload or None

    api.on("resources_discover", _on_discover)


__all__ = ["MANIFEST", "install"]
