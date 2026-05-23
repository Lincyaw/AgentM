"""Verifier-scenario atom: publish the verifier ``skills/`` directory.

Mirrors :mod:`agentm_rca.worker_skills` — the scenario manifest can't
interpolate the on-disk skill directory path, so an atom resolves it
at install time and contributes it via :class:`ResourcesDiscoverEvent`.
``skill_loader`` (builtin) consumes the event and folds the SKILL.md
bodies into the system prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.core.abi.events import ResourcesDiscoverEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILLS_DIR = _REPO_ROOT / "contrib" / "scenarios" / "verifier" / "skills"


MANIFEST = ExtensionManifest(
    name="verifier_skills",
    description=(
        "Publish the verifier scenario's bundled skills directory to "
        "skill_loader via ResourcesDiscoverEvent."
    ),
    registers=("event:resources_discover",),
)


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    skills_dir = str(_SKILLS_DIR)

    def _contribute(_event: ResourcesDiscoverEvent) -> dict[str, Any] | None:
        if not _SKILLS_DIR.is_dir():
            return None
        return {"skill_paths": [skills_dir]}

    api.on(ResourcesDiscoverEvent.CHANNEL, _contribute)


__all__ = ["MANIFEST", "install"]
