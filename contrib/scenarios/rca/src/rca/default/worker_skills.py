"""Scenario-local atom: contribute the rca ``skills/`` directory to ``skill_loader``.

The scenario manifest cannot interpolate the on-disk location of the
scenario directory into a config value, so we resolve it here at install
time and publish via :class:`ResourcesDiscoverEvent` — the documented
hook ``skill_loader`` reads to extend its discovery roots.
"""

from __future__ import annotations

from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.core.abi.events import ResourcesDiscoverEvent
from agentm.core.abi.extension import ExtensionAPI

from rca import SCENARIO_ROOT

_SKILLS_DIR = SCENARIO_ROOT / "skills"


MANIFEST = ExtensionManifest(
    name="worker_skills",
    description=(
        "Publish the rca scenario's bundled skills directory to "
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
