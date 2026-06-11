"""Scenario-local atom: contribute the rca ``skills/`` directory to ``skill_loader``.

The scenario manifest cannot interpolate the on-disk location of the
scenario directory into a config value, so we resolve it here at install
time and publish via :class:`ResourcesDiscoverEvent` — the documented
hook ``skill_loader`` reads to extend its discovery roots.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentm.extensions import ExtensionManifest
from agentm.core.abi import ExtensionAPI, ResourcesDiscoverEvent

from rca import SCENARIO_ROOT


class WorkerSkillsConfig(BaseModel):
    model_config = {"extra": "forbid"}
    subdir: str = ""


MANIFEST = ExtensionManifest(
    name="worker_skills",
    description=(
        "Publish the rca scenario's bundled skills directory to "
        "skill_loader via ResourcesDiscoverEvent."
    ),
    registers=("event:resources_discover",),
    config_schema=WorkerSkillsConfig,
)

async def install(api: ExtensionAPI, config: WorkerSkillsConfig) -> None:
    skills_dir = SCENARIO_ROOT / "skills" / config.subdir if config.subdir else SCENARIO_ROOT / "skills"

    def _contribute(_event: ResourcesDiscoverEvent) -> dict[str, Any] | None:
        if not skills_dir.is_dir():
            return None
        return {"skill_paths": [str(skills_dir)]}

    api.on(ResourcesDiscoverEvent.CHANNEL, _contribute)
