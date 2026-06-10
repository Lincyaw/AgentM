"""§11 atom: distiller tools for the evolution skill-synthesis agent.

Registers browse_reports, get_report_summary, and submit_skill.
All data is passed via config at install time.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

from rca_evolution.distiller_tools import (
    build_browse_reports_tool,
    build_get_existing_skills_tool,
    build_get_report_summary_tool,
    build_submit_skill_tool,
)

class EvolutionDistillerConfig(BaseModel):
    model_config = {"extra": "forbid"}

    reports: list[Any] = []
    report_summary: str = "(no summary)"
    existing_skills: list[Any] = []


MANIFEST = ExtensionManifest(
    name="evolution_distiller",
    description="Distiller tools for self-evolution skill synthesis.",
    registers=(
        "tool:browse_reports",
        "tool:get_report_summary",
        "tool:get_existing_skills",
        "tool:submit_skill",
    ),
    config_schema=EvolutionDistillerConfig,
)


async def install(api: ExtensionAPI, config: EvolutionDistillerConfig) -> None:
    reports = config.reports
    summary = config.report_summary
    existing_skills = config.existing_skills

    api.register_tool(build_browse_reports_tool(reports))
    api.register_tool(build_get_report_summary_tool(summary))
    api.register_tool(build_get_existing_skills_tool(existing_skills))
    api.register_tool(build_submit_skill_tool())
