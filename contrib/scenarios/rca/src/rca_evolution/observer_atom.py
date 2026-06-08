"""§11 atom: observer tools for the evolution divergence-analysis agent.

Registers get_turn, get_gt_info, get_trajectory_summary, and
submit_divergence_report. All data is passed via config at install time.
"""

from __future__ import annotations

from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

from rca_evolution.observer_tools import (
    build_get_gt_info_tool,
    build_get_trajectory_summary_tool,
    build_get_turn_tool,
    build_submit_divergence_report_tool,
)

MANIFEST = ExtensionManifest(
    name="evolution_observer",
    description="Observer tools for self-evolution divergence analysis.",
    registers=(
        "tool:get_turn",
        "tool:get_gt_info",
        "tool:get_trajectory_summary",
        "tool:submit_divergence_report",
    ),
)


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    snapshot = config.get("trajectory_snapshot", [])
    gt_info = config.get("gt_info", {})
    summary = config.get("trajectory_summary", "(no summary)")
    total_turns = len(snapshot)

    api.register_tool(build_get_turn_tool(snapshot))
    api.register_tool(build_get_gt_info_tool(gt_info))
    api.register_tool(build_get_trajectory_summary_tool(summary, total_turns))
    api.register_tool(build_submit_divergence_report_tool())
