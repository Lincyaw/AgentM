"""SS11 single-file trigger atom: fire on terminal tool submission.

Fires when ``ctx.tool_names_called`` intersects a configured set of
tool names. Default tools: ``submit_final_report`` and
``submit_investigation``.

SS11 contract: single file, no atom-to-atom imports, no
``core._internal`` import, no ``harness.session`` import.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from ..runtime.triggers import SERVICE_KEY, TriggerContext, TriggerDecision, TriggerRegistry

class TriggerOnSubmissionConfig(BaseModel):
    tool_names: list[str] = Field(
        default=["submit_final_report", "submit_investigation"],
    )


MANIFEST = ExtensionManifest(
    name="trigger_on_submission",
    description=(
        "Terminal-tool submission trigger: fires the auditor when the "
        "agent calls a terminal tool (e.g. submit_final_report). "
        "Registers itself via the llmharness.audit_triggers service."
    ),
    registers=(),
    config_schema=TriggerOnSubmissionConfig,
    api_version=1,
    tier=1,
)

_DEFAULT_TOOL_NAMES = ("submit_final_report", "submit_investigation")


class _OnSubmissionTrigger:
    """Fires when ``ctx.tool_names_called`` intersects the configured set."""

    name: str = "on_submission"

    def __init__(self, tool_names: frozenset[str]) -> None:
        self._tool_names = tool_names

    def should_fire(self, ctx: TriggerContext) -> TriggerDecision:
        matched = ctx.tool_names_called & self._tool_names
        if matched:
            names_str = ", ".join(sorted(matched))
            return TriggerDecision(
                fire=True,
                reason=f"terminal tool(s) {names_str} called",
            )
        return TriggerDecision(fire=False)


def install(api: ExtensionAPI, config: TriggerOnSubmissionConfig) -> None:
    """Register the on-submission trigger on the parent trigger registry."""
    tool_names = frozenset(n for n in config.tool_names if n)
    registry = api.get_service(SERVICE_KEY)
    if not isinstance(registry, TriggerRegistry):
        raise RuntimeError(
            "trigger registry service not published; mount llmharness.atom first"
        )
    registry.register_trigger(_OnSubmissionTrigger(tool_names))


__all__ = ["MANIFEST", "install"]
