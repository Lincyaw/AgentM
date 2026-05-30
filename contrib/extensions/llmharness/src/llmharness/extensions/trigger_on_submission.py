"""SS11 single-file trigger atom: fire on terminal tool submission.

Fires when ``ctx.tool_names_called`` intersects a configured set of
tool names. Default tools: ``submit_final_report`` and
``submit_investigation``.

SS11 contract: single file, no atom-to-atom imports, no
``core._internal`` import, no ``harness.session`` import.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from ..audit.triggers import SERVICE_KEY, TriggerContext, TriggerDecision, TriggerRegistry

MANIFEST = ExtensionManifest(
    name="trigger_on_submission",
    description=(
        "Terminal-tool submission trigger: fires the auditor when the "
        "agent calls a terminal tool (e.g. submit_final_report). "
        "Registers itself via the llmharness.audit_triggers service."
    ),
    registers=(),
    config_schema={
        "type": "object",
        "properties": {
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["submit_final_report", "submit_investigation"],
            },
        },
        "additionalProperties": False,
    },
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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    """Register the on-submission trigger on the parent trigger registry."""
    raw_names = config.get("tool_names")
    if isinstance(raw_names, list):
        tool_names = frozenset(str(n) for n in raw_names if isinstance(n, str) and n)
    else:
        tool_names = frozenset(_DEFAULT_TOOL_NAMES)
    registry = api.get_service(SERVICE_KEY)
    if not isinstance(registry, TriggerRegistry):
        raise RuntimeError(
            "trigger registry service not published; mount llmharness.adapters.agentm first"
        )
    registry.register_trigger(_OnSubmissionTrigger(tool_names))


__all__ = ["MANIFEST", "install"]
