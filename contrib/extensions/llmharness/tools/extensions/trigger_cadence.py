"""SS11 single-file trigger atom: periodic cadence trigger.

Fires when ``ctx.turn_count % interval == 0``. Replaces the hardcoded
``turn_count % audit_interval == 0`` cadence that was previously baked
into the runner.

SS11 contract: single file, no atom-to-atom imports, no
``core._internal`` import, no ``harness.session`` import.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from ..runtime.triggers import SERVICE_KEY, TriggerContext, TriggerDecision, TriggerRegistry

class TriggerCadenceConfig(BaseModel):
    interval: int = Field(default=5, ge=1)


MANIFEST = ExtensionManifest(
    name="trigger_cadence",
    description=(
        "Periodic cadence trigger: fires the auditor every N turns. "
        "Registers itself via the llmharness.audit_triggers service."
    ),
    registers=(),
    config_schema=TriggerCadenceConfig,
    api_version=1,
    tier=1,
)

_DEFAULT_INTERVAL = 5


class _CadenceTrigger:
    """Fires when ``turn_count % interval == 0``."""

    name: str = "cadence"

    def __init__(self, interval: int) -> None:
        self._interval = max(1, interval)

    def should_fire(self, ctx: TriggerContext) -> TriggerDecision:
        if ctx.turn_count % self._interval == 0:
            return TriggerDecision(
                fire=True,
                reason=f"turn {ctx.turn_count} is a multiple of {self._interval}",
            )
        return TriggerDecision(fire=False)


def install(api: ExtensionAPI, config: TriggerCadenceConfig) -> None:
    """Register the cadence trigger on the parent trigger registry."""
    interval = config.interval
    registry = api.get_service(SERVICE_KEY)
    if not isinstance(registry, TriggerRegistry):
        raise RuntimeError(
            "trigger registry service not published; mount llmharness.atom first"
        )
    registry.register_trigger(_CadenceTrigger(interval))


__all__ = ["MANIFEST", "install"]
