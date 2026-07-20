# code-health: ignore-file[AM025] -- atom tools validate untyped tool, config, and service payloads
"""Builtin ``loop_budget`` atom -- sets the agent-loop turn / tool budget.

The loop budget is a policy, so it lives as an atom rather than a
privileged manifest field: a scenario that wants a hard ceiling lists
this atom with ``config``, exactly like any other capability.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    LOOP_BUDGET_SERVICE,
    LoopConfig,
)
from agentm.extensions import ExtensionManifest


class LoopBudgetConfig(BaseModel):
    max_turns: int | None = None
    max_tool_calls: int | None = None


MANIFEST = ExtensionManifest(
    name="loop_budget",
    description="Sets the agent-loop turn / tool-call budget for the session.",
    registers=("service:loop_budget",),
    config_schema=LoopBudgetConfig,
    requires=(),
    priority=AtomInstallPriority.SERVICE,
)


def install(api: AtomAPI, config: LoopBudgetConfig) -> None:
    loop_config = LoopConfig(
        max_turns=_positive_or_none(config.max_turns, "max_turns"),
        max_tool_calls=_positive_or_none(config.max_tool_calls, "max_tool_calls"),
    )
    api.services.register(LOOP_BUDGET_SERVICE, loop_config, scope="session")


def _positive_or_none(value: int | None, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"loop_budget: '{key}' must be a positive integer or null; got {value!r}"
        )
    return value
