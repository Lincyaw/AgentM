"""Scenario adapter seam (DESIGN §9 — no scenario conflation).

The generic harness knows nothing about RCA (or any benchmark). A scenario
provides three things through this Protocol: the tool call that terminates a
trajectory, how to judge a rollout against ground truth, and the case ground
truth used to build oracle/targeted treatments. Scenarios implement it in their
own package (RCA does so in ``rca_eval``) and the CLI selects one by name, so the
harness carries zero static scenario imports.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentm.core.abi import AgentMessage, AssistantMessage, ToolCallBlock

from .corpus import TrajectoryRef


@dataclass(frozen=True)
class ScoredOutcome:
    """One judged rollout: continuous score primary, binary secondary (DESIGN §6)."""

    binary_success: bool | None
    normalized_score: float | None
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class GroundTruth:
    """Scenario-opaque ground truth for a case.

    ``targets`` are free-form target strings (RCA: root-cause services; coding:
    files / requirements). ``summary`` is a human phrasing for the oracle dump.
    """

    targets: tuple[str, ...]
    summary: str
    fault_kinds: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_target(self) -> str | None:
        return self.targets[0] if self.targets else None


@runtime_checkable
class ScenarioAdapter(Protocol):
    """What a benchmark must provide for the rescue-window harness to score it."""

    name: str
    final_tool: str  # tool whose call terminates a trajectory / carries the submission

    async def judge(
        self, messages: list[AgentMessage], ref: TrajectoryRef
    ) -> ScoredOutcome:
        ...

    def ground_truth(self, ref: TrajectoryRef) -> GroundTruth:
        ...


# name -> "module:class"; resolved lazily so the harness never imports a scenario.
_ADAPTERS: dict[str, str] = {
    "rca": "rca_eval.rescue_window_eval:RcaRescueAdapter",
}


def load_adapter(name: str) -> ScenarioAdapter:
    spec = _ADAPTERS.get(name)
    if spec is None:
        raise ValueError(
            f"unknown scenario adapter {name!r}; known: {sorted(_ADAPTERS)}"
        )
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def extract_tool_args(
    messages: list[AgentMessage], tool_name: str
) -> dict[str, Any] | None:
    """Arguments of the last call to ``tool_name`` in ``messages`` (generic)."""

    for message in reversed(messages):
        if not isinstance(message, AssistantMessage):
            continue
        for block in reversed(message.content):
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return dict(block.arguments)
    return None
