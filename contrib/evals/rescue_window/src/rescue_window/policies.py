"""Intervention-policy abstractions for rescue-window experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentm.core.abi import AgentMessage

from .schema import ActionType, Intervention, InterventionDecision


@dataclass(frozen=True)
class PolicyContext:
    """Prefix context visible to an intervention policy."""

    source_session_id: str
    messages: list[AgentMessage]
    cwd: str
    provider: tuple[str, dict[str, Any]] | None = None
    trajectory_id: str | None = None
    fork_turn_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InterventionPolicy(Protocol):
    """Policy mapping a visible prefix to one intervention decision."""

    @property
    def policy_id(self) -> str:
        ...

    async def decide(self, context: PolicyContext) -> InterventionDecision:
        ...


@dataclass(frozen=True)
class StaticPolicy:
    """A deterministic fixed intervention policy."""

    policy_id: str
    intervention: Intervention
    should_intervene: bool = True
    reason: str = "static"

    async def decide(self, context: PolicyContext) -> InterventionDecision:
        del context
        return InterventionDecision(
            policy_id=self.policy_id,
            intervention=self.intervention,
            should_intervene=self.should_intervene,
            reason=self.reason,
        )


@dataclass(frozen=True)
class ManualPolicy:
    """A named manual or oracle intervention replayed as a policy."""

    policy_id: str
    intervention: Intervention
    source: str
    uses_oracle: bool = False

    async def decide(self, context: PolicyContext) -> InterventionDecision:
        del context
        metadata = {"source": self.source, "uses_oracle": self.uses_oracle}
        return InterventionDecision(
            policy_id=self.policy_id,
            intervention=Intervention(
                action=self.intervention.action,
                condition_id=self.intervention.condition_id,
                content_level=self.intervention.content_level,
                message=self.intervention.message,
                target=self.intervention.target,
                evidence=self.intervention.evidence,
                strength=self.intervention.strength,
                valid_until=self.intervention.valid_until,
                metadata={**self.intervention.metadata, **metadata},
            ),
            should_intervene=self.intervention.action != ActionType.CONTINUE
            or bool(self.intervention.message),
            reason="manual",
            metadata=metadata,
        )
