"""Bounded critic interface + baselines (doc §3.4 / E4).

A critic maps the visible prefix h_t (no GT) to one intervention, or abstains.
It is just one column in the experiment matrix: its chosen intervention is run
through the same runner/judge/store, and G^C_t is read back from those rows
(DESIGN §1). llmharness is intentionally not imported here; it may later plug in
as one ``Critic`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentm.core.abi import AgentMessage

from ..model import (
    ActionType,
    ContentLevel,
    Intervention,
    LadderRung,
    PrefixPoint,
    Treatment,
)


class Critic(Protocol):
    @property
    def critic_id(self) -> str:
        ...

    async def decide(
        self, *, messages: list[AgentMessage], prefix: PrefixPoint
    ) -> Intervention | None:
        ...


def as_treatment(intervention: Intervention, *, critic_id: str) -> Treatment:
    """Wrap a critic's intervention as a BOUNDED-rung treatment row."""

    return Treatment(
        treatment_id=f"critic:{critic_id}",
        content_level=_content_level(intervention.content_level),
        action=intervention.action,
        intervention=intervention,
        rung=LadderRung.BOUNDED,
        metadata={"critic_id": critic_id},
    )


def _content_level(value: str) -> ContentLevel:
    try:
        return ContentLevel(value)
    except ValueError:
        return ContentLevel.TYPE


@dataclass(frozen=True)
class AbstainCritic:
    """Never intervenes — the abstention floor for coverage analysis."""

    critic_id: str = "abstain"

    async def decide(
        self, *, messages: list[AgentMessage], prefix: PrefixPoint
    ) -> Intervention | None:
        del messages, prefix
        return None


@dataclass(frozen=True)
class AlwaysVerifyCritic:
    """Fires a generic typed VERIFY at every prefix — the high-coverage baseline."""

    critic_id: str = "always_verify"
    action: ActionType = ActionType.VERIFY
    metadata: dict[str, Any] = field(default_factory=dict)

    async def decide(
        self, *, messages: list[AgentMessage], prefix: PrefixPoint
    ) -> Intervention | None:
        del messages, prefix
        return Intervention(
            action=self.action,
            condition_id=ContentLevel.TYPE.value,
            content_level=ContentLevel.TYPE.value,
            message=(
                f"[{self.action.value}] Before you conclude, re-verify a key "
                "requirement, assumption, or test interpretation."
            ),
            strength="advisory",
            metadata={"critic_id": self.critic_id, **self.metadata},
        )
