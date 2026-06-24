"""Optional llmharness-backed intervention policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .policies import PolicyContext
from .schema import ActionType, Intervention, InterventionDecision


@dataclass(frozen=True)
class LlmHarnessPolicy:
    """Use llmharness offline auditing as one bounded critic policy."""

    policy_id: str = "llmharness"
    provider: tuple[str, dict[str, Any]] | None = None
    auditor_prompt: str = "minimal_index"
    audit_interval: int = 5
    action: ActionType = ActionType.VERIFY
    condition_id: str = "LLMHARNESS_AUDITOR"
    content_level: str = "LLMHARNESS_FREEFORM"
    metadata: dict[str, Any] = field(default_factory=dict)

    async def decide(self, context: PolicyContext) -> InterventionDecision:
        from llmharness import offline_audit

        if self.provider is None:
            return InterventionDecision(
                policy_id=self.policy_id,
                intervention=Intervention(
                    action=ActionType.CONTINUE,
                    condition_id=self.condition_id,
                    content_level="ABSTAIN_NO_PROVIDER",
                ),
                should_intervene=False,
                reason="no llmharness provider configured",
            )

        result = await offline_audit(
            context.messages,
            cwd=context.cwd,
            provider=self.provider,
            audit_interval=self.audit_interval,
            auditor_prompt=self.auditor_prompt,
            stop_on_first_surface=True,
        )
        if not result.surfaces:
            return InterventionDecision(
                policy_id=self.policy_id,
                intervention=Intervention(
                    action=ActionType.CONTINUE,
                    condition_id=self.condition_id,
                    content_level="ABSTAIN_NO_SURFACE",
                ),
                should_intervene=False,
                reason="llmharness did not surface an intervention",
            )

        surface = result.surfaces[0]
        metadata: dict[str, Any] = {
            **self.metadata,
            "surface_turn_index": surface.turn_index,
            "auditor_prompt": self.auditor_prompt,
            "audit_interval": self.audit_interval,
        }
        return InterventionDecision(
            policy_id=self.policy_id,
            intervention=Intervention(
                action=self.action,
                condition_id=self.condition_id,
                content_level=self.content_level,
                message=surface.reminder_text,
                metadata=metadata,
            ),
            should_intervene=True,
            reason="llmharness surfaced reminder",
            metadata=metadata,
        )
