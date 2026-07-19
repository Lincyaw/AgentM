"""Provider/session config-change policy helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from agentm.core.abi.provider import ProviderSessionIdentity
from agentm.core.abi.trajectory import SessionConfigChange, TrajectoryNode


ConfigChangeDecision = Literal["allow", "fork_required", "deny"]


@dataclass(frozen=True, slots=True)
class ProviderConfigChangeRequest:
    session_id: str
    key: str
    before: str | None
    after: str | None
    committed_turn_count: int = 0
    explicit: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ProviderConfigChangeResult:
    decision: ConfigChangeDecision
    message: str = ""
    change: SessionConfigChange | None = None


class ProviderConfigChangePolicy:
    """Fail-stop provider/model drift after history exists."""

    def decide(
        self,
        request: ProviderConfigChangeRequest,
        *,
        identity: ProviderSessionIdentity | None = None,
    ) -> ProviderConfigChangeResult:
        if request.before == request.after:
            return ProviderConfigChangeResult(decision="allow", message="unchanged")
        if request.committed_turn_count <= 0:
            return ProviderConfigChangeResult(
                decision="allow",
                message="provider config is still mutable before the first commit",
                change=_change(request),
            )
        if request.explicit:
            return ProviderConfigChangeResult(
                decision="allow",
                message="explicit config_change control node required",
                change=_change(request, identity=identity),
            )
        return ProviderConfigChangeResult(
            decision="fork_required",
            message="provider/model identity is frozen after the first committed turn",
        )

    def control_node(
        self,
        change: SessionConfigChange,
        *,
        seq: int,
        parent_id: str | None,
    ) -> TrajectoryNode:
        return TrajectoryNode(
            id=change.node_id or f"config-change:{change.change_id}",
            session_id=change.session_id,
            seq=seq,
            kind="config_change",
            parent_id=parent_id,
            turn_id=change.turn_id,
            turn_index=change.turn_index,
            payload={
                "change_id": change.change_id,
                "key": change.key,
                "before": change.before,
                "after": change.after,
                "reason": change.reason,
                "metadata": dict(change.metadata),
            },
            timestamp=time.time(),
        )


def _change(
    request: ProviderConfigChangeRequest,
    *,
    identity: ProviderSessionIdentity | None = None,
) -> SessionConfigChange:
    metadata: dict[str, object] = {}
    if identity is not None:
        metadata["provider"] = identity.name
        if identity.model_id is not None:
            metadata["model_id"] = identity.model_id
        if identity.active_set_digest is not None:
            metadata["active_set_digest"] = identity.active_set_digest
    return SessionConfigChange(
        change_id=f"{request.key}:{int(time.time() * 1000)}",
        session_id=request.session_id,
        key=request.key,
        before=request.before,
        after=request.after,
        reason=request.reason,
        metadata=metadata,
    )


__all__ = [
    "ConfigChangeDecision",
    "ProviderConfigChangePolicy",
    "ProviderConfigChangeRequest",
    "ProviderConfigChangeResult",
]
