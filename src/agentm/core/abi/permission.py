"""Permission decision ABI for tool calls and host-mediated actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import TextContent
from agentm.core.abi.tool import ToolResult

PermissionDecisionKind = Literal["allow", "deny"]
PermissionAudience = Literal["user", "subagent", "system"]
PermissionSource = Literal["user", "policy", "classifier", "mode", "system"]


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    """One permission check for a tool call or host-mediated action."""

    action: str
    session_id: str = ""
    turn_id: str = ""
    turn_index: int = 0
    tool_call_id: str = ""
    tool_name: str = ""
    args: Mapping[str, object] = field(default_factory=dict)
    audience: PermissionAudience = "user"
    origin: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """Typed decision returned by permission policies."""

    kind: PermissionDecisionKind = "allow"
    reason: str = ""
    source: PermissionSource = "policy"
    audience: PermissionAudience = "user"
    guidance: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.kind == "allow"


@runtime_checkable
class PermissionPolicy(Protocol):
    """Replaceable permission boundary for tools and other host actions."""

    async def decide(
        self,
        request: PermissionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> PermissionDecision:
        """Return the final permission decision.

        Policies that need user interaction should await that interaction
        internally and then return ``allow`` or ``deny``. The runtime does not
        expose a separate deferred state.
        """
        ...


def permission_denial_result(
    request: PermissionRequest,
    decision: PermissionDecision,
) -> ToolResult:
    """Build the default tool_result for a denied permission decision."""

    if decision.guidance:
        text = decision.guidance
    elif decision.audience == "subagent":
        text = (
            "Permission for this tool use was denied. The tool use was rejected. "
            "Try a different approach or report the limitation to complete your task."
        )
    else:
        text = (
            "The user doesn't want to proceed with this tool use. The tool use "
            "was rejected. STOP what you are doing and wait for the user to tell "
            "you how to proceed."
        )
    if decision.reason:
        text = f"{text}\n\nReason: {decision.reason}"
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=True,
        extras={
            "permission": {
                "action": request.action,
                "tool_call_id": request.tool_call_id,
                "tool_name": request.tool_name,
                "decision": decision.kind,
                "source": decision.source,
                "audience": decision.audience,
                "reason": decision.reason,
            }
        },
    )


__all__ = [
    "PermissionAudience",
    "PermissionDecision",
    "PermissionDecisionKind",
    "PermissionPolicy",
    "PermissionRequest",
    "PermissionSource",
    "permission_denial_result",
]
