"""Typed boundary between the persistent driver and one reaction."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agentm.core.abi.bus import EventBus
from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.compaction import ContextProjection
from agentm.core.abi.context import ContextPolicy
from agentm.core.abi.messages import InterruptionMessagePolicy
from agentm.core.abi.permission import PermissionAudience, PermissionPolicy
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.stream import Model, StreamFn, ThinkingLevel
from agentm.core.abi.tool import Tool
from agentm.core.abi.tool_executor import ToolExecutor
from agentm.core.abi.tool_orchestration import ToolOrchestrator
from agentm.core.abi.trajectory import Outcome, TurnCheckpoint, TurnMeta
from agentm.core.abi.trigger import Trigger, TriggerMetadata, TriggerRenderer
from agentm.core.runtime.execution import Execution
from agentm.core.runtime.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class ReactionDependencies:
    """Only the session capabilities needed to execute one reaction."""

    trajectory: Trajectory
    bus: EventBus
    stream_fn: StreamFn
    model: Model
    tools: tuple[Tool, ...]
    system: str | None
    context_policies: tuple[ContextPolicy, ...]
    trigger_renderers: dict[str, TriggerRenderer] | None
    interrupt: CancelSignal
    shutdown: CancelSignal
    cancel_signal: CancelSignal | None
    thinking: ThinkingLevel
    tool_executor: ToolExecutor | None
    tool_orchestrator: ToolOrchestrator
    permission_policy: PermissionPolicy | None
    store: TrajectoryStore | None
    session_id: str
    root_session_id: str
    parent_session_id: str | None
    permission_audience: PermissionAudience
    tool_allowlist: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class ReactionRequest:
    execution: Execution
    trigger: Trigger
    trigger_metadata: TriggerMetadata
    dependencies: ReactionDependencies
    context_projection: ContextProjection | None
    interruption_policy: InterruptionMessagePolicy | None
    tool_calls_remaining: int | None
    checkpoint: Callable[[TurnCheckpoint], Awaitable[None]] | None = None


@dataclass(frozen=True, slots=True)
class ReactionResult:
    outcome: Outcome
    meta: TurnMeta
    tool_calls_used: int
    continuation_system_prompt: str | None


__all__ = ["ReactionDependencies", "ReactionRequest", "ReactionResult"]
