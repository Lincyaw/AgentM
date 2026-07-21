# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Context reconstruction — async policies and pluggable transforms.

Context is a COMPUTED view, not stored data.  ``build_context`` walks
the committed turns, converts each to messages, then applies registered
ContextPolicies in priority order.

Policies are the pluggability axis for compaction, injection, cache
discipline, system reminders, and any future context transformation.

Two context-transform mechanisms exist by design:

- **ContextPolicy** — runs inside ``build_context`` before each provider
  request. Sees committed Turns only. Async. Has access to
  session identity and services via ``bind()``.  Use for transforms
  that depend on the full trajectory structure (compaction, summary).

- **ContextEvent** (bus) — runs after ``build_context`` for the active Turn.
  Sees committed Turns + trigger + in-flight messages.
  Async.  Use for transforms that need the live tail (cache-discipline
  suffix, token injection).

The split is deliberate: policies transform the durable prefix,
bus handlers transform the live tail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentm.core.abi.stream import Model, StreamFn

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import (
    AgentMessage,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
    synthetic_user_message,
)
from agentm.core.abi.trajectory import Turn
from agentm.core.abi.trigger import (
    BackgroundCompletion,
    ContinueTrigger,
    Injection,
    MonitorFire,
    SubagentResult,
    Trigger,
    TriggerMetadata,
    TriggerRenderer,
    UserInput,
)


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Session-scoped context given to a ContextPolicy at bind time."""

    session_id: str = ""
    parent_session_id: str | None = None
    services: Mapping[str, object] | None = None
    store: object | None = None
    model: "Model | None" = None
    stream_fn: "StreamFn | None" = None
    trigger_renderers: dict[str, TriggerRenderer] | None = None


@runtime_checkable
class ContextPolicy(Protocol):
    """Transform the message list during context reconstruction.

    ``transform`` is async so policies can make LLM calls (compaction)
    or query external services.

    Session binding is a separate optional capability; transform-only policies
    satisfy this protocol without implementing a no-op method.
    """

    async def transform(
        self,
        messages: list[AgentMessage],
        turns: Sequence[Turn],
    ) -> list[AgentMessage]: ...


class ContextTransformCancelled(Exception):
    """Raised when cancellation interrupts an async context transform."""


@runtime_checkable
class CancellableContextPolicy(Protocol):
    """Optional per-turn cancellation capability for async policies."""

    async def transform_with_signal(
        self,
        messages: list[AgentMessage],
        turns: Sequence[Turn],
        *,
        signal: CancelSignal,
    ) -> list[AgentMessage]: ...


@runtime_checkable
class BindableContextPolicy(Protocol):
    """Optional session-binding capability for context policies."""

    def bind(self, ctx: PolicyContext) -> None: ...


# --- Trigger → message renderers --------------------------------------------

_SYSTEM_REMINDER_SOURCES = frozenset({"background", "monitor", "subagent"})


def _render_user_input(trigger: UserInput) -> list[AgentMessage]:
    return [
        UserMessage(
            role="user",
            content=list(trigger.content),
            timestamp=0.0,
        )
    ]


def _render_system_reminder(source: str, text: str) -> list[AgentMessage]:
    wrapped = f'<system-reminder source="{source}">\n{text}\n</system-reminder>'
    return [
        synthetic_user_message(
            wrapped,
            kind=f"{source}_reminder",
            origin=source,
            visibility="hidden",
        )
    ]


def render_trigger(
    trigger: Trigger | object,
    renderers: dict[str, TriggerRenderer] | None = None,
) -> list[AgentMessage]:
    """Convert a Trigger into AgentMessages using registered renderers."""

    if not isinstance(trigger, Trigger):
        raise TypeError("trigger must implement the Trigger protocol")
    source = trigger.source
    if not isinstance(source, str) or not source:
        raise ValueError("trigger source must be a non-empty string")
    if renderers and source in renderers:
        return renderers[source].render(trigger)

    if isinstance(trigger, UserInput):
        return _render_user_input(trigger)

    if isinstance(trigger, ContinueTrigger):
        return []

    if isinstance(trigger, Injection):
        return list(trigger.messages)

    if isinstance(trigger, BackgroundCompletion):
        return _render_system_reminder("background", trigger.payload)

    if isinstance(trigger, MonitorFire):
        return _render_system_reminder("monitor", trigger.payload)

    if isinstance(trigger, SubagentResult):
        return _render_system_reminder("subagent", trigger.payload)

    raise LookupError(f"trigger source {source!r} has no registered TriggerRenderer")


# --- Turn → messages --------------------------------------------------------


def turn_to_messages(
    turn: Turn,
    renderers: dict[str, TriggerRenderer] | None = None,
    *,
    include_non_replayable: bool = False,
) -> list[AgentMessage]:
    """Extract the AgentMessages that a committed Turn contributes to context."""

    if not turn.outcome.cause.replayable and not include_non_replayable:
        return []

    messages: list[AgentMessage] = []
    messages.extend(
        apply_trigger_metadata(
            render_trigger(turn.trigger, renderers),
            turn.trigger_metadata,
        )
    )

    if turn.response is not None:
        messages.append(turn.response)
        if turn.tool_results:
            result_blocks = [
                ToolResultBlock(
                    type="tool_result",
                    tool_call_id=tr.call.id,
                    content=list(tr.result.content),
                    is_error=tr.result.is_error,
                    deterministic=tr.result.deterministic,
                    extras=tr.result.extras,
                )
                for tr in turn.tool_results
            ]
            messages.append(
                ToolResultMessage(
                    role="tool_result",
                    content=result_blocks,
                    timestamp=0.0,
                )
            )
    messages.extend(turn.outcome.injected)

    return messages


def apply_trigger_metadata(
    messages: Sequence[AgentMessage],
    metadata: TriggerMetadata | None,
) -> list[AgentMessage]:
    """Project durable trigger controls onto provider-facing messages."""

    if metadata is None:
        return list(messages)
    projected: list[AgentMessage] = []
    for message in messages:
        tags = dict(message.meta.tags)
        tags.update(metadata.meta)
        projected.append(
            replace(
                message,
                meta=replace(
                    message.meta,
                    origin=metadata.origin or message.meta.origin,
                    visibility=(
                        "hidden" if metadata.is_meta else message.meta.visibility
                    ),
                    target_session_id=(
                        metadata.target_session_id or message.meta.target_session_id
                    ),
                    target_agent_id=(
                        metadata.target_agent_id or message.meta.target_agent_id
                    ),
                    mode=metadata.mode or message.meta.mode,
                    tags=tags,
                ),
            )
        )
    return projected


def route_messages(
    messages: Sequence[AgentMessage],
    *,
    session_id: str,
    agent_id: str | None = None,
) -> list[AgentMessage]:
    """Keep unaddressed messages and messages addressed to this session."""

    effective_agent_id = agent_id or session_id
    return [
        message
        for message in messages
        if message.meta.target_session_id in (None, session_id)
        and message.meta.target_agent_id in (None, effective_agent_id)
    ]


# --- build_context ----------------------------------------------------------


async def build_context(
    turns: Sequence[Turn],
    policies: Sequence[ContextPolicy] = (),
    renderers: dict[str, TriggerRenderer] | None = None,
    signal: CancelSignal | None = None,
) -> list[AgentMessage]:
    """Build the message list from committed turns, then apply policies.

    Async so policies can make LLM calls (compaction).
    """

    replayable_turns = tuple(turn for turn in turns if turn.outcome.cause.replayable)
    messages: list[AgentMessage] = []
    for turn in replayable_turns:
        messages.extend(turn_to_messages(turn, renderers))

    return await apply_context_policies(
        messages,
        replayable_turns,
        policies,
        signal=signal,
    )


async def apply_context_policies(
    messages: list[AgentMessage],
    turns: Sequence[Turn],
    policies: Sequence[ContextPolicy],
    *,
    signal: CancelSignal | None = None,
) -> list[AgentMessage]:
    """Apply policies while preserving per-turn cancellation semantics."""

    for policy in policies:
        if signal is not None and signal.is_set():
            raise ContextTransformCancelled
        try:
            if signal is not None and isinstance(policy, CancellableContextPolicy):
                messages = await policy.transform_with_signal(
                    messages,
                    turns,
                    signal=signal,
                )
            else:
                messages = await policy.transform(messages, turns)
        except Exception as exc:
            if signal is not None and signal.is_set():
                raise ContextTransformCancelled from exc
            raise
        if signal is not None and signal.is_set():
            raise ContextTransformCancelled
    return messages


def build_context_sync(
    turns: Sequence[Turn],
    renderers: dict[str, TriggerRenderer] | None = None,
) -> list[AgentMessage]:
    """Sync context build WITHOUT policies.  For offline tools / queries."""

    messages: list[AgentMessage] = []
    for turn in turns:
        messages.extend(turn_to_messages(turn, renderers))
    return messages


__all__ = [
    "BindableContextPolicy",
    "CancellableContextPolicy",
    "ContextPolicy",
    "ContextTransformCancelled",
    "PolicyContext",
    "apply_trigger_metadata",
    "apply_context_policies",
    "build_context",
    "build_context_sync",
    "render_trigger",
    "route_messages",
    "turn_to_messages",
]
