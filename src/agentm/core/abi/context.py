"""Context reconstruction — async policies and pluggable transforms.

Context is a COMPUTED view, not stored data.  ``build_context`` walks
the committed turns, converts each to messages, then applies registered
ContextPolicies in priority order.

Policies are the pluggability axis for compaction, injection, cache
discipline, system reminders, and any future context transformation.

Two context-transform mechanisms exist by design:

- **ContextPolicy** — runs inside ``build_context`` at the START of
  each round.  Sees committed turns only.  Async.  Has access to
  session identity and services via ``bind()``.  Use for transforms
  that depend on the full trajectory structure (compaction, summary).

- **ContextEvent** (bus) — runs AFTER ``build_context``, inside the
  round loop.  Sees committed turns + trigger + in-flight messages.
  Async.  Use for transforms that need the live tail (cache-discipline
  suffix, token injection).

The split is deliberate: policies transform the durable prefix,
bus handlers transform the live tail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentm.core.abi.stream import Model, StreamFn

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
    TriggerRenderer,
    UserInput,
)

@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Session-scoped context given to a ContextPolicy at bind time."""

    session_id: str = ""
    parent_session_id: str | None = None
    services: dict[str, Any] | None = None
    store: Any | None = None
    model: "Model | None" = None
    stream_fn: "StreamFn | None" = None


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

    source = getattr(trigger, "source", "unknown")
    if renderers and source in renderers:
        return renderers[source].render(trigger)  # type: ignore[arg-type]

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

    return _render_system_reminder(source, str(trigger))


# --- Turn → messages --------------------------------------------------------


def turn_to_messages(
    turn: Turn,
    renderers: dict[str, TriggerRenderer] | None = None,
) -> list[AgentMessage]:
    """Extract the AgentMessages that a committed Turn contributes to context."""

    messages: list[AgentMessage] = []
    messages.extend(render_trigger(turn.trigger, renderers))

    injected_by_round = {
        injection.after_round: injection.messages
        for injection in turn.outcome.injected
    }
    for round_index, rnd in enumerate(turn.rounds):
        messages.append(rnd.response)
        if rnd.tool_results:
            result_blocks = [
                ToolResultBlock(
                    type="tool_result",
                    tool_call_id=tr.call.id,
                    content=list(tr.result.content),
                    is_error=tr.result.is_error,
                    deterministic=tr.result.deterministic,
                    extras=tr.result.extras,
                )
                for tr in rnd.tool_results
            ]
            messages.append(
                ToolResultMessage(
                    role="tool_result",
                    content=result_blocks,
                    timestamp=0.0,
                )
            )
        messages.extend(injected_by_round.get(round_index, ()))

    return messages


# --- build_context ----------------------------------------------------------


async def build_context(
    turns: Sequence[Turn],
    policies: Sequence[ContextPolicy] = (),
    renderers: dict[str, TriggerRenderer] | None = None,
) -> list[AgentMessage]:
    """Build the message list from committed turns, then apply policies.

    Async so policies can make LLM calls (compaction).
    """

    messages: list[AgentMessage] = []
    for turn in turns:
        messages.extend(turn_to_messages(turn, renderers))

    for policy in policies:
        messages = await policy.transform(messages, turns)

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
    "ContextPolicy",
    "PolicyContext",
    "build_context",
    "build_context_sync",
    "render_trigger",
    "turn_to_messages",
]
