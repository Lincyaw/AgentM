"""Trigger types — what causes a Turn to begin.

Trigger is an open Protocol: atoms can define new trigger kinds by
satisfying the ``source`` property.  The trajectory stores the trigger
on the Turn without interpreting it.  Context reconstruction uses
registered TriggerRenderers to convert triggers into AgentMessages.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.messages import (
    AgentMessage,
    ImageContent,
    TextContent,
)

TriggerPriority = Literal["now", "next", "later"]
_TRIGGER_PRIORITY_RANK: dict[TriggerPriority, int] = {
    "now": 0,
    "next": 1,
    "later": 2,
}


@runtime_checkable
class Trigger(Protocol):
    """Any object that triggers a new Turn.  Open type — atoms may define
    additional trigger kinds as long as they satisfy this protocol."""

    @property
    def source(self) -> str: ...


@runtime_checkable
class TriggerRenderer(Protocol):
    """Convert a Trigger into AgentMessages for context reconstruction.

    Registered per ``source`` string.  If no renderer is registered for
    a trigger's source, the default renderer wraps ``str(trigger)`` in
    a system-reminder UserMessage.
    """

    def render(self, trigger: Trigger) -> list[AgentMessage]: ...


@dataclass(frozen=True, slots=True)
class TriggerEnvelope:
    """Control-plane metadata for a queued trigger.

    The wrapped ``trigger`` is the durable input stored on a Turn. Everything
    else describes queueing and routing policy: priority, target, origin, and
    presenter/system metadata. Hosts may keep their own process-level queues;
    this envelope is the SDK boundary they map into a session.
    """

    trigger: Trigger
    priority: TriggerPriority = "next"
    target_session_id: str | None = None
    target_agent_id: str | None = None
    origin: str | None = None
    mode: str = "prompt"
    is_meta: bool = False
    skip_commands: bool = False
    meta: Mapping[str, object] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return getattr(self.trigger, "source", "unknown")


def trigger_priority_rank(priority: TriggerPriority) -> int:
    """Return the numeric priority order used by queues."""

    return _TRIGGER_PRIORITY_RANK[priority]


# --- Built-in trigger types -------------------------------------------------


@dataclass(frozen=True, slots=True)
class UserInput:
    """Direct user input (text and/or images)."""

    content: tuple[TextContent | ImageContent, ...]
    source: str = "user"


@dataclass(frozen=True, slots=True)
class BackgroundCompletion:
    """A backgrounded tool finished (or errored)."""

    task_id: str
    payload: str
    terminal: bool = False
    source: str = "background"


@dataclass(frozen=True, slots=True)
class MonitorFire:
    """A scheduled or condition-based monitor fired."""

    monitor_id: str
    payload: str
    source: str = "monitor"


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """A child session completed and posted its result."""

    child_session_id: str
    payload: str
    terminal: bool = False
    source: str = "subagent"


@dataclass(frozen=True, slots=True)
class ContinueTrigger:
    """Resume/replay: continue from the current trajectory with no new input."""

    source: str = "continue"


@dataclass(frozen=True, slots=True)
class Injection:
    """Extension-injected messages (from a decide_turn_action Inject override)."""

    messages: tuple[AgentMessage, ...]
    source: str = "injection"


__all__ = [
    "BackgroundCompletion",
    "ContinueTrigger",
    "Injection",
    "MonitorFire",
    "SubagentResult",
    "Trigger",
    "TriggerEnvelope",
    "TriggerPriority",
    "TriggerRenderer",
    "UserInput",
    "trigger_priority_rank",
]
