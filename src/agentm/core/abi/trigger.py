"""Trigger types — what causes a Turn to begin.

Trigger is an open Protocol: atoms can define new trigger kinds by
satisfying the ``source`` property.  The trajectory stores the trigger
on the Turn without interpreting it.  Context reconstruction uses
registered TriggerRenderers to convert triggers into AgentMessages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentm.core.abi.messages import (
    AgentMessage,
    ImageContent,
    TextContent,
)


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
    "TriggerRenderer",
    "UserInput",
]
