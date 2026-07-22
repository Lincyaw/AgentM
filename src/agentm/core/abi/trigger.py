# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Trigger types — what causes a Turn to begin.

Trigger is an open Protocol: atoms can define new trigger kinds by
satisfying the ``source`` property.  The trajectory stores the trigger
on the Turn without interpreting it.  Context reconstruction uses
registered TriggerRenderers to convert triggers into AgentMessages.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol, runtime_checkable

from agentm.core.abi.messages import (
    AgentMessage,
    ImageContent,
    JsonValue,
    TextContent,
    freeze_json,
)

TriggerPriority = Literal["now", "next", "later"]
_TRIGGER_PRIORITY_RANK: Final[dict[TriggerPriority, int]] = {
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

    Registered per ``source`` string. Custom trigger sources must install a
    renderer; the runtime never derives durable prompt content from object
    ``str`` or ``repr`` output.
    """

    def render(self, trigger: Trigger) -> list[AgentMessage]: ...


@dataclass(frozen=True, slots=True)
class TriggerMetadata:
    """Durable queue and routing metadata attached to a turn."""

    priority: TriggerPriority = "next"
    target_session_id: str | None = None
    target_agent_id: str | None = None
    origin: str | None = None
    mode: str = "prompt"
    is_meta: bool = False
    skip_commands: bool = False
    meta: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.priority not in _TRIGGER_PRIORITY_RANK:
            raise ValueError(f"invalid trigger priority: {self.priority!r}")
        for label, value in (
            ("target_session_id", self.target_session_id),
            ("target_agent_id", self.target_agent_id),
            ("origin", self.origin),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{label} must be a string or None")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("trigger mode must be a non-empty string")
        object.__setattr__(self, "meta", freeze_json(self.meta))


@dataclass(frozen=True, slots=True)
class TriggerEnvelope:
    """One queued trigger plus metadata that survives durable commit.

    Hosts route to a concrete session before pushing an envelope. A session
    queue is intentionally not a hidden process-global router.
    """

    trigger: Trigger
    metadata: TriggerMetadata = field(default_factory=TriggerMetadata)

    @property
    def source(self) -> str:
        return self.trigger.source

    @property
    def priority(self) -> TriggerPriority:
        return self.metadata.priority


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
class CompactTrigger:
    """Run a maintenance compaction pass without creating a conversation turn."""

    source: str = "compact"


@dataclass(frozen=True, slots=True)
class Injection:
    """Extension-injected messages (from a decide_turn_action Inject override)."""

    messages: tuple[AgentMessage, ...]
    source: str = "injection"


__all__ = [
    "BackgroundCompletion",
    "CompactTrigger",
    "ContinueTrigger",
    "Injection",
    "MonitorFire",
    "SubagentResult",
    "Trigger",
    "TriggerEnvelope",
    "TriggerMetadata",
    "TriggerPriority",
    "TriggerRenderer",
    "UserInput",
    "trigger_priority_rank",
]
