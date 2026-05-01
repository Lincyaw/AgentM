"""Data models for the RCA investigation sanitizer system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agentm_rca.stores import HypothesisStore, ServiceProfileStore
    from agentm_rca.sanitizer.tracker import InvestigationTracker


class Severity(str, Enum):
    """Severity level for sanitizer findings."""

    BLOCK = "BLOCK"
    WARN = "WARN"
    INFO = "INFO"


@dataclass(frozen=True)
class SanitizerFinding:
    """A single finding produced by a sanitizer check."""

    code: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvestigationEvent:
    """A recorded event during an RCA investigation."""

    round: int
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SanitizerContext:
    """Minimal loop context required by the ported sanitizer checks."""

    agent_id: str
    step: int
    max_steps: int | None
    tool_call_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Sanitizer(Protocol):
    """Protocol for individual sanitizer checks."""

    def check(
        self,
        trigger: str,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        tracker: InvestigationTracker,
        ctx: SanitizerContext,
    ) -> list[SanitizerFinding]: ...
