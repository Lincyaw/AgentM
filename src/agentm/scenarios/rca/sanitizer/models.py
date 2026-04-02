"""Data models for the RCA investigation sanitizer system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agentm.harness.types import LoopContext
    from agentm.scenarios.rca.hypothesis_store import HypothesisStore
    from agentm.scenarios.rca.sanitizer.tracker import InvestigationTracker
    from agentm.scenarios.rca.service_profile import ServiceProfileStore


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


class Sanitizer(Protocol):
    """Protocol for individual sanitizer checks."""

    def check(
        self,
        trigger: str,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        tracker: InvestigationTracker,
        ctx: LoopContext,
    ) -> list[SanitizerFinding]: ...
