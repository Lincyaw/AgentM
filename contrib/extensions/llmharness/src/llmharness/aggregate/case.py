"""Case-level data structures + canonical directory layout.

A :class:`CaseData` is the lossless in-memory representation of one
case (one main-agent session on one input). :class:`CaseLayout`
captures the on-disk file/directory naming so collector, writer, and
downstream tools agree on paths without string duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Phase = Literal["extractor", "auditor"]
FiringStatus = Literal["ok", "no_call", "spawn_error", "prompt_error"]


@dataclass(frozen=True, slots=True)
class CaseMeta:
    """Summary metadata written to ``meta.json`` for each case."""

    case_id: str
    session_id: str
    trace_id: str
    sample_id: str | None
    dataset_name: str | None
    dataset_path: str | None
    started_at_ns: int
    ended_at_ns: int
    extractor_firings: int
    auditor_firings: int
    surfaced_reminders: int
    silent_verdicts: int


@dataclass(frozen=True, slots=True)
class FiringRecord:
    """One extractor or auditor invocation."""

    phase: Phase
    sequence: int  # 1-based order within phase
    turn_index: int
    ts_ns: int
    input_payload: dict[str, Any]
    output: dict[str, Any] | None
    status: FiringStatus
    error: str | None
    latency_ms: int
    raw_assistant_messages: list[dict[str, Any]] = field(default_factory=list)
    """Chronological list of serialized assistant content blocks from the
    child loop. Downstream SFT exporters may consume thinking blocks to
    produce Qwen/GLM-style ``<think>...</think>`` training targets."""


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    """Cumulative index state after one extractor firing."""

    after_extractor_firing: int  # 1-based
    turn_index: int
    events: list[dict[str, Any]]
    edges: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CaseData:
    """Lossless in-memory view of one case, ready to serialise."""

    meta: CaseMeta
    main_agent_messages: list[dict[str, Any]]
    extractor_firings: list[FiringRecord] = field(default_factory=list)
    auditor_firings: list[FiringRecord] = field(default_factory=list)
    index_snapshots: list[IndexSnapshot] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CaseLayout:
    """Canonical on-disk layout under a case directory."""

    root: Path

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def main_agent_path(self) -> Path:
        return self.root / "main_agent.jsonl"

    @property
    def extractor_dir(self) -> Path:
        return self.root / "extractor"

    @property
    def auditor_dir(self) -> Path:
        return self.root / "auditor"

    @property
    def index_dir(self) -> Path:
        return self.root / "context_index"

    @property
    def verdicts_path(self) -> Path:
        return self.root / "verdicts.jsonl"

    @property
    def trajectory_path(self) -> Path:
        return self.root / "trajectory.jsonl"

    @property
    def readme_path(self) -> Path:
        return self.root / "README.md"

    def firing_path(self, phase: Phase, sequence: int, turn_index: int) -> Path:
        return (
            self.extractor_dir if phase == "extractor" else self.auditor_dir
        ) / f"{sequence:03d}_turn_{turn_index:03d}.json"

    def snapshot_path(self, after_firing: int) -> Path:
        return self.index_dir / f"after_extractor_{after_firing:03d}.json"


__all__ = [
    "CaseData",
    "CaseLayout",
    "CaseMeta",
    "FiringRecord",
    "FiringStatus",
    "IndexSnapshot",
    "Phase",
]
