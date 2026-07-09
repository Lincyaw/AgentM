"""Unified task result envelope — common schema across all benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskResult:
    task_id: str
    status: str  # pass, fail, error, skipped
    score: dict[str, Any] = field(default_factory=dict)
    session_ids: list[str] = field(default_factory=list)
    latency_ms: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status,
            "score": self.score,
            "session_ids": self.session_ids,
            "latency_ms": self.latency_ms,
        }
        if self.error:
            d["error"] = self.error
        if self.metadata:
            d["metadata"] = self.metadata
        return d
