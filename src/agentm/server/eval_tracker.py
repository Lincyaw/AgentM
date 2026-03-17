"""Thread-safe batch evaluation state tracker for the Eval Dashboard.

Tracks per-sample status across the eval pipeline and notifies
listeners (WebSocket broadcaster) on every state change.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class SampleStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


@dataclass(frozen=True)
class SampleInfo:
    sample_id: str
    dataset_index: int
    data_dir: str
    status: SampleStatus = SampleStatus.pending
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    run_id: str | None = None
    trajectory_path: str | None = None


class EvalTracker:
    """Thread-safe tracker for batch evaluation sample state.

    All mutations are guarded by a threading.Lock so both the async
    event loop thread and the eval runner can safely update state.
    """

    def __init__(self, trajectory_dir: str = "./trajectories") -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, SampleInfo] = {}
        self._order: list[str] = []  # insertion-order sample ids
        self._listeners: list[Callable[[dict[str, Any]], Any]] = []
        self._trajectory_dir = trajectory_dir

    # -- Registration ----------------------------------------------------------

    def register_sample(
        self,
        sample_id: str,
        dataset_index: int,
        data_dir: str,
    ) -> None:
        with self._lock:
            info = SampleInfo(
                sample_id=sample_id,
                dataset_index=dataset_index,
                data_dir=data_dir,
            )
            self._samples[sample_id] = info
            self._order.append(sample_id)

    # -- Status transitions ----------------------------------------------------

    def mark_running(self, sample_id: str, run_id: str) -> None:
        with self._lock:
            info = self._samples.get(sample_id)
            if info is None:
                return
            traj_path = str(Path(self._trajectory_dir) / f"{run_id}.jsonl")
            updated = replace(
                info,
                status=SampleStatus.running,
                started_at=datetime.now().isoformat(),
                run_id=run_id,
                trajectory_path=traj_path,
            )
            self._samples[sample_id] = updated
        self._notify(updated)

    def mark_completed(self, sample_id: str) -> None:
        with self._lock:
            info = self._samples.get(sample_id)
            if info is None:
                return
            now = datetime.now().isoformat()
            duration = None
            if info.started_at:
                try:
                    start = datetime.fromisoformat(info.started_at)
                    duration = (datetime.now() - start).total_seconds()
                except ValueError:
                    pass
            updated = replace(
                info,
                status=SampleStatus.completed,
                completed_at=now,
                duration_seconds=duration,
            )
            self._samples[sample_id] = updated
        self._notify(updated)

    def mark_failed(self, sample_id: str, error: str = "") -> None:
        with self._lock:
            info = self._samples.get(sample_id)
            if info is None:
                return
            now = datetime.now().isoformat()
            duration = None
            if info.started_at:
                try:
                    start = datetime.fromisoformat(info.started_at)
                    duration = (datetime.now() - start).total_seconds()
                except ValueError:
                    pass
            updated = replace(
                info,
                status=SampleStatus.failed,
                completed_at=now,
                duration_seconds=duration,
                error=error,
            )
            self._samples[sample_id] = updated
        self._notify(updated)

    def mark_skipped(self, sample_id: str, reason: str = "") -> None:
        with self._lock:
            info = self._samples.get(sample_id)
            if info is None:
                return
            updated = replace(
                info,
                status=SampleStatus.skipped,
                error=reason,
            )
            self._samples[sample_id] = updated
        self._notify(updated)

    def update_trajectory_path(self, sample_id: str, trajectory_path: str) -> None:
        """Update the trajectory file path for a sample after the real path is known."""
        with self._lock:
            info = self._samples.get(sample_id)
            if info is None:
                return
            updated = replace(info, trajectory_path=trajectory_path)
            self._samples[sample_id] = updated
        # No broadcast — this is a metadata correction, not a status change

    # -- Queries ---------------------------------------------------------------

    def get_summary(self) -> dict[str, int]:
        with self._lock:
            counts = {s.value: 0 for s in SampleStatus}
            for info in self._samples.values():
                counts[info.status.value] += 1
            counts["total"] = len(self._samples)
            return counts

    def get_samples(
        self,
        offset: int = 0,
        limit: int = 50,
        status_filter: str | None = None,
        search: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            filtered: list[SampleInfo] = []
            for sid in self._order:
                info = self._samples[sid]
                if status_filter and info.status.value != status_filter:
                    continue
                if search:
                    q = search.lower()
                    if (
                        q not in info.sample_id.lower()
                        and q not in info.data_dir.lower()
                        and q not in str(info.dataset_index)
                    ):
                        continue
                filtered.append(info)

            total = len(filtered)
            page = filtered[offset : offset + limit]
            return [_sample_to_dict(s) for s in page], total

    def get_sample(self, sample_id: str) -> dict[str, Any] | None:
        with self._lock:
            info = self._samples.get(sample_id)
            if info is None:
                return None
            return _sample_to_dict(info)

    # -- Listener management ---------------------------------------------------

    def add_listener(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        self._listeners.append(callback)

    def _notify(self, info: SampleInfo) -> None:
        event = {
            "channel": "eval",
            "event_type": "sample_status",
            "data": _sample_to_dict(info),
        }
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass


def _sample_to_dict(info: SampleInfo) -> dict[str, Any]:
    return {
        "sample_id": info.sample_id,
        "dataset_index": info.dataset_index,
        "data_dir": info.data_dir,
        "status": info.status.value,
        "started_at": info.started_at,
        "completed_at": info.completed_at,
        "duration_seconds": info.duration_seconds,
        "error": info.error,
        "run_id": info.run_id,
        "trajectory_path": info.trajectory_path,
    }
