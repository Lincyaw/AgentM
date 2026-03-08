"""Trajectory collection — structured event capture to JSONL."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Optional

from pydantic import BaseModel


class TrajectoryEvent(BaseModel):
    """A single event in an AgentM execution trace. One per JSONL line."""

    run_id: str
    seq: int
    timestamp: str
    agent_path: list[str]
    node_name: str
    event_type: str
    data: dict[str, Any]
    task_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    parent_seq: Optional[int] = None


class TrajectoryCollector:
    """Collects execution events and writes them as JSONL.

    One file per run at ``{output_dir}/{run_id}.jsonl``.
    Thread-safe via ``asyncio.Lock``.
    """

    def __init__(self, run_id: str, output_dir: str = "./trajectories") -> None:
        self._run_id = run_id
        self._output_dir = Path(output_dir)
        self._seq = 0
        self._lock = asyncio.Lock()
        self._file: IO[str] | None = None
        self._events: list[dict[str, Any]] = []

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def events(self) -> list[dict[str, Any]]:
        """In-memory copy of all recorded events."""
        return list(self._events)

    @property
    def file_path(self) -> Path:
        return self._output_dir / f"{self._run_id}.jsonl"

    async def record(
        self,
        event_type: str,
        agent_path: list[str],
        data: dict[str, Any],
        node_name: str = "",
        task_id: str | None = None,
        hypothesis_id: str | None = None,
        parent_seq: int | None = None,
    ) -> int:
        """Record an event. Returns the assigned sequence number."""
        async with self._lock:
            self._seq += 1
            event = TrajectoryEvent(
                run_id=self._run_id,
                seq=self._seq,
                timestamp=datetime.now().isoformat(),
                agent_path=agent_path,
                node_name=node_name,
                event_type=event_type,
                data=data,
                task_id=task_id,
                hypothesis_id=hypothesis_id,
                parent_seq=parent_seq,
            )
            dumped = event.model_dump()
            line = event.model_dump_json() + "\n"
            self._ensure_file()
            assert self._file is not None
            self._file.write(line)
            self._file.flush()
            self._events.append(dumped)
            return self._seq

    def record_sync(
        self,
        event_type: str,
        agent_path: list[str],
        data: dict[str, Any],
        node_name: str = "",
        task_id: str | None = None,
        hypothesis_id: str | None = None,
        parent_seq: int | None = None,
    ) -> int:
        """Synchronous variant for use from sync tool functions.

        Writes directly without the async lock — only safe when called from
        within the event loop thread (which sync tool functions are).
        """
        self._seq += 1
        event = TrajectoryEvent(
            run_id=self._run_id,
            seq=self._seq,
            timestamp=datetime.now().isoformat(),
            agent_path=agent_path,
            node_name=node_name,
            event_type=event_type,
            data=data,
            task_id=task_id,
            hypothesis_id=hypothesis_id,
            parent_seq=parent_seq,
        )
        dumped = event.model_dump()
        line = event.model_dump_json() + "\n"
        self._ensure_file()
        assert self._file is not None
        self._file.write(line)
        self._file.flush()
        self._events.append(dumped)
        return self._seq

    def _ensure_file(self) -> None:
        if self._file is None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(self.file_path, "a", encoding="utf-8")

    async def close(self) -> str | None:
        """Flush and close. Returns the file path if a file was opened."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            path = str(self._file.name)
            self._file = None
            return path
        return None
