"""Trajectory collection — structured event capture to JSONL."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrajectoryEvent(BaseModel):
    """A single event in an AgentM execution trace. One per JSONL line."""

    run_id: str
    seq: int
    timestamp: str
    agent_path: list[str]
    node_name: str
    event_type: str
    data: dict[str, Any]
    task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_seq: int | None = None


class TrajectoryCollector:
    """Collects execution events and writes them as JSONL.

    One file per run at ``{output_dir}/{run_id}.jsonl``.
    Thread-safe via ``asyncio.Lock``.
    """

    def __init__(
        self,
        run_id: str,
        output_dir: str = "./trajectories",
        thread_id: str = "",
        checkpoint_db: str = "",
    ) -> None:
        self._run_id = run_id
        self._output_dir = Path(output_dir)
        self._thread_id = thread_id
        self._checkpoint_db = checkpoint_db
        self._seq = 0
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._file: IO[str] | None = None
        self._meta_written = False
        self._events: list[dict[str, Any]] = []
        self._listeners: list[Callable[[dict[str, Any]], Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

    def add_listener(self, listener: Callable[[dict[str, Any]], Any]) -> None:
        """Register a callback invoked on every recorded event.

        Listeners receive the TrajectoryEvent dict after it has been written
        to JSONL and buffered in memory. Async and sync callables are both
        supported.
        """
        self._listeners.append(listener)
        # Capture event loop on first async listener registration so
        # _notify_listeners_sync can schedule coroutines from other threads.
        if self._loop is None and asyncio.iscoroutinefunction(listener):
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    async def _notify_listeners(self, event_dict: dict[str, Any]) -> None:
        """Fan out event to all registered listeners (async context)."""
        for listener in self._listeners:
            if asyncio.iscoroutinefunction(listener):
                await listener(event_dict)
            else:
                listener(event_dict)

    def _notify_listeners_sync(self, event_dict: dict[str, Any]) -> None:
        """Fan out event from sync context. Schedules async listeners as tasks."""
        for listener in self._listeners:
            if asyncio.iscoroutinefunction(listener):
                # Try current thread's loop first, fall back to saved loop.
                try:
                    asyncio.get_running_loop().create_task(listener(event_dict))
                except RuntimeError:
                    if self._loop is not None and self._loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            listener(event_dict), self._loop
                        )
                    else:
                        logger.warning(
                            "Dropping async listener %r: no running event loop",
                            getattr(listener, "__name__", listener),
                        )
            else:
                listener(event_dict)

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
        metadata: dict[str, Any] | None = None,
        parent_seq: int | None = None,
        **kwargs: Any,
    ) -> int:
        """Record an event. Returns the assigned sequence number.

        Scenarios can pass arbitrary key-value pairs via *metadata*
        (e.g. ``metadata={"hypothesis_id": "H1"}``).
        Legacy callers passing ``hypothesis_id=`` directly are handled
        via **kwargs for backward compatibility.
        """
        # Merge legacy hypothesis_id kwarg into metadata
        _meta = dict(metadata) if metadata else {}
        if "hypothesis_id" in kwargs:
            _meta.setdefault("hypothesis_id", kwargs.pop("hypothesis_id"))

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
                metadata=_meta,
                parent_seq=parent_seq,
            )
            dumped = event.model_dump()
            line = event.model_dump_json() + "\n"
            self._ensure_file()
            if self._file is None:
                raise RuntimeError(
                    "Trajectory file not initialized after _ensure_file()"
                )
            self._file.write(line)
            self._file.flush()
            self._events.append(dumped)
            await self._notify_listeners(dumped)
            return self._seq

    def record_sync(
        self,
        event_type: str,
        agent_path: list[str],
        data: dict[str, Any],
        node_name: str = "",
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        parent_seq: int | None = None,
        **kwargs: Any,
    ) -> int:
        """Synchronous variant for use from sync tool functions.

        Writes directly without the async lock — only safe when called from
        within the event loop thread (which sync tool functions are).
        """
        _meta = dict(metadata) if metadata else {}
        if "hypothesis_id" in kwargs:
            _meta.setdefault("hypothesis_id", kwargs.pop("hypothesis_id"))

        with self._sync_lock:
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
                metadata=_meta,
                parent_seq=parent_seq,
            )
            dumped = event.model_dump()
            line = event.model_dump_json() + "\n"
            self._ensure_file()
            if self._file is None:
                raise RuntimeError(
                    "Trajectory file not initialized after _ensure_file()"
                )
            self._file.write(line)
            self._file.flush()
            self._events.append(dumped)
            self._notify_listeners_sync(dumped)
            return self._seq

    def _ensure_file(self) -> None:
        if self._file is None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(self.file_path, "a", encoding="utf-8")
        if not self._meta_written:
            self._meta_written = True
            meta = {
                "_meta": {
                    "run_id": self._run_id,
                    "thread_id": self._thread_id,
                    "checkpoint_db": self._checkpoint_db,
                }
            }
            self._file.write(json.dumps(meta) + "\n")
            self._file.flush()

    @staticmethod
    def read_metadata(path: str | Path) -> dict[str, Any]:
        """Read the _meta header from a trajectory file. Returns {} if absent."""
        with open(path, encoding="utf-8") as f:
            first_line = f.readline()
        try:
            data = json.loads(first_line)
            return data.get("_meta", {})
        except (json.JSONDecodeError, KeyError):
            return {}

    async def close(self) -> str | None:
        """Flush and close. Returns the file path if a file was opened."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            path = str(self._file.name)
            self._file = None
            logger.info("Trajectory saved: %s (%d events)", path, len(self._events))
            return path
        return None
