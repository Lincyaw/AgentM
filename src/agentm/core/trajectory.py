"""Trajectory collection — structured event capture to JSONL."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Sequence
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


class _ReadOnlyEventView(Sequence[dict[str, Any]]):
    """Lightweight read-only view over the internal events list.

    Avoids a full copy on every ``.events`` access while preventing
    accidental mutation by external callers.
    """

    __slots__ = ("_data",)

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    def __getitem__(self, index: int | slice) -> Any:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"_ReadOnlyEventView(len={len(self._data)})"


class TrajectoryCollector:
    """Collects execution events and writes them as JSONL.

    One file per run at ``{output_dir}/{run_id}.jsonl``.
    Thread-safe via ``asyncio.Lock``.

    Args:
        max_memory_events: When set, old events are evicted from the
            in-memory buffer once this limit is reached (they are already
            persisted to the JSONL file on disk).  ``None`` means no limit.
        flush_batch_size: Number of buffered lines before an immediate
            async flush is triggered.
        flush_interval: Seconds between periodic background flushes of
            the async write buffer.
    """

    def __init__(
        self,
        run_id: str,
        output_dir: str = "./trajectories",
        thread_id: str = "",
        checkpoint_db: str = "",
        *,
        max_memory_events: int | None = None,
        flush_batch_size: int = 10,
        flush_interval: float = 1.0,
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
        self._max_memory_events = max_memory_events
        self._listeners: list[Callable[[dict[str, Any]], Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

        # Async write buffer (D3)
        self._write_buffer: list[str] = []
        self._flush_batch_size = flush_batch_size
        self._flush_interval = flush_interval
        self._flush_task: asyncio.Task[None] | None = None

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
    def events(self) -> Sequence[dict[str, Any]]:
        """Read-only view of in-memory events (no copy).

        Note: if ``max_memory_events`` is set, old events may have been
        evicted.  Use :meth:`read_all_events` for the full history.
        """
        return _ReadOnlyEventView(self._events)

    @property
    def file_path(self) -> Path:
        return self._output_dir / f"{self._run_id}.jsonl"

    @classmethod
    def read_all_events(cls, path: Path) -> list[dict[str, Any]]:
        """Read all events from a trajectory JSONL file on disk.

        This is the reliable way to obtain the full event history when
        in-memory events may have been evicted due to ``max_memory_events``.
        """
        _, events = read_trajectory(path)
        return events

    def _build_event(
        self,
        event_type: str,
        agent_path: list[str],
        data: dict[str, Any],
        node_name: str,
        task_id: str | None,
        metadata: dict[str, Any] | None,
        parent_seq: int | None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str, int]:
        """Create a TrajectoryEvent and return (dumped_dict, jsonl_line, seq)."""
        _meta = dict(metadata) if metadata else {}
        if "hypothesis_id" in kwargs:
            _meta.setdefault("hypothesis_id", kwargs.pop("hypothesis_id"))

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
        dumped = event.model_dump(mode="json")
        line = json.dumps(dumped, ensure_ascii=False) + "\n"
        return dumped, line, self._seq

    def _append_event(self, dumped: dict[str, Any]) -> None:
        """Append event to in-memory buffer, evicting old entries if needed."""
        self._events.append(dumped)
        if (
            self._max_memory_events is not None
            and len(self._events) > self._max_memory_events
        ):
            # Evict oldest events (already persisted to JSONL on disk)
            overflow = len(self._events) - self._max_memory_events
            del self._events[:overflow]

    def _record_core(
        self,
        event_type: str,
        agent_path: list[str],
        data: dict[str, Any],
        node_name: str,
        task_id: str | None,
        metadata: dict[str, Any] | None,
        parent_seq: int | None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int]:
        """Record an event with synchronous direct file write.

        Used by ``record_sync()`` — writes directly to avoid cross-thread
        buffer issues with the async write buffer.
        """
        dumped, line, seq = self._build_event(
            event_type, agent_path, data, node_name,
            task_id, metadata, parent_seq, **kwargs,
        )
        self._ensure_file()
        if self._file is None:
            raise RuntimeError(
                "Trajectory file not initialized after _ensure_file()"
            )
        self._file.write(line)
        self._file.flush()
        self._append_event(dumped)
        return dumped, seq

    def _record_core_buffered(
        self,
        event_type: str,
        agent_path: list[str],
        data: dict[str, Any],
        node_name: str,
        task_id: str | None,
        metadata: dict[str, Any] | None,
        parent_seq: int | None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, bool]:
        """Record an event into the async write buffer.

        Returns ``(dumped_dict, seq, should_flush)`` where *should_flush*
        is ``True`` when the buffer has reached ``_flush_batch_size``.
        """
        dumped, line, seq = self._build_event(
            event_type, agent_path, data, node_name,
            task_id, metadata, parent_seq, **kwargs,
        )
        self._write_buffer.append(line)
        self._append_event(dumped)
        should_flush = len(self._write_buffer) >= self._flush_batch_size
        return dumped, seq, should_flush

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

        Uses a buffered async write strategy: events are appended to an
        in-memory buffer and flushed to disk either when the batch size
        is reached or by a periodic background task.
        """
        async with self._lock:
            dumped, seq, should_flush = self._record_core_buffered(
                event_type, agent_path, data, node_name,
                task_id, metadata, parent_seq, **kwargs,
            )
            if should_flush:
                await self._flush()
            # Start periodic flush task on first async record
            if self._flush_task is None:
                self._flush_task = asyncio.create_task(
                    self._periodic_flush()
                )
            await self._notify_listeners(dumped)
            return seq

    async def _flush(self) -> None:
        """Flush the async write buffer to disk via a thread."""
        if not self._write_buffer:
            return
        lines = self._write_buffer
        self._write_buffer = []
        await asyncio.to_thread(self._write_lines_sync, lines)

    def _write_lines_sync(self, lines: list[str]) -> None:
        """Write buffered lines to the JSONL file (runs in a thread)."""
        self._ensure_file()
        if self._file is None:
            raise RuntimeError(
                "Trajectory file not initialized after _ensure_file()"
            )
        self._file.writelines(lines)
        self._file.flush()

    async def _periodic_flush(self) -> None:
        """Background task that flushes the write buffer periodically."""
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                async with self._lock:
                    await self._flush()
        except asyncio.CancelledError:
            return

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
        with self._sync_lock:
            dumped, seq = self._record_core(
                event_type, agent_path, data, node_name,
                task_id, metadata, parent_seq, **kwargs,
            )
            self._notify_listeners_sync(dumped)
            return seq

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
        """Flush remaining buffer, cancel background task, and close file.

        Returns the file path if a file was opened.
        """
        # Cancel periodic flush task
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Flush remaining buffered lines
        async with self._lock:
            await self._flush()

        if self._file is not None:
            self._file.flush()
            self._file.close()
            path = str(self._file.name)
            self._file = None
            logger.info("Trajectory saved: %s (%d events)", path, len(self._events))
            return path
        return None


def read_trajectory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a trajectory JSONL file, returning (metadata_dict, event_list).

    The first line with a ``_meta`` key is treated as metadata header.
    All other valid JSON lines are returned as events.
    """
    meta: dict[str, Any] = {}
    events: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if "_meta" in data:
                meta = data["_meta"]
                continue
            events.append(data)

    return meta, events
