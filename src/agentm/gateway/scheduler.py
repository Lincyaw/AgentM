"""Durable gateway-owned scheduled prompts.

This is host/gateway plumbing, not an atom: jobs survive session process
teardown because they live under the gateway state directory, and firing a job
uses the gateway's normal inbound path to push a prompt into the target
session's inbox.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


class CronError(ValueError):
    """Raised when a 5-field cron expression cannot be parsed."""


@dataclass(frozen=True, slots=True)
class _CronSchedule:
    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    dom_constrained: bool
    dow_constrained: bool

    @classmethod
    def parse(cls, expr: str) -> _CronSchedule:
        fields = expr.split()
        if len(fields) != 5:
            raise CronError("cron expression must have exactly 5 fields")
        minute, hour, dom, month, dow = fields
        return cls(
            minutes=_parse_field(minute, 0, 59),
            hours=_parse_field(hour, 0, 23),
            days_of_month=_parse_field(dom, 1, 31),
            months=_parse_field(month, 1, 12),
            days_of_week=_parse_field(dow, 0, 6, sunday_7=True),
            dom_constrained=dom != "*",
            dow_constrained=dow != "*",
        )

    def matches(self, dt: datetime) -> bool:
        if dt.minute not in self.minutes:
            return False
        if dt.hour not in self.hours:
            return False
        if dt.month not in self.months:
            return False
        dom_match = dt.day in self.days_of_month
        # Python: Monday=0..Sunday=6. Cron: Sunday=0, Monday=1..Saturday=6.
        cron_dow = (dt.weekday() + 1) % 7
        dow_match = cron_dow in self.days_of_week
        if self.dom_constrained and self.dow_constrained:
            return dom_match or dow_match
        return dom_match and dow_match

    def next_after(self, after_ts: float) -> float:
        dt = datetime.fromtimestamp(after_ts).replace(second=0, microsecond=0)
        dt += timedelta(minutes=1)
        # Five years covers sparse expressions such as leap-day schedules while
        # still keeping invalid-impossible schedules bounded.
        for _ in range(5 * 366 * 24 * 60):
            if self.matches(dt):
                return dt.timestamp()
            dt += timedelta(minutes=1)
        raise CronError("cron expression has no matching time in the next 5 years")


def _parse_field(
    text: str,
    min_value: int,
    max_value: int,
    *,
    sunday_7: bool = False,
) -> frozenset[int]:
    if not text:
        raise CronError("empty cron field")
    values: set[int] = set()
    for part in text.split(","):
        if not part:
            raise CronError(f"empty list item in cron field {text!r}")
        values.update(
            _parse_field_part(
                part,
                min_value,
                max_value,
                sunday_7=sunday_7,
            )
        )
    return frozenset(values)


def _parse_field_part(
    part: str,
    min_value: int,
    max_value: int,
    *,
    sunday_7: bool,
) -> set[int]:
    range_part, slash, step_part = part.partition("/")
    step = 1
    if slash:
        try:
            step = int(step_part)
        except ValueError as exc:
            raise CronError(f"invalid step {step_part!r}") from exc
        if step <= 0:
            raise CronError("cron step must be positive")

    if range_part == "*":
        start = min_value
        end = max_value
    elif "-" in range_part:
        raw_start, raw_end = range_part.split("-", 1)
        start = _parse_int(raw_start, min_value, max_value, sunday_7=sunday_7)
        end = _parse_int(raw_end, min_value, max_value, sunday_7=sunday_7)
        if start > end:
            raise CronError(f"invalid descending range {range_part!r}")
    else:
        value = _parse_int(range_part, min_value, max_value, sunday_7=sunday_7)
        start = value
        end = value

    values = set(range(start, end + 1, step))
    if sunday_7:
        values = {0 if value == 7 else value for value in values}
    return values


def _parse_int(
    text: str,
    min_value: int,
    max_value: int,
    *,
    sunday_7: bool,
) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise CronError(f"invalid integer {text!r}") from exc
    if sunday_7 and value == 7:
        return value
    if value < min_value or value > max_value:
        raise CronError(f"value {value} outside {min_value}..{max_value}")
    return value


@dataclass(slots=True)
class ScheduledJob:
    id: str
    session_key: str
    channel: str
    chat_id: str
    prompt: str
    cron: str
    next_fire_at: float
    created_at: float
    thread_id: str | None = None
    sender_id: str = ""
    scenario: str | None = None
    recurring: bool = True
    enabled: bool = True
    fire_count: int = 0
    last_fire_at: float | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledJob:
        return cls(
            id=str(data["id"]),
            session_key=str(data["session_key"]),
            channel=str(data["channel"]),
            chat_id=str(data["chat_id"]),
            prompt=str(data["prompt"]),
            cron=str(data["cron"]),
            next_fire_at=float(data["next_fire_at"]),
            created_at=float(data["created_at"]),
            thread_id=(
                str(data["thread_id"]) if data.get("thread_id") is not None else None
            ),
            sender_id=str(data.get("sender_id") or ""),
            scenario=str(data["scenario"]) if data.get("scenario") is not None else None,
            recurring=bool(data.get("recurring", True)),
            enabled=bool(data.get("enabled", True)),
            fire_count=int(data.get("fire_count", 0)),
            last_fire_at=(
                float(data["last_fire_at"])
                if data.get("last_fire_at") is not None
                else None
            ),
            last_error=(
                str(data["last_error"]) if data.get("last_error") is not None else None
            ),
        )


class GatewayScheduleStore:
    """Small JSON-backed schedule store under the gateway state directory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._jobs: dict[str, ScheduledJob] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("schedule store: failed to load {}: {}", self._path, exc)
            return
        rows = raw.get("jobs") if isinstance(raw, dict) else None
        if not isinstance(rows, list):
            return
        now = time.time()
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                job = ScheduledJob.from_dict(row)
                schedule = _CronSchedule.parse(job.cron)
                if job.enabled and job.next_fire_at <= 0:
                    job.next_fire_at = schedule.next_after(now)
            except Exception as exc:  # noqa: BLE001
                logger.warning("schedule store: dropping invalid job: {}", exc)
                continue
            self._jobs[job.id] = job

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "jobs": [
                self._jobs[job_id].to_dict() for job_id in sorted(self._jobs)
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._path.parent,
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            tmp = fh.name
        os.replace(tmp, self._path)

    def create(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        prompt: str,
        cron: str,
        thread_id: str | None = None,
        sender_id: str = "",
        scenario: str | None = None,
        recurring: bool = True,
    ) -> ScheduledJob:
        schedule = _CronSchedule.parse(cron)
        now = time.time()
        job_id = uuid.uuid4().hex[:8]
        while job_id in self._jobs:
            job_id = uuid.uuid4().hex[:8]
        job = ScheduledJob(
            id=job_id,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            thread_id=thread_id,
            sender_id=sender_id,
            scenario=scenario,
            prompt=prompt,
            cron=cron,
            recurring=recurring,
            next_fire_at=schedule.next_after(now),
            created_at=now,
        )
        self._jobs[job.id] = job
        self._persist()
        return job

    def list(self, *, session_key: str | None = None) -> list[ScheduledJob]:
        jobs = list(self._jobs.values())
        if session_key is not None:
            jobs = [job for job in jobs if job.session_key == session_key]
        return sorted(jobs, key=lambda job: (job.next_fire_at, job.id))

    def get(self, job_id: str) -> ScheduledJob | None:
        return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        if job_id not in self._jobs:
            return False
        self._jobs.pop(job_id)
        self._persist()
        return True

    def mark_fired(
        self,
        job_id: str,
        *,
        fired_at: float,
        next_fire_at: float | None,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.fire_count += 1
        job.last_fire_at = fired_at
        job.last_error = None
        if next_fire_at is None:
            self._jobs.pop(job_id, None)
        else:
            job.next_fire_at = next_fire_at
        self._persist()

    def mark_error(self, job_id: str, error: str, *, retry_at: float) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.last_error = error[:800]
        job.next_fire_at = retry_at
        self._persist()


FireCallback = Callable[[ScheduledJob], Awaitable[None]]


class GatewayScheduler:
    """Async scheduler that wakes gateway sessions by synthetic inbound."""

    def __init__(
        self,
        *,
        store: GatewayScheduleStore,
        fire: FireCallback,
        poll_interval: float = 1.0,
    ) -> None:
        self._store = store
        self._fire = fire
        self._poll_interval = poll_interval
        self._task: asyncio.Task[None] | None = None
        self._changed = asyncio.Event()
        self._closed = False

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="gateway-scheduler")

    async def stop(self) -> None:
        self._closed = True
        self._changed.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    def create(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        prompt: str,
        cron: str,
        thread_id: str | None = None,
        sender_id: str = "",
        scenario: str | None = None,
        recurring: bool = True,
    ) -> ScheduledJob:
        job = self._store.create(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            thread_id=thread_id,
            sender_id=sender_id,
            scenario=scenario,
            prompt=prompt,
            cron=cron,
            recurring=recurring,
        )
        self._changed.set()
        return job

    def list(self, *, session_key: str | None = None) -> list[ScheduledJob]:
        return self._store.list(session_key=session_key)

    def delete(self, job_id: str, *, session_key: str | None = None) -> bool:
        job = self._store.get(job_id)
        if job is None:
            return False
        if session_key is not None and job.session_key != session_key:
            return False
        ok = self._store.delete(job_id)
        if ok:
            self._changed.set()
        return ok

    async def fire_now(
        self, job_id: str, *, session_key: str | None = None
    ) -> tuple[bool, str]:
        job = self._store.get(job_id)
        if job is None:
            return (False, "no such scheduled prompt")
        if session_key is not None and job.session_key != session_key:
            return (False, "no such scheduled prompt")
        return await self._fire_job(job)

    async def _run(self) -> None:
        while not self._closed:
            try:
                await self._fire_due()
            except Exception:  # noqa: BLE001
                logger.exception("gateway scheduler: due-job scan failed")
            timeout = self._next_sleep_seconds()
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=timeout)
            except TimeoutError:
                pass

    async def _fire_due(self) -> None:
        now = time.time()
        due = [
            job
            for job in self._store.list()
            if job.enabled and job.next_fire_at <= now
        ]
        for job in due:
            await self._fire_job(job)

    async def _fire_job(self, job: ScheduledJob) -> tuple[bool, str]:
        fired_at = time.time()
        try:
            await self._fire(job)
        except Exception as exc:  # noqa: BLE001
            logger.exception("gateway scheduler: job {} failed", job.id)
            message = str(exc) or type(exc).__name__
            self._store.mark_error(job.id, message, retry_at=fired_at + 60.0)
            return (False, message)

        if job.recurring:
            try:
                next_fire_at = _CronSchedule.parse(job.cron).next_after(fired_at)
            except CronError as exc:
                logger.warning(
                    "gateway scheduler: disabling job {} after invalid cron: {}",
                    job.id,
                    exc,
                )
                next_fire_at = None
        else:
            next_fire_at = None
        self._store.mark_fired(job.id, fired_at=fired_at, next_fire_at=next_fire_at)
        return (True, "fired")

    def _next_sleep_seconds(self) -> float:
        now = time.time()
        future = [
            job.next_fire_at - now
            for job in self._store.list()
            if job.enabled and job.next_fire_at > now
        ]
        if not future:
            return max(self._poll_interval, 30.0)
        return max(self._poll_interval, min(30.0, min(future)))


__all__ = [
    "CronError",
    "GatewayScheduleStore",
    "GatewayScheduler",
    "ScheduledJob",
]
