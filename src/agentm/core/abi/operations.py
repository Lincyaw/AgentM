"""Shell operation Protocol (stable ABI).

Operations are a constitution-level port: the ``tool_bash`` atom consumes
``BashOperations`` exposed by the session services.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExecResult:
    stdout: bytes
    stderr: bytes
    exit_code: int
    timed_out: bool


class BashOperations(Protocol):
    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        on_data: Callable[[bytes], None] | None = None,
        signal: asyncio.Event | None = None,
        log_path: str | None = None,
    ) -> ExecResult:
        """Run *cmd*; when ``log_path`` is set, the backend also appends the
        combined output to that file **at the execution site** (created with
        parents, path resolved against ``cwd``) — a remote backend writes it
        inside the sandbox so log bytes never round-trip through the host.
        Best-effort: the authoritative output is still the returned
        ``ExecResult``."""
        ...


@dataclass(frozen=True, slots=True)
class Operations:
    """Bundle returned by the session services."""

    bash: BashOperations
