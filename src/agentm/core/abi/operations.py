"""Execution-environment operation Protocols."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .cancel import CancelSignal


@dataclass(frozen=True, slots=True)
class ExecResult:
    stdout: bytes
    stderr: bytes
    exit_code: int
    timed_out: bool


@runtime_checkable
class BashOperations(Protocol):
    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        on_data: Callable[[bytes], None] | None = None,
        signal: CancelSignal | None = None,
        log_path: str | None = None,
    ) -> ExecResult:
        """Run *cmd*; when ``log_path`` is set, the backend also appends the
        combined output to that file **at the execution site** (created with
        parents, path resolved against ``cwd``) — a remote backend writes it
        inside the sandbox so log bytes never round-trip through the host.
        Best-effort: the authoritative output is still the returned
        ``ExecResult``."""
        ...


EnvironmentKind = Literal["local", "sandbox", "remote", "host"]


@dataclass(frozen=True, slots=True)
class EnvironmentRef:
    """Stable identity for the execution environment a session targets."""

    id: str
    kind: EnvironmentKind = "local"
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )


@runtime_checkable
class EnvironmentOperations(Protocol):
    """Typed backend bundle for world-effect operations."""

    @property
    def ref(self) -> EnvironmentRef:
        ...

    @property
    def bash(self) -> BashOperations:
        ...

    async def snapshot(self) -> str | None:
        ...

    async def close(self) -> None:
        ...


__all__ = [
    "BashOperations",
    "EnvironmentKind",
    "EnvironmentOperations",
    "EnvironmentRef",
    "ExecResult",
]
