# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Execution-environment operation Protocols."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from .cancel import CancelSignal


@dataclass(frozen=True, slots=True)
class ExecResult:
    stdout: bytes
    stderr: bytes
    exit_code: int
    timed_out: bool

    def __post_init__(self) -> None:
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise TypeError("execution stdout and stderr must be bytes")
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise TypeError("execution exit_code must be an integer")
        if not isinstance(self.timed_out, bool):
            raise TypeError("execution timed_out must be a bool")


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
        ``ExecResult``.

        If ``signal`` fires before process completion, the backend must
        terminate and reap the execution before raising ``CancelledError``.
        Cancellation of the calling coroutine has the same terminate-and-reap
        requirement."""
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

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise TypeError("environment id must be a non-empty string")
        if self.kind not in {"local", "sandbox", "remote", "host"}:
            raise ValueError(f"invalid environment kind: {self.kind!r}")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("environment metadata must be a mapping")
        copied: dict[str, str | int | float | bool | None] = {}
        for key, value in self.metadata.items():
            if not isinstance(key, str):
                raise TypeError("environment metadata keys must be strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"environment metadata {key!r} must be a JSON scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"environment metadata {key!r} must be finite")
            copied[key] = value
        object.__setattr__(self, "metadata", MappingProxyType(copied))


@runtime_checkable
class EnvironmentOperations(Protocol):
    """Typed backend bundle for world-effect operations."""

    @property
    def ref(self) -> EnvironmentRef: ...

    @property
    def bash(self) -> BashOperations: ...

    async def snapshot(self) -> str | None: ...

    async def close(self) -> None: ...


__all__ = [
    "BashOperations",
    "EnvironmentKind",
    "EnvironmentOperations",
    "EnvironmentRef",
    "ExecResult",
]
