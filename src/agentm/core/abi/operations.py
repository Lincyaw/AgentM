"""File and shell operation Protocols (stable ABI).

See `.claude/designs/extension-as-scenario.md` section 10b.6 for the
`FileOperations` / `BashOperations` boundary that keeps core transport-agnostic.

Operations are a constitution-level port: tool atoms consume the bundle exposed
by ``ExtensionAPI.get_operations()``, but this ABI intentionally does not define
a runtime registration method for atom-driven replacement.
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


@dataclass(frozen=True, slots=True)
class FileStat:
    """Portable file-stat result returned by :meth:`FileOperations.stat`."""

    size: int
    mtime_ns: int
    is_file: bool
    is_dir: bool


class FileOperations(Protocol):
    async def read_file(self, path: str) -> bytes: ...

    async def access(self, path: str) -> bool: ...

    async def is_dir(self, path: str) -> bool: ...

    async def is_file(self, path: str) -> bool: ...

    async def list_dir(self, path: str) -> list[str]: ...

    async def stat(self, path: str) -> FileStat: ...

    async def write_file(self, path: str, data: bytes) -> None: ...

    async def makedirs(self, path: str, exist_ok: bool = True) -> None: ...


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
    ) -> ExecResult: ...


@dataclass(frozen=True, slots=True)
class Operations:
    """Bundle of operation backends an extension may need.

    Returned from ``ExtensionAPI.get_operations()``. The default bundle wraps
    the local stdlib-backed implementations; the runtime may inject a different
    bundle when constructing a session. Atoms can consume the active bundle but
    cannot replace it via ``ExtensionAPI`` in v0.
    """

    file: FileOperations
    bash: BashOperations
