"""File and shell operation Protocols (stable ABI).

See `.claude/designs/extension-as-scenario.md` section 10b.6 for the
`FileOperations` / `BashOperations` boundary that keeps core transport-agnostic.
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


class FileOperations(Protocol):
    async def read_file(self, path: str) -> bytes: ...

    async def write_file(self, path: str, content: bytes) -> None: ...

    async def access(self, path: str) -> bool: ...

    async def list_dir(self, path: str) -> list[str]: ...


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
    the local stdlib-backed implementations; sessions may later swap these
    for sandboxed / remoted backends without atoms changing shape.
    """

    file: FileOperations
    bash: BashOperations
