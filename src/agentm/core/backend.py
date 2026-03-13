"""StorageBackend protocol — abstraction for file-like I/O.

Compatible with Deep Agents' ``BackendProtocol`` concept: provides
read/write/ls/grep/glob operations that can be backed by local
filesystem, S3, or any other storage medium.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for pluggable storage backends.

    All path arguments use forward-slash separated paths.
    Implementations map these to the underlying storage medium.
    """

    def read(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> str:
        """Read file contents as text.

        ``offset`` and ``limit`` refer to line numbers (0-based).
        """
        ...

    def write(self, file_path: str, content: str) -> Any:
        """Write (create or overwrite) a file."""
        ...

    def ls(self, path: str) -> list[str]:
        """List entries in a directory.  Returns basenames."""
        ...

    def glob(self, pattern: str, path: str = ".") -> list[str]:
        """Return paths matching a glob pattern relative to *path*."""
        ...

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search file contents for *pattern* (regex).

        Returns a list of match dicts with at least ``file``, ``line``,
        and ``content`` keys.
        """
        ...

    def exists(self, file_path: str) -> bool:
        """Check whether a file or directory exists."""
        ...

    def mkdir(self, path: str) -> None:
        """Create a directory (and parents) if it does not exist."""
        ...
