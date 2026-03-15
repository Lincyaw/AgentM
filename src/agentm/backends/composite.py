"""Composite backend — routes operations by path prefix.

Modeled after Deep Agents' ``CompositeBackend``: register multiple
backends under distinct prefixes and dispatch based on the path.
"""

from __future__ import annotations

from typing import Any

from agentm.core.backend import StorageBackend


class CompositeBackend:
    """Routes storage operations to sub-backends based on path prefix.

    Prefixes are matched longest-first.  A ``default`` backend handles
    paths that don't match any prefix.

    Example::

        composite = CompositeBackend(default=FilesystemBackend("/data"))
        composite.mount("/knowledge", knowledge_backend)
        composite.mount("/trajectory", s3_backend)

        # Routes to knowledge_backend:
        composite.read("/knowledge/entries/foo.json")

        # Routes to default:
        composite.read("/other/path.txt")
    """

    def __init__(self, default: StorageBackend) -> None:
        self._default = default
        self._mounts: list[tuple[str, StorageBackend]] = []

    def mount(self, prefix: str, backend: StorageBackend) -> None:
        """Mount a backend at a path prefix.

        Prefix should start with ``/`` and not end with ``/``.
        """
        normalized = prefix.rstrip("/")
        self._mounts.append((normalized, backend))
        # Sort longest-first for greedy matching
        self._mounts.sort(key=lambda x: -len(x[0]))

    def _route(self, file_path: str) -> tuple[StorageBackend, str]:
        """Find the backend for a path and strip the prefix."""
        for prefix, backend in self._mounts:
            if file_path == prefix or file_path.startswith(prefix + "/"):
                relative = file_path[len(prefix) :].lstrip("/")
                return backend, relative or "."
        return self._default, file_path

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        backend, rel_path = self._route(file_path)
        return backend.read(rel_path, offset=offset, limit=limit)

    def write(self, file_path: str, content: str) -> Any:
        backend, rel_path = self._route(file_path)
        return backend.write(rel_path, content)

    def ls(self, path: str) -> list[str]:
        backend, rel_path = self._route(path)
        return backend.ls(rel_path)

    def glob(self, pattern: str, path: str = ".") -> list[str]:
        backend, rel_path = self._route(path)
        return backend.glob(pattern, rel_path)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        backend, rel_path = self._route(path or "/")
        return backend.grep(pattern, path=rel_path, glob_filter=glob_filter)

    def exists(self, file_path: str) -> bool:
        backend, rel_path = self._route(file_path)
        return backend.exists(rel_path)

    def mkdir(self, path: str) -> None:
        backend, rel_path = self._route(path)
        backend.mkdir(rel_path)
