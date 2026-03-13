"""Local filesystem implementation of StorageBackend."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class FilesystemBackend:
    """StorageBackend backed by the local filesystem.

    All paths are resolved relative to *root_dir*.
    """

    def __init__(self, root_dir: str | Path = ".") -> None:
        self._root = Path(root_dir).resolve()

    def _resolve(self, file_path: str) -> Path:
        resolved = (self._root / file_path).resolve()
        # Prevent path traversal above root
        if not str(resolved).startswith(str(self._root)):
            raise ValueError(
                f"Path {file_path!r} resolves outside root {self._root}"
            )
        return resolved

    def read(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> str:
        path = self._resolve(file_path)
        with open(path) as f:
            lines = f.readlines()
        selected = lines[offset : offset + limit]
        return "".join(selected)

    def write(self, file_path: str, content: str) -> None:
        path = self._resolve(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def ls(self, path: str) -> list[str]:
        resolved = self._resolve(path)
        if not resolved.is_dir():
            return []
        return sorted(entry.name for entry in resolved.iterdir())

    def glob(self, pattern: str, path: str = ".") -> list[str]:
        base = self._resolve(path)
        matches = sorted(base.glob(pattern))
        return [str(m.relative_to(self._root)) for m in matches]

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        search_root = self._resolve(path) if path else self._root
        compiled = re.compile(pattern)

        files: list[Path]
        if glob_filter:
            files = sorted(search_root.rglob(glob_filter))
        else:
            files = sorted(
                f for f in search_root.rglob("*") if f.is_file()
            )

        results: list[dict[str, Any]] = []
        for fp in files:
            try:
                for lineno, line in enumerate(
                    fp.read_text().splitlines(), 1
                ):
                    if compiled.search(line):
                        results.append(
                            {
                                "file": str(fp.relative_to(self._root)),
                                "line": lineno,
                                "content": line,
                            }
                        )
            except (UnicodeDecodeError, PermissionError):
                continue

        return results

    def exists(self, file_path: str) -> bool:
        return self._resolve(file_path).exists()

    def mkdir(self, path: str) -> None:
        self._resolve(path).mkdir(parents=True, exist_ok=True)
