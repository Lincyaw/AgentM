"""Path expansion helpers shared by CLI and gateway peers."""

from __future__ import annotations

import os
from pathlib import Path


def expand_path(path: Path | str) -> Path:
    """Expand environment variables and a leading user home marker in ``path``."""

    return Path(os.path.expandvars(str(path))).expanduser()


def expand_path_text(path: Path | str) -> str:
    """Text form of :func:`expand_path` for APIs that accept plain strings."""

    return str(expand_path(path))


def expand_optional_path_text(path: Path | str | None) -> str | None:
    """Expand an optional path-like value, preserving ``None`` and empty input."""

    if path is None:
        return None
    raw_path = str(path)
    if not raw_path:
        return None
    return expand_path_text(raw_path)
