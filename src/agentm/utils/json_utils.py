"""Shared JSON and text utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def safe_load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None on parse or I/O errors."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def truncate(text: str, max_len: int, *, oneline: bool = False) -> str:
    """Truncate text to max_len characters, appending '...' if truncated."""
    if oneline:
        text = text.replace("\n", " ").replace("\r", "")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
