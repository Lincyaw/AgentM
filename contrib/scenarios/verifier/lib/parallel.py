"""Helpers for fault-tolerant workflow parallel fan-out."""
from __future__ import annotations

from typing import Any


def normalize_parallel(raw: object) -> list[Any]:
    """Normalize workflow parallel output after per-item agent failures."""
    return raw if isinstance(raw, list) else []
