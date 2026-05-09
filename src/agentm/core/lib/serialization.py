"""Shared JSON-compatible serialization helpers for atoms."""

from __future__ import annotations

import base64
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any, *, _depth: int = 0) -> Any:
    """Convert common AgentM runtime objects into JSON-compatible values."""

    if _depth > 12:
        return "<max-depth>"
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        return {"type": "bytes", "base64": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if callable(obj):
        return f"<callable {getattr(obj, '__qualname__', repr(obj))}>"
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            field.name: _to_jsonable(getattr(obj, field.name), _depth=_depth + 1)
            for field in fields(obj)
        }
    if isinstance(obj, dict):
        return {
            str(key): _to_jsonable(value, _depth=_depth + 1)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(value, _depth=_depth + 1) for value in obj]
    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        return _to_jsonable(obj_dict, _depth=_depth + 1)
    return repr(obj)


__all__ = ["_to_jsonable"]
