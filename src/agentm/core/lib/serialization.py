"""Explicit, non-reflective JSON encoding for diagnostic payloads."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any


_MAX_DEPTH = 12


def to_jsonable(obj: Any) -> Any:
    """Encode a value without invoking arbitrary object stringification.

    Unsupported, cyclic, and depth-truncated values remain visible as typed
    diagnostic markers. The encoder never inspects ``__dict__`` and never
    calls user-defined ``repr``/``str`` methods.
    """

    return _encode(obj, depth=0, seen=frozenset())


def _encode(obj: Any, *, depth: int, seen: frozenset[int]) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return {
            "type": "non_finite_float",
            "value": "nan" if math.isnan(obj) else "infinity" if obj > 0 else "-infinity",
        }
    if depth >= _MAX_DEPTH:
        return {
            "type": "truncated",
            "reason": "max_depth",
            "python_type": _python_type_name(obj),
        }
    if isinstance(obj, bytes):
        return {"type": "bytes", "base64": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, Enum):
        return _encode(obj.value, depth=depth + 1, seen=seen)
    if isinstance(obj, Path):
        return str(obj)
    if callable(obj):
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        return {
            "type": "callable",
            "module": module if isinstance(module, str) else type(obj).__module__,
            "qualname": (
                qualname if isinstance(qualname, str) else type(obj).__qualname__
            ),
        }
    track_identity = (
        is_dataclass(obj)
        and not isinstance(obj, type)
        or isinstance(obj, (Mapping, list, tuple, set, frozenset))
    )
    if track_identity and id(obj) in seen:
        return {
            "type": "truncated",
            "reason": "cycle",
            "python_type": _python_type_name(obj),
        }
    child_seen = seen | {id(obj)} if track_identity else seen
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            field.name: _encode(
                getattr(obj, field.name),
                depth=depth + 1,
                seen=child_seen,
            )
            for field in fields(obj)
        }
    if isinstance(obj, Mapping):
        if all(isinstance(key, str) for key in obj):
            return {
                key: _encode(value, depth=depth + 1, seen=child_seen)
                for key, value in obj.items()
            }
        entries = [
            {
                "key": _encode(key, depth=depth + 1, seen=child_seen),
                "value": _encode(value, depth=depth + 1, seen=child_seen),
            }
            for key, value in obj.items()
        ]
        entries.sort(key=_canonical_json)
        return {"type": "mapping", "entries": entries}
    if isinstance(obj, (list, tuple)):
        return [
            _encode(value, depth=depth + 1, seen=child_seen) for value in obj
        ]
    if isinstance(obj, (set, frozenset)):
        values = [
            _encode(value, depth=depth + 1, seen=child_seen) for value in obj
        ]
        values.sort(key=_canonical_json)
        return {"type": "set", "items": values}
    return {
        "type": "unsupported",
        "python_type": _python_type_name(obj),
    }


def _python_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


__all__ = ["to_jsonable"]
