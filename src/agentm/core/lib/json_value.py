# code-health: ignore-file[AM025] -- JSON boundary validates runtime value shapes
"""Strict JSON value encoding shared by codecs and storage adapters."""

# code-health: ignore-file[AM022] -- validates heterogeneous JSON boundary values

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a JSON-safe value without leaking mutable implementation objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def json_restore(value: Any) -> Any:
    """Restore values encoded by :func:`json_safe` and reject invalid JSON."""

    if isinstance(value, dict):
        if set(value) == {"__bytes_hex__"} and isinstance(value["__bytes_hex__"], str):
            try:
                return bytes.fromhex(value["__bytes_hex__"])
            except ValueError as exc:
                raise ValueError("invalid encoded bytes value") from exc
        if not all(isinstance(key, str) for key in value):
            raise ValueError("encoded JSON object keys must be strings")
        return {key: json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_restore(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("encoded JSON numbers must be finite")
        return value
    raise ValueError(f"encoded value is not JSON-safe: {type(value).__name__}")


__all__ = ["json_restore", "json_safe"]
