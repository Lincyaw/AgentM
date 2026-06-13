"""Pure OTLP/JSON parsing utilities.

Moved from ``runtime/otel_export.py`` so ``lib/`` and ``abi/`` consumers
can use them without importing the runtime layer.
"""

from __future__ import annotations

from typing import Any


def otlp_unwrap(value: Any) -> Any:
    """Unwrap an OTLP proto-JSON tagged-union value into a plain Python object.

    OTLP encodes attribute and body values as tagged unions
    (``{"stringValue": ...}``, ``{"intValue": "12"}``,
    ``{"kvlistValue": {"values": [...]}}``, ``{"arrayValue": ...}``).
    Readers (``SessionManager._load``, the catalog indexer, the tuner
    tools) need plain Python types to pattern-match against; this helper
    is the single canonical converter.

    Returns the input unchanged when it isn't a tagged union — useful
    for already-unwrapped intermediate dicts.
    """
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        raw = value["intValue"]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if "doubleValue" in value:
        raw = value["doubleValue"]
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw
    if "kvlistValue" in value:
        out: dict[str, Any] = {}
        for item in value["kvlistValue"].get("values", []) or []:
            key = item.get("key")
            if not isinstance(key, str):
                continue
            out[key] = otlp_unwrap(item.get("value"))
        return out
    if "arrayValue" in value:
        return [otlp_unwrap(v) for v in value["arrayValue"].get("values", []) or []]
    return value


def iter_spans(line: dict[str, Any]) -> list[dict[str, Any]]:
    """Iterate spans on one OTLP/JSON ndjson line.

    Handles both the collector-compatible format
    (``{"resourceSpans": [{...}]}``) and the legacy unwrapped format
    (``{"scopeSpans": [...]}``) transparently.
    """
    out: list[dict[str, Any]] = []
    resources = line.get("resourceSpans")
    if resources:
        for rs in resources:
            for scope in rs.get("scopeSpans", []) or []:
                out.extend(scope.get("spans", []) or [])
    else:
        for scope in line.get("scopeSpans", []) or []:
            out.extend(scope.get("spans", []) or [])
    return out


def iter_log_records(line: dict[str, Any]) -> list[dict[str, Any]]:
    """Iterate log records on one OTLP/JSON ndjson line.

    Handles both ``{"resourceLogs": [{...}]}`` and legacy ``{"scopeLogs": [...]}``.
    """
    out: list[dict[str, Any]] = []
    resources = line.get("resourceLogs")
    if resources:
        for rl in resources:
            for scope in rl.get("scopeLogs", []) or []:
                out.extend(scope.get("logRecords", []) or [])
    else:
        for scope in line.get("scopeLogs", []) or []:
            out.extend(scope.get("logRecords", []) or [])
    return out


__all__ = [
    "iter_log_records",
    "iter_spans",
    "otlp_unwrap",
]
