"""Pure, backend-neutral OTLP proto-JSON decoding helpers."""

from __future__ import annotations

from collections.abc import Mapping


def otlp_unwrap(value: object) -> object:
    """Unwrap one OTLP proto-JSON tagged-union value."""
    if not isinstance(value, Mapping):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        raw = value["intValue"]
        if not isinstance(raw, (str, bytes, int, float)):
            return raw
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if "doubleValue" in value:
        raw = value["doubleValue"]
        if not isinstance(raw, (str, bytes, int, float)):
            return raw
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw
    if "kvlistValue" in value:
        container = _mapping(value["kvlistValue"], "OTLP kvlistValue")
        result: dict[str, object] = {}
        for item in _mapping_list(
            container.get("values"),
            "OTLP kvlistValue.values",
        ):
            key = item.get("key")
            if isinstance(key, str):
                result[key] = otlp_unwrap(item.get("value"))
        return result
    if "arrayValue" in value:
        container = _mapping(value["arrayValue"], "OTLP arrayValue")
        values = container.get("values")
        if values is None:
            return []
        if not isinstance(values, list):
            raise ValueError("OTLP arrayValue.values must be a list")
        return [otlp_unwrap(item) for item in values]
    return dict(value)


def iter_spans(line: Mapping[str, object]) -> list[dict[str, object]]:
    """Return spans from one resource-wrapped or scope-only OTLP JSON line."""
    resources = _mapping_list(
        line.get("resourceSpans"),
        "OTLP resourceSpans",
    )
    scopes: list[dict[str, object]] = []
    if resources:
        for resource in resources:
            scopes.extend(
                _mapping_list(
                    resource.get("scopeSpans"),
                    "OTLP scopeSpans",
                )
            )
    else:
        scopes = _mapping_list(line.get("scopeSpans"), "OTLP scopeSpans")
    spans: list[dict[str, object]] = []
    for scope in scopes:
        spans.extend(_mapping_list(scope.get("spans"), "OTLP spans"))
    return spans


def iter_log_records(line: Mapping[str, object]) -> list[dict[str, object]]:
    """Return records from one resource-wrapped or scope-only OTLP JSON line."""
    resources = _mapping_list(
        line.get("resourceLogs"),
        "OTLP resourceLogs",
    )
    scopes: list[dict[str, object]] = []
    if resources:
        for resource in resources:
            scopes.extend(
                _mapping_list(
                    resource.get("scopeLogs"),
                    "OTLP scopeLogs",
                )
            )
    else:
        scopes = _mapping_list(line.get("scopeLogs"), "OTLP scopeLogs")
    records: list[dict[str, object]] = []
    for scope in scopes:
        records.extend(_mapping_list(scope.get("logRecords"), "OTLP logRecords"))
    return records


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _mapping_list(value: object, label: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return [dict(_mapping(item, f"{label} item")) for item in value]


__all__ = [
    "iter_log_records",
    "iter_spans",
    "otlp_unwrap",
]
