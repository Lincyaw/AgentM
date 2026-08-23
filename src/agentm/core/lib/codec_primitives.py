# code-health: ignore-file[AM025] -- shape checks at the JSON trust boundary are this module
"""Strict JSON shape checks shared by codecs, wire formats, and storage adapters.

Every check raises ``ValueError`` labelled with a caller-supplied path so that a
malformed payload names the field that broke it.

Two call shapes sit over one implementation:

* ``expect_*`` takes a value the caller already pulled out of a payload and a
  path to label it with.  This is the core; prefer it.
* ``field_*`` reads ``key`` out of a mapping and labels errors ``<path>.<key>``.
  Each one is a one-line wrapper over its ``expect_*`` counterpart, kept for
  record decoders that would otherwise repeat the key in every call.
"""

# code-health: ignore-file[AM022] -- validates heterogeneous JSON boundary values

from __future__ import annotations

import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any


def expect_object(value: object, path: str) -> dict[str, Any]:
    """Return ``value`` as a string-keyed JSON object."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be an object")
    return value


def expect_array(value: object, path: str) -> list[Any]:
    """Return ``value`` as a JSON array."""

    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def expect_string(value: object, path: str, *, allow_empty: bool = True) -> str:
    """Return ``value`` as a string."""

    if not isinstance(value, str) or (not allow_empty and not value):
        suffix = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{path} must be {suffix}")
    return value


def expect_optional_string(
    value: object,
    path: str,
    *,
    allow_empty: bool = True,
) -> str | None:
    """Return ``value`` as a string, passing ``None`` through."""

    if value is None:
        return None
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{path} must be a string or null")
    return value


def expect_boolean(value: object, path: str) -> bool:
    """Return ``value`` as a bool."""

    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a bool")
    return value


def expect_integer(value: object, path: str, *, minimum: int | None = None) -> int:
    """Return ``value`` as an int, rejecting bools."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    return value


def expect_optional_integer(value: object, path: str) -> int | None:
    """Return ``value`` as an int, passing ``None`` through."""

    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer or null")
    return value


def expect_number(value: object, path: str) -> float:
    """Return ``value`` as a finite float."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def expect_literal(value: object, path: str, allowed: AbstractSet[str]) -> str:
    """Return ``value`` as one of ``allowed``."""

    text = expect_string(value, path)
    if text not in allowed:
        raise ValueError(f"{path} has invalid value {text!r}")
    return text


def expect_string_tuple(value: object, path: str) -> tuple[str, ...]:
    """Return ``value`` as a tuple of strings."""

    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path} must be a list of strings")
    return tuple(value)


def expect_only_fields(
    data: Mapping[str, object],
    allowed: AbstractSet[str],
    path: str,
) -> None:
    """Reject any key of ``data`` outside ``allowed``."""

    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")


def field_string(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    allow_empty: bool = False,
) -> str:
    """Read ``key`` from ``data`` as a string."""

    return expect_string(data.get(key), f"{path}.{key}", allow_empty=allow_empty)


def field_integer(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    minimum: int | None = None,
) -> int:
    """Read ``key`` from ``data`` as an int."""

    return expect_integer(data.get(key), f"{path}.{key}", minimum=minimum)


def field_number(data: Mapping[str, Any], key: str, *, path: str) -> float:
    """Read ``key`` from ``data`` as a finite float."""

    return expect_number(data.get(key), f"{path}.{key}")


def field_boolean(data: Mapping[str, Any], key: str, *, path: str) -> bool:
    """Read ``key`` from ``data`` as a bool."""

    return expect_boolean(data.get(key), f"{path}.{key}")


def field_literal(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    allowed: AbstractSet[str],
) -> str:
    """Read ``key`` from ``data`` as one of ``allowed``."""

    return expect_literal(data.get(key), f"{path}.{key}", allowed)


__all__ = [
    "expect_array",
    "expect_boolean",
    "expect_integer",
    "expect_literal",
    "expect_number",
    "expect_object",
    "expect_only_fields",
    "expect_optional_integer",
    "expect_optional_string",
    "expect_string",
    "expect_string_tuple",
    "field_boolean",
    "field_integer",
    "field_literal",
    "field_number",
    "field_string",
]
