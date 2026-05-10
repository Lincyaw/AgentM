"""Single source-of-truth enum value list for audit JSON-Schema payloads.

The extractor needs to embed the set of valid ``EventKind`` values in its
tool schema. Deriving them from the enum class here keeps the schema in
lockstep with ``schema.py`` and removes hand-listed copies that can
silently drift.

``DriftType`` and ``DRIFT_TYPE_VALUES`` are removed in V2 (issue #134,
2026-05-10). The auditor's ``submit_verdict`` schema (V2, design §6.2)
uses no drift-type enum; the verdict shape is free-text
(``reminder_text`` + ``continuation_notes``).

This module is audit-internal: it is *not* part of the public schema
contract. Keep it under ``audit/`` for that reason.
"""

from __future__ import annotations

from typing import Final

from ..schema import EventKind

EVENT_KIND_VALUES: Final[list[str]] = [k.value for k in EventKind]
"""All ``EventKind`` string values, in declaration order."""

__all__ = ["EVENT_KIND_VALUES"]
