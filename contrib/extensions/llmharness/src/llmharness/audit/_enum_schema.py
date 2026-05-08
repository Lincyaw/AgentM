"""Single source-of-truth enum value lists for audit JSON-Schema payloads.

The extractor and auditor both need to embed the set of valid ``EventKind``
and ``DriftType`` values in their tool schemas. Deriving them from the enum
classes here keeps the two consumers in lockstep with ``schema.py`` and
removes the V0 hand-listed copies that can silently drift.

This module is audit-internal: it is *not* part of the public schema
contract exported to downstream consumers (rca-autorl). Keep it under
``audit/`` for that reason.
"""

from __future__ import annotations

from typing import Final

from llmharness.schema import DriftType, EventKind

EVENT_KIND_VALUES: Final[list[str]] = [k.value for k in EventKind]
"""All ``EventKind`` string values, in declaration order."""

DRIFT_TYPE_VALUES: Final[list[str | None]] = [t.value for t in DriftType] + [None]
"""All ``DriftType`` string values plus a trailing ``None`` for "no drift"."""
