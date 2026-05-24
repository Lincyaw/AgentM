"""Single source-of-truth enum value lists for audit JSON-Schema payloads.

The extractor needs to embed the set of valid ``EventKind`` and
``EdgeKind`` values in its tool schemas. Deriving them from the enum
classes here keeps the schema in lockstep with ``schema.py`` and removes
hand-listed copies that can silently drift. JSON-Schema ``enum``
literals declared in ``audit/extractor/*`` import from this module.

V4 (2026-05-24): ``EventKind.EVID`` is gone. ``EVENT_KIND_VALUES`` now
holds ``task``, ``hyp``, ``act``, ``dec``, ``concl`` (five) — derived
from :class:`~llmharness.schema.EventKind`. Linear blocks fold into a
single ``act`` whose summary records both probes and results.

V3 (issue #134, 2026-05-10) — historical:
- ``EVENT_KIND_VALUES`` historically included ``evid`` alongside
  ``task``, ``hyp``, ``act``, ``dec``, ``concl``; ``evid`` was
  dropped in V4 (see note above). Values are derived from
  :class:`~llmharness.schema.EventKind`, so this list updates
  automatically with the enum.
- ``EDGE_KIND_VALUES`` added: ``data``, ``ref`` — derived from
  :class:`~llmharness.schema.EdgeKind` (unchanged in V4).

V2 carried-forward: the auditor's ``submit_verdict`` schema (V2,
design §6.2) uses no drift-type enum; the verdict shape is free-text
(``reminder_text`` + ``continuation_notes``). ``DriftType`` /
``DRIFT_TYPE_VALUES`` stay removed.

This module is audit-internal: it is *not* part of the public schema
contract. Keep it under ``audit/`` for that reason.
"""

from __future__ import annotations

from typing import Final

from ..schema import EdgeKind, EventKind

EVENT_KIND_VALUES: Final[list[str]] = [k.value for k in EventKind]
"""All ``EventKind`` string values, in declaration order."""

EDGE_KIND_VALUES: Final[list[str]] = [k.value for k in EdgeKind]
"""All ``EdgeKind`` string values, in declaration order."""

__all__ = ["EDGE_KIND_VALUES", "EVENT_KIND_VALUES"]
