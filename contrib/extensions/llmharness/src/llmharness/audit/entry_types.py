"""Shared entry-type constants for cognitive-audit session entries.

Both the runtime adapter (``llmharness.adapters.agentm``) and the offline
inspector / dataset exporter (``llmharness.cli``) read and write these
``SessionEntry.type`` strings. Centralising them prevents silent drift
between the writer and the readers.

Entry types:
- ``AUDIT_EVENT`` / ``AUDIT_EDGE`` / ``VERDICT`` / ``EXTRACTOR_CURSOR`` /
  ``REMINDER_DELIVERED`` — successful-path entries that form the
  durable audit graph.
- ``EXTRACTOR_NO_CALL`` / ``EXTRACTOR_ERROR`` / ``EXTRACTOR_EMPTY``
  / ``EXTRACTOR_PARTIAL`` — Phase 1 typed-failure / partial-success
  entries.
- ``AUDIT_NO_CALL`` / ``AUDIT_ERROR`` — Phase 2 typed-failure entries.
- ``MESSAGE`` — echoed message-record type from the AgentM session JSONL.

V3 (issue #134, 2026-05-10):
- ``EXTRACTOR_INVALID`` is removed; the graph validator goes away in
  commit 2 in favour of the witness-pipeline at edge-construction time.
- ``EXTRACTOR_PARTIAL`` records a firing where the extractor committed
  some events but had to drop one or more edges after exhausting the
  witness retry budget. Payload:
  ``{"dropped_edges": list[dict], "turn_window": [int, int]}``.
- ``AUDIT_EDGE`` records one accepted edge. Payload is
  :meth:`llmharness.schema.Edge.to_dict`.
"""

from __future__ import annotations

# Successful-path entries — make up the durable audit graph.
AUDIT_EVENT = "llmharness.audit_event"
# One accepted Edge record. Payload: ``Edge.to_dict()``. Persisted
# alongside ``AUDIT_EVENT`` entries so graph traversal can replay both.
AUDIT_EDGE = "llmharness.audit_edge"
# One Phase record produced by the mechanical merger
# (``audit.phase.merge_to_phases``). Persisted after the raw events of
# a successful firing so the auditor can read a coalesced "basic block"
# view of the trajectory and drill back to raw events via
# ``get_event_detail``. Payload: ``Phase.to_dict()``.
AUDIT_PHASE = "llmharness.audit_phase"
VERDICT = "llmharness.verdict"
EXTRACTOR_CURSOR = "llmharness.extractor_cursor"
REMINDER_DELIVERED = "llmharness.reminder_delivered"

# Typed-failure / partial-success entries — surface specific failure modes
# for offline review.
EXTRACTOR_NO_CALL = "llmharness.extractor_no_call"
EXTRACTOR_ERROR = "llmharness.extractor_error"
EXTRACTOR_EMPTY = "llmharness.extractor_empty"
# Phase 1 committed some events but dropped one or more edges after
# exhausting the witness retry budget (design §4.c, §6).
# Payload: ``{"dropped_edges": list[dict], "turn_window": [int, int]}``.
EXTRACTOR_PARTIAL = "llmharness.extractor_partial"
AUDIT_NO_CALL = "llmharness.audit_no_call"
AUDIT_ERROR = "llmharness.audit_error"

# Echoed message-record type from the AgentM session JSONL — used by the
# dataset exporter to walk the trajectory in record order.
MESSAGE = "message"

# Audit cadence / window sizes. The adapter emits payloads using these
# slices at firing time; the dataset exporter MUST reproduce the same
# cuts when reconstructing each case, otherwise the recorded inputs and
# the runtime inputs diverge. recent_graph is no longer truncated — the
# extractor sees the full prior graph each firing — so only the auditor
# tail constant lives here.
RECENT_VERDICTS_FOR_AUDITOR = 5
