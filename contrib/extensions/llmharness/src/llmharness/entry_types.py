"""Shared entry-type constants for cognitive-audit session entries.

The runtime adapter (``llmharness.atom``) writes these
``SessionEntry.type`` strings into the AgentM session log; the offline
hydration path and dataset exporter read them back. Centralising the
literals prevents silent drift between the writer and the readers.

Entry types:
- ``AUDIT_GRAPH_OP`` / ``VERDICT`` / ``EXTRACTOR_CURSOR`` /
  ``REMINDER_DELIVERED`` — successful-path entries that form the
  durable audit graph.
- ``EXTRACTOR_NO_CALL`` / ``EXTRACTOR_ERROR`` / ``EXTRACTOR_EMPTY``
  / ``EXTRACTOR_PARTIAL`` — Phase 1 typed-failure / partial-success
  entries.
- ``AUDIT_NO_CALL`` / ``AUDIT_ERROR`` — Phase 2 typed-failure entries.
- ``MESSAGE`` — echoed message-record type from the AgentM session JSONL.
"""

from __future__ import annotations

# One graph op produced by an extractor firing. Payload is the result of
# ``llmharness.graph.ops.GraphOp.to_dict()`` — i.e. an ``"op"``
# discriminator (``node_upsert`` / ``node_delete`` / ``edge_upsert`` /
# ``edge_delete``) plus the op-specific fields, augmented with firing
# metadata: ``firing_id`` (int), ``op_index`` (int — the op's position
# inside its firing), and ``caused_by_turn_window`` (``[lo, hi]``
# inclusive trajectory range of the new-turn window the firing
# consumed).
AUDIT_GRAPH_OP = "llmharness.audit_graph_op"
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
