"""Shared entry-type constants for cognitive-audit session entries.

Both the runtime adapter (``llmharness.adapters.agentm``) and the offline
inspector / dataset exporter (``llmharness.cli``) read and write these
``SessionEntry.type`` strings. Centralising them prevents silent drift
between the writer and the readers.
"""

from __future__ import annotations

# Successful-path entries — make up the durable audit graph.
AUDIT_EVENT = "llmharness.audit_event"
VERDICT = "llmharness.verdict"
EXTRACTOR_CURSOR = "llmharness.extractor_cursor"
REMINDER_DELIVERED = "llmharness.reminder_delivered"

# Typed-failure entries — surface specific failure modes for offline review.
EXTRACTOR_NO_CALL = "llmharness.extractor_no_call"
EXTRACTOR_ERROR = "llmharness.extractor_error"
EXTRACTOR_EMPTY = "llmharness.extractor_empty"
AUDIT_NO_CALL = "llmharness.audit_no_call"
AUDIT_ERROR = "llmharness.audit_error"

# Echoed message-record type from the AgentM session JSONL — used by the
# dataset exporter to walk the trajectory in record order.
MESSAGE = "message"

# Audit cadence / window sizes. The adapter emits payloads using these
# slices at firing time; the dataset exporter MUST reproduce the same
# cuts when reconstructing each case, otherwise the recorded inputs and
# the runtime inputs diverge.
RECENT_GRAPH_SLICE_FOR_EXTRACTOR = 20
RECENT_VERDICTS_FOR_AUDITOR = 5
