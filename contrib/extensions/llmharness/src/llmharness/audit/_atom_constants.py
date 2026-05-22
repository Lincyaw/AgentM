"""Single source of truth for audit atom module names + service keys.

Centralizing these strings here keeps ``_session_helpers`` and
``_compose`` from drifting against the extractor atom, and avoids the
helper module reaching into ``audit.extractor.extensions`` to import a
SERVICE_KEY constant (cross-atom import).

Why this exists in ``audit/`` rather than ``core``:
- The constants are llmharness-internal coordinates, not part of any
  public API. They describe how the audit subsystem composes itself out
  of SDK-builtin atoms (observability/otel_tracing/operations_local/
  system_prompt) plus the extractor atom's service key.
"""

from __future__ import annotations

from typing import Final

# llmharness extractor atom — service key registered by
# ``audit.extractor.extensions`` and consumed by both ``_session_helpers``
# (host→child propagation) and the extractor tool implementations.
EXTRACTOR_STATE_SERVICE_KEY: Final[str] = "llmharness.extractor_state"

# llmharness atom dotted module names referenced by the composers.
EXTRACTOR_TOOLS_MODULE: Final[str] = "llmharness.audit.extractor.atom"

# SDK builtin atom dotted module names that the audit composers wire in.
# A host-side rename of any of these surfaces here first.
SYSTEM_PROMPT_MODULE: Final[str] = "agentm.extensions.builtin.system_prompt"
OBSERVABILITY_MODULE: Final[str] = "agentm.extensions.builtin.observability"
OTEL_TRACING_MODULE: Final[str] = "agentm.extensions.builtin.otel_tracing"
OPERATIONS_MODULE: Final[str] = "agentm.extensions.builtin.operations_local"

__all__ = [
    "EXTRACTOR_STATE_SERVICE_KEY",
    "EXTRACTOR_TOOLS_MODULE",
    "OBSERVABILITY_MODULE",
    "OPERATIONS_MODULE",
    "OTEL_TRACING_MODULE",
    "SYSTEM_PROMPT_MODULE",
]
