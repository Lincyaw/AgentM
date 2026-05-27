"""Single source of truth for audit atom module names + service keys.

Centralizing these strings here keeps ``seams.session`` and
``seams.compose`` from drifting against the extractor atom, and avoids the
helper module reaching into ``audit.extractor.extensions`` to import a
SERVICE_KEY constant (cross-atom import).

Why this exists in ``audit/`` rather than ``core``:
- The constants are llmharness-internal coordinates, not part of any
  public API. They describe how the audit subsystem composes itself out
  of SDK-builtin atoms (observability/operations_local/system_prompt)
  plus the extractor atom's service key.
"""

from __future__ import annotations

from typing import Final

# llmharness extractor-state service key — set per-firing by
# ``seams.session.bind_extractor_state`` (as a ``config['state']``
# fallback) and read by the extractor atom's tool implementations.
EXTRACTOR_STATE_SERVICE_KEY: Final[str] = "llmharness.extractor_state"

# llmharness atom dotted module names referenced by the composers.
EXTRACTOR_TOOLS_MODULE: Final[str] = "llmharness.audit.extractor.atom"

# SDK builtin atom dotted module names that the audit composers wire in.
# A host-side rename of any of these surfaces here first.
SYSTEM_PROMPT_MODULE: Final[str] = "agentm.extensions.builtin.system_prompt"
OBSERVABILITY_MODULE: Final[str] = "agentm.extensions.builtin.observability"
OPERATIONS_MODULE: Final[str] = "agentm.extensions.builtin.operations_local"

# Optional atoms mounted on the extractor child only when
# ``extractor_tool_call_budget`` is set. ``loop_budget`` is an SDK
# builtin; ``turn_reminder`` is a sibling contrib atom — a soft
# dependency, so the budget path fails at child freeze (opaque
# ExtensionLoadError) if it is renamed or removed.
LOOP_BUDGET_MODULE: Final[str] = "agentm.extensions.builtin.loop_budget"
TURN_REMINDER_MODULE: Final[str] = "contrib.extensions.turn_reminder"

__all__ = [
    "EXTRACTOR_STATE_SERVICE_KEY",
    "EXTRACTOR_TOOLS_MODULE",
    "LOOP_BUDGET_MODULE",
    "OBSERVABILITY_MODULE",
    "OPERATIONS_MODULE",
    "SYSTEM_PROMPT_MODULE",
    "TURN_REMINDER_MODULE",
]
