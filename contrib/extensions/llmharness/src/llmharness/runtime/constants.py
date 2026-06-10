"""Audit atom module names + service keys.

Why this exists in ``audit/`` rather than ``core``:
- The constants are llmharness-internal coordinates, not part of any
  public API.
"""

from __future__ import annotations

from typing import Final

EXTRACTOR_STATE_SERVICE_KEY: Final[str] = "llmharness.extractor_state"

EXTRACTOR_TOOLS_MODULE: Final[str] = "llmharness.agents.extractor.extractor_tools"

SYSTEM_PROMPT_MODULE: Final[str] = "agentm.extensions.builtin.system_prompt"
OBSERVABILITY_MODULE: Final[str] = "agentm.extensions.builtin.observability"
OPERATIONS_MODULE: Final[str] = "agentm.extensions.builtin.operations"

__all__ = [
    "EXTRACTOR_STATE_SERVICE_KEY",
    "EXTRACTOR_TOOLS_MODULE",
    "OBSERVABILITY_MODULE",
    "OPERATIONS_MODULE",
    "SYSTEM_PROMPT_MODULE",
]
