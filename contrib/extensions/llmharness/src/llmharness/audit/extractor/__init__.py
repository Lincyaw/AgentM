"""V19 extractor: witness-based event/edge graph builder, incremental tool flow.

The extractor child runs once per main-agent turn (or every k turns,
adapter-configurable), walks the new turn window, and emits a graph via
incremental edits:

* ``submit_plan`` — captures the block plan once per firing.
* ``upsert_node`` / ``delete_node`` — node-level edits (this firing or
  any prior firing — the id resolves against the folded view).
* ``upsert_edge`` / ``delete_edge`` — witness-bearing edge edits.
* ``reset_extraction`` — drop pending state and retry.
* ``finalize_extraction`` — terminator; runs the cross-graph degree
  check and either ToolTerminates or returns a three-section
  passthrough-recovery error so the LLM can promote passthrough nodes
  to true branch points with further edits.

The state IS the output: the adapter constructs an
:class:`ExtractionState`, hands it to the merged atom via the install
config, runs the child loop, and snapshots
:class:`RawExtractorOutput` from the state at the end.

Public contract (re-exported from :mod:`llmharness`):

* :data:`EXTRACTOR_TOOL_NAMES` — every name :mod:`atom` registers.
* :data:`EXTRACTOR_TERMINATION_REASON` — the ``ToolTerminate.reason``
  the child loop emits when the model calls ``finalize_extraction``.
"""

from __future__ import annotations

from .atom import (
    EXTRACTOR_STATE_SERVICE_KEY,
    EXTRACTOR_TERMINATION_REASON,
    EXTRACTOR_TOOL_NAMES,
)
from .extensions import compose_extractor_extensions
from .finalize_extraction import (
    FINALIZE_EXTRACTION_REASON,
    FINALIZE_EXTRACTION_TOOL_NAME,
)
from .output import RawExtractorOutput
from .prompt import load_extractor_prompt
from .state import ExtractionState

__all__ = [
    "EXTRACTOR_STATE_SERVICE_KEY",
    "EXTRACTOR_TERMINATION_REASON",
    "EXTRACTOR_TOOL_NAMES",
    "FINALIZE_EXTRACTION_REASON",
    "FINALIZE_EXTRACTION_TOOL_NAME",
    "ExtractionState",
    "RawExtractorOutput",
    "compose_extractor_extensions",
    "load_extractor_prompt",
]
