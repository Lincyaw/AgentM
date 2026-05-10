"""V3 extractor: witness-based event/edge graph builder.

The extractor child runs once per main-agent turn, walks the new turn
window, and emits a graph via three tools:

- ``register_event(turn_indices, kind, summary)``
- ``add_edge(...)`` (with witness validation)
- ``submit_extraction()`` (terminator)

The state IS the output: the adapter constructs an
:class:`ExtractionState`, publishes it under
:data:`EXTRACTOR_STATE_SERVICE_KEY`, mounts the extractor extensions,
runs the child loop, and snapshots :class:`RawExtractorOutput` from
the state at the end.
"""

from __future__ import annotations

from .extensions import (
    EXTRACTOR_STATE_SERVICE_KEY,
    compose_extractor_extensions,
)
from .output import RawExtractorOutput
from .prompt import EXTRACTOR_SYSTEM_PROMPT
from .state import ExtractionState
from .tools import (
    ADD_EDGE_TOOL_NAME,
    EXTRACTOR_TOOL_NAMES,
    REGISTER_EVENT_TOOL_NAME,
    SUBMIT_EXTRACTION_REASON,
    SUBMIT_EXTRACTION_TOOL_NAME,
    build_extractor_tools,
)

__all__ = [
    "ADD_EDGE_TOOL_NAME",
    "EXTRACTOR_STATE_SERVICE_KEY",
    "EXTRACTOR_SYSTEM_PROMPT",
    "EXTRACTOR_TOOL_NAMES",
    "REGISTER_EVENT_TOOL_NAME",
    "SUBMIT_EXTRACTION_REASON",
    "SUBMIT_EXTRACTION_TOOL_NAME",
    "ExtractionState",
    "RawExtractorOutput",
    "build_extractor_tools",
    "compose_extractor_extensions",
]
