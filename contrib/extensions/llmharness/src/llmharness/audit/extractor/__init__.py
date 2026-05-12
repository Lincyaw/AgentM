"""V3.1 extractor: witness-based event/edge graph builder, single-tool flow.

The extractor child runs once per main-agent turn (or every k turns,
adapter-configurable), walks the new turn window, and emits a graph in
ONE tool call:

- ``submit_events(events=[...])`` — the entire graph for this firing.
  Events embed ``refs[]`` linking earlier events with witness fields;
  validation runs in :meth:`ExtractionState.commit`.

The state IS the output: the adapter constructs an
:class:`ExtractionState`, hands it to the extension via the install
config, runs the child loop, and snapshots :class:`RawExtractorOutput`
from the state at the end.
"""

from __future__ import annotations

from .extensions import (
    EXTRACTOR_STATE_SERVICE_KEY,
    compose_extractor_extensions,
)
from .output import RawExtractorOutput
from .prompt import load_extractor_prompt
from .state import ExtractionState
from .tools import (
    EXTRACTOR_TOOL_NAMES,
    SUBMIT_EVENTS_REASON,
    SUBMIT_EVENTS_TOOL_NAME,
    build_extractor_tools,
)

__all__ = [
    "EXTRACTOR_STATE_SERVICE_KEY",
    "EXTRACTOR_TOOL_NAMES",
    "SUBMIT_EVENTS_REASON",
    "SUBMIT_EVENTS_TOOL_NAME",
    "ExtractionState",
    "RawExtractorOutput",
    "build_extractor_tools",
    "compose_extractor_extensions",
    "load_extractor_prompt",
]
