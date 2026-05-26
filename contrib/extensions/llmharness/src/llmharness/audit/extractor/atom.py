"""§11 single-file extension: register the extractor tool surface.

Merges the v18 ``tools.py`` builder into one atom that mirrors the
auditor's structure (see :mod:`llmharness.audit.auditor.atom`). The
state handoff stays exactly as it was: the adapter constructs a
per-firing :class:`ExtractionState` and passes it via ``config['state']``
at child-session spawn time, or pre-publishes it under the
``llmharness.extractor_state`` service key for tests on the same
session.

Tool set, all stateful (each closes over the same ``ExtractionState``):

* ``upsert_node`` / ``delete_node`` — direct node edits against the
  folded view (this firing's pending ops + prior firings).
* ``upsert_edge`` / ``delete_edge`` — witness-bearing edge edits.
* ``reset_extraction`` — drop pending state and retry.
* ``finalize_extraction`` — terminator; runs the cross-graph degree
  check and either ToolTerminates or returns a three-section
  passthrough-recovery error.

Public contract (also re-exported from :mod:`llmharness.audit.extractor`
and the top-level :mod:`llmharness` package):

* :data:`EXTRACTOR_TOOL_NAMES` — every name this atom can register.
* :data:`EXTRACTOR_TERMINATION_REASON` — the ``ToolTerminate.reason``
  the child loop emits when the model calls ``finalize_extraction``.

§11 contract: single file, no atom-to-atom imports; the sibling tool
builders under ``tools/`` (``upsert_node.py`` / ``upsert_edge.py`` /
...) and the ``state/`` extraction-state package are NOT atoms — they
expose pure builders and dataclasses this module imports.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from ..toolkit.atom_constants import EXTRACTOR_STATE_SERVICE_KEY
from .state import ExtractionState
from .tools.delete_edge import DELETE_EDGE_TOOL_NAME, build_delete_edge_tool
from .tools.delete_node import DELETE_NODE_TOOL_NAME, build_delete_node_tool
from .tools.finalize_extraction import (
    FINALIZE_EXTRACTION_REASON,
    FINALIZE_EXTRACTION_TOOL_NAME,
    build_finalize_extraction_tool,
)
from .tools.reset_extraction import (
    RESET_EXTRACTION_TOOL_NAME,
    build_reset_extraction_tool,
)
from .tools.upsert_edge import UPSERT_EDGE_TOOL_NAME, build_upsert_edge_tool
from .tools.upsert_node import UPSERT_NODE_TOOL_NAME, build_upsert_node_tool

EXTRACTOR_TOOL_NAMES: tuple[str, ...] = (
    UPSERT_NODE_TOOL_NAME,
    DELETE_NODE_TOOL_NAME,
    UPSERT_EDGE_TOOL_NAME,
    DELETE_EDGE_TOOL_NAME,
    RESET_EXTRACTION_TOOL_NAME,
    FINALIZE_EXTRACTION_TOOL_NAME,
)

EXTRACTOR_TERMINATION_REASON: str = FINALIZE_EXTRACTION_REASON


MANIFEST = ExtensionManifest(
    name="extractor_tools",
    description=(
        "Register the v19 extractor child-session tool surface — "
        "incremental upserts / deletes plus the no-payload "
        "``finalize_extraction`` terminator. Replaces the legacy "
        "``submit_events_batch`` flow with a tool set that gets "
        "narrow validation feedback per edit, so witness errors "
        "are self-correcting rather than batch-rejection retries."
    ),
    registers=tuple(f"tool:{name}" for name in EXTRACTOR_TOOL_NAMES),
    config_schema={
        "type": "object",
        "properties": {
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Tool names to mount; defaults to every EXTRACTOR_TOOL_NAMES entry."
                ),
            },
            "state": {"type": "object"},
        },
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)


_BUILDERS: dict[str, Any] = {
    UPSERT_NODE_TOOL_NAME: build_upsert_node_tool,
    DELETE_NODE_TOOL_NAME: build_delete_node_tool,
    UPSERT_EDGE_TOOL_NAME: build_upsert_edge_tool,
    DELETE_EDGE_TOOL_NAME: build_delete_edge_tool,
    RESET_EXTRACTION_TOOL_NAME: build_reset_extraction_tool,
    FINALIZE_EXTRACTION_TOOL_NAME: build_finalize_extraction_tool,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # Two handoff channels for the per-firing ExtractionState — the
    # spawn-time ``config['state']`` path (used by the live adapter)
    # and the service-registry fallback (used by unit tests that
    # pre-publish the state on the same session).
    state = config.get("state") if isinstance(config, dict) else None
    if not isinstance(state, ExtractionState):
        state = api.get_service(EXTRACTOR_STATE_SERVICE_KEY)
    if not isinstance(state, ExtractionState):
        raise RuntimeError(
            "extractor_tools.install: expected an ExtractionState via "
            f"config['state'] or service key {EXTRACTOR_STATE_SERVICE_KEY!r}; "
            "the adapter must supply one before mounting this extension."
        )

    tools_raw = config.get("tools") if isinstance(config, dict) else None
    if tools_raw is None:
        tools = list(EXTRACTOR_TOOL_NAMES)
    elif isinstance(tools_raw, (list, tuple)):
        tools = list(tools_raw)
    else:
        tools = list(EXTRACTOR_TOOL_NAMES)

    unknown = [t for t in tools if t not in EXTRACTOR_TOOL_NAMES]
    if unknown:
        raise ValueError(
            f"extractor_tools: unknown tool names in config['tools']: "
            f"{unknown!r}; known: {EXTRACTOR_TOOL_NAMES!r}"
        )

    for name in tools:
        api.register_tool(_BUILDERS[name](state))


__all__ = [
    "EXTRACTOR_STATE_SERVICE_KEY",
    "EXTRACTOR_TERMINATION_REASON",
    "EXTRACTOR_TOOL_NAMES",
    "MANIFEST",
    "install",
]
