"""§11 single-file extension: register the auditor tool surface.

Merges the three legacy atoms (``submit_tool`` / ``get_turn_tool`` /
``get_event_detail_tool``) into one module. The composer decides which
tools the child should mount via the ``tools`` config key — names that
are absent are silently skipped, so the same atom serves both the
``minimal`` (submit-only) and ``with_drill_down`` profiles.

Config schema:

* ``tools`` (``list[str]``, default = ``["submit_verdict"]``) — names of
  the tools to register. Unknown names raise ``ValueError`` so a typo in
  the composer never silently produces a degenerate child.
* ``trajectory_snapshot`` (``list[dict]``) — only consulted when
  ``get_turn`` is requested.
* ``events`` / ``edges`` — only consulted when ``get_event_detail`` is
  requested. Each accepts a list of either dataclass instances or their
  ``to_dict()`` shape.

Public contract (also re-exported from :mod:`llmharness.__init__`):

* :data:`AUDITOR_TOOLS` — the stateless prototypes (just the
  ``submit_verdict`` tool today; the two drill-down tools are built
  per-firing from config, but their stateless schemas are reachable via
  their submodules: see ``GET_TURN_PARAMETERS`` /
  ``GET_EVENT_DETAIL_PARAMETERS``).
* :data:`AUDITOR_TOOL_NAMES` — every name this atom can register.
* :data:`AUDITOR_TERMINATION_REASON` — the ``ToolTerminate.reason`` the
  child loop emits when the auditor calls the terminal tool.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from .get_event_detail import (
    GET_EVENT_DETAIL_TOOL_NAME,
    build_get_event_detail_tool,
)
from .get_turn import GET_TURN_TOOL_NAME, build_get_turn_tool
from .submit_verdict import SUBMIT_VERDICT_TOOL, SUBMIT_VERDICT_TOOL_NAME

MANIFEST = ExtensionManifest(
    name="auditor_tools",
    description=(
        "Register the auditor child-session tool surface — ``submit_verdict`` "
        "(terminal) plus optionally the ``get_turn`` and "
        "``get_event_detail`` drill-down tools, selected via the ``tools`` "
        "config key. Replaces the three legacy atoms "
        "(auditor_submit_tool / auditor_get_turn_tool / "
        "auditor_get_event_detail_tool)."
    ),
    registers=(
        "tool:submit_verdict",
        "tool:get_turn",
        "tool:get_event_detail",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": ("Tool names to mount; defaults to ['submit_verdict']."),
            },
            "trajectory_snapshot": {
                "type": "array",
                "items": {"type": "object"},
            },
            "events": {
                "type": "array",
                "items": {"type": "object"},
            },
            "edges": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)


AUDITOR_TOOLS: tuple[FunctionTool, ...] = (SUBMIT_VERDICT_TOOL,)
AUDITOR_TOOL_NAMES: tuple[str, ...] = (
    SUBMIT_VERDICT_TOOL_NAME,
    GET_TURN_TOOL_NAME,
    GET_EVENT_DETAIL_TOOL_NAME,
)
AUDITOR_TERMINATION_REASON: str = "llmharness:submit_verdict"


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    tools_raw = config.get("tools", [SUBMIT_VERDICT_TOOL_NAME])
    tools = list(tools_raw) if isinstance(tools_raw, (list, tuple)) else [SUBMIT_VERDICT_TOOL_NAME]

    unknown = [t for t in tools if t not in AUDITOR_TOOL_NAMES]
    if unknown:
        raise ValueError(
            f"auditor_tools: unknown tool names in config['tools']: {unknown!r}; "
            f"known: {AUDITOR_TOOL_NAMES!r}"
        )

    if SUBMIT_VERDICT_TOOL_NAME in tools:
        api.register_tool(SUBMIT_VERDICT_TOOL)

    if GET_TURN_TOOL_NAME in tools:
        snapshot_raw = config.get("trajectory_snapshot", [])
        snapshot: list[dict[str, Any]] = (
            list(snapshot_raw) if isinstance(snapshot_raw, list) else []
        )
        api.register_tool(build_get_turn_tool(snapshot))

    if GET_EVENT_DETAIL_TOOL_NAME in tools:
        events = config.get("events", ())
        edges = config.get("edges", ())
        api.register_tool(build_get_event_detail_tool(events, edges))


__all__ = [
    "AUDITOR_TERMINATION_REASON",
    "AUDITOR_TOOLS",
    "AUDITOR_TOOL_NAMES",
    "MANIFEST",
    "install",
]
