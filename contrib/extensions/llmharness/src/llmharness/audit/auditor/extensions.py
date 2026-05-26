"""Phase 2 auditor child-session ``extensions`` list (v3).

Pluggability knobs:

* ``tools`` — tuple of tool names (from :mod:`audit.auditor.profiles`)
  that the child should mount. Default: ``("submit_verdict",)``.
* ``base_prompt`` — framing text the dynamic builder appends
  PHASES / GRAPH / FINDINGS / CONTINUATION_NOTES onto. The adapter
  resolves this from the ``auditor_prompt`` config key (a named
  variant or absolute path).

To add a new variant without touching this file: drop a markdown
file under ``audit/auditor/prompts/`` and point ``auditor_prompt``
at it; optionally add an entry to
:data:`audit.auditor.profiles.PROFILES` for a new tool combo.
"""

from __future__ import annotations

from typing import Any

from ...schema import Edge, Event, Finding, Phase
from ..seams.compose import UNSET, compose_audit_extensions
from .profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    TOOL_GET_EVENT_DETAIL,
    TOOL_GET_TURN,
    TOOL_SUBMIT_VERDICT,
    resolve_tools,
)
from .prompt import (
    DEFAULT_PROMPT_NAME,
    build_auditor_system_prompt,
    load_auditor_prompt,
)

_AUDITOR_TOOLS_MODULE = "llmharness.audit.auditor.atom"


def compose_auditor_extensions(
    *,
    base_prompt: str | None = None,
    observability_config: dict[str, Any] | None = UNSET,
    trajectory_snapshot: list[dict[str, Any]] | None = None,
    events: tuple[Event, ...] | None = None,
    edges: tuple[Edge, ...] | None = None,
    phases: tuple[Phase, ...] | None = None,
    findings: list[Finding] | None = None,
    check_errors: dict[str, str] | None = None,
    continuation_notes: list[str] | None = None,
    summary_threshold: int = 30,
    tools: tuple[str, ...] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Build the extensions list for an auditor firing.

    ``base_prompt`` is the framing text. When ``events`` / ``edges``
    are provided, it is passed to :func:`build_auditor_system_prompt`
    and the dynamic data sections are appended; otherwise it is used
    as-is. Default framing is the ``minimal`` variant loaded via
    :func:`load_auditor_prompt`.

    ``tools`` decides which tool atoms are appended. The submit tool
    is always mounted by :func:`compose_audit_extensions`; drill-down
    tools are appended here only when their name is in ``tools`` AND
    the relevant inputs are available.
    """
    tools_tuple = tools if tools is not None else PROFILES[DEFAULT_PROFILE]
    framing = base_prompt if base_prompt is not None else load_auditor_prompt(DEFAULT_PROMPT_NAME)

    if events is not None or edges is not None:
        prompt_text = build_auditor_system_prompt(
            events=events or (),
            edges=edges or (),
            phases=phases or (),
            findings=findings or [],
            check_errors=check_errors or {},
            continuation_notes=continuation_notes or [],
            summary_threshold=summary_threshold,
            base_prompt=framing,
        )
    else:
        prompt_text = framing

    # Build the auditor_tools config: which tools to mount + any state the
    # drill-down tools need. Names absent from ``tools_tuple`` are omitted
    # so the merged atom only registers what the profile asks for. Drill-
    # down state is only attached when the relevant inputs are available;
    # otherwise the atom would still mount the tool but over an empty
    # backing snapshot/graph, which is a worse failure mode than simply
    # not exposing the tool.
    selected: list[str] = []
    if TOOL_SUBMIT_VERDICT in tools_tuple:
        selected.append(TOOL_SUBMIT_VERDICT)
    if TOOL_GET_TURN in tools_tuple and trajectory_snapshot is not None:
        selected.append(TOOL_GET_TURN)
    if TOOL_GET_EVENT_DETAIL in tools_tuple and (events is not None or edges is not None):
        selected.append(TOOL_GET_EVENT_DETAIL)

    auditor_tools_cfg: dict[str, Any] = {"tools": selected}
    if TOOL_GET_TURN in selected:
        auditor_tools_cfg["trajectory_snapshot"] = trajectory_snapshot
    if TOOL_GET_EVENT_DETAIL in selected:
        auditor_tools_cfg["events"] = list(events or ())
        auditor_tools_cfg["edges"] = list(edges or ())

    extensions = compose_audit_extensions(
        submit_tool_module=_AUDITOR_TOOLS_MODULE,
        default_prompt=prompt_text,
        observability_config=observability_config,
        submit_tool_config=auditor_tools_cfg,
    )
    return extensions


__all__ = ["compose_auditor_extensions", "resolve_tools"]
