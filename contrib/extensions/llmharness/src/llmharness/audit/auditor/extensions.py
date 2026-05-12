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
from .._compose import UNSET, compose_audit_extensions
from .profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    TOOL_GET_EVENT_DETAIL,
    TOOL_GET_TURN,
    resolve_tools,
)
from .prompt import (
    DEFAULT_PROMPT_NAME,
    build_auditor_system_prompt,
    load_auditor_prompt,
)

_SUBMIT_TOOL_MODULE = "llmharness.audit.auditor.submit_tool"
_GET_TURN_TOOL_MODULE = "llmharness.audit.auditor.get_turn_tool"
_GET_EVENT_DETAIL_TOOL_MODULE = "llmharness.audit.auditor.get_event_detail_tool"


def compose_auditor_extensions(
    *,
    base_prompt: str | None = None,
    cards_tools_config: dict[str, Any] | None = UNSET,
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
    tools_tuple = (
        tools if tools is not None else PROFILES[DEFAULT_PROFILE]
    )
    framing = (
        base_prompt
        if base_prompt is not None
        else load_auditor_prompt(DEFAULT_PROMPT_NAME)
    )

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

    extensions = compose_audit_extensions(
        submit_tool_module=_SUBMIT_TOOL_MODULE,
        default_prompt=prompt_text,
        prompt_override=None,
        cards_tools_config=cards_tools_config,
        observability_config=observability_config,
    )
    if TOOL_GET_TURN in tools_tuple and trajectory_snapshot is not None:
        extensions.append(
            (_GET_TURN_TOOL_MODULE, {"trajectory_snapshot": trajectory_snapshot})
        )
    if TOOL_GET_EVENT_DETAIL in tools_tuple and (
        events is not None or edges is not None
    ):
        extensions.append(
            (
                _GET_EVENT_DETAIL_TOOL_MODULE,
                {
                    "events": list(events or ()),
                    "edges": list(edges or ()),
                },
            )
        )
    return extensions


__all__ = ["compose_auditor_extensions", "resolve_tools"]
