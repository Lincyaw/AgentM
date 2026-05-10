"""Phase 2 auditor child-session ``extensions`` list (v3)."""

from __future__ import annotations

from typing import Any

from ...schema import Edge, Event, Finding
from .._compose import UNSET, compose_audit_extensions
from .prompt import AUDITOR_SYSTEM_PROMPT, build_auditor_system_prompt

_SUBMIT_TOOL_MODULE = "llmharness.audit.auditor.submit_tool"
_GET_TURN_TOOL_MODULE = "llmharness.audit.auditor.get_turn_tool"
_GET_EVENT_DETAIL_TOOL_MODULE = "llmharness.audit.auditor.get_event_detail_tool"


def compose_auditor_extensions(
    *,
    prompt_override: str | None = None,
    cards_tools_config: dict[str, Any] | None = UNSET,
    observability_config: dict[str, Any] | None = UNSET,
    trajectory_snapshot: list[dict[str, Any]] | None = None,
    events: tuple[Event, ...] | None = None,
    edges: tuple[Edge, ...] | None = None,
    findings: list[Finding] | None = None,
    check_errors: dict[str, str] | None = None,
    continuation_notes: list[str] | None = None,
    summary_threshold: int = 30,
) -> list[tuple[str, dict[str, Any]]]:
    """Build the extensions list for an auditor firing.

    Default order: observability → cards_tools → submit_tool → system_prompt
    → (get_turn_tool) → (get_event_detail_tool).

    Pass ``None`` for ``cards_tools_config`` / ``observability_config`` to drop
    that extension; ``submit_tool`` and ``system_prompt`` always survive.

    When ``events`` and ``edges`` are provided, ``get_event_detail_tool`` is
    appended and the v3 system prompt is built via
    :func:`build_auditor_system_prompt` (unless ``prompt_override`` is set).
    Without those, the static :data:`AUDITOR_SYSTEM_PROMPT` is used —
    convenient for early bootstrap and tests that don't need the v3 inputs.

    The bridging contract matches commit 3's pattern: per-firing data is
    handed to atom installs through the ``config`` dict.
    """
    if prompt_override is not None:
        prompt_text = prompt_override
    elif events is not None or edges is not None:
        prompt_text = build_auditor_system_prompt(
            events=events or (),
            edges=edges or (),
            findings=findings or [],
            check_errors=check_errors or {},
            continuation_notes=continuation_notes or [],
            summary_threshold=summary_threshold,
        )
    else:
        prompt_text = AUDITOR_SYSTEM_PROMPT

    extensions = compose_audit_extensions(
        submit_tool_module=_SUBMIT_TOOL_MODULE,
        default_prompt=prompt_text,
        prompt_override=None,
        cards_tools_config=cards_tools_config,
        observability_config=observability_config,
    )
    if trajectory_snapshot is not None:
        extensions.append((_GET_TURN_TOOL_MODULE, {"trajectory_snapshot": trajectory_snapshot}))
    if events is not None or edges is not None:
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


__all__ = ["compose_auditor_extensions"]
