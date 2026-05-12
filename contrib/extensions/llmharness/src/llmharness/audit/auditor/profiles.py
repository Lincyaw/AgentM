"""Auditor tool profiles — pluggable composition of the auditor child.

Two named presets ship today; adding a new one is a one-line addition
to :data:`PROFILES`. The adapter resolves the configured profile (or
explicit ``tools`` list) into the tuple of tool names that the
auditor child should mount. The same tuple flows into
:func:`build_auditor_system_prompt` as ``available_tools`` so the
prompt only mentions tools that are actually present.

This module is import-pure (no I/O, no side effects) — safe to import
from atom files and the adapter alike.
"""

from __future__ import annotations

# Tool identifiers. These match the ``name=`` field on each tool's
# ``api.register_tool`` call.
TOOL_SUBMIT_VERDICT = "submit_verdict"
TOOL_GET_EVENT_DETAIL = "get_event_detail"
TOOL_GET_TURN = "get_turn"


PROFILES: dict[str, tuple[str, ...]] = {
    # The default. Single-shot single-tool. Designed for small-model
    # SFT targets: the student sees exactly one tool and is trained to
    # call it once. No drill-down, no multi-turn planning.
    "minimal": (TOOL_SUBMIT_VERDICT,),
    # Full drill-down surface — matches pre-profile behavior. Use for
    # larger teacher models or as the A/B upper-bound.
    "with_drill_down": (
        TOOL_SUBMIT_VERDICT,
        TOOL_GET_EVENT_DETAIL,
        TOOL_GET_TURN,
    ),
}

DEFAULT_PROFILE = "minimal"


def resolve_tools(
    *,
    profile: str | None,
    tools: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Resolve a (profile, tools) pair into the actual tool tuple.

    Explicit ``tools`` wins. Otherwise the named profile is expanded.
    Unknown profile or empty resolution falls back to
    :data:`DEFAULT_PROFILE`. ``submit_verdict`` is always present —
    silently re-added if the caller dropped it, since an auditor that
    can't terminate has no value.
    """
    if tools is not None:
        resolved = tuple(t for t in tools if isinstance(t, str) and t)
    else:
        key = profile if isinstance(profile, str) and profile else DEFAULT_PROFILE
        resolved = PROFILES.get(key, PROFILES[DEFAULT_PROFILE])

    if TOOL_SUBMIT_VERDICT not in resolved:
        resolved = (TOOL_SUBMIT_VERDICT, *resolved)
    return resolved


__all__ = [
    "DEFAULT_PROFILE",
    "PROFILES",
    "TOOL_GET_EVENT_DETAIL",
    "TOOL_GET_TURN",
    "TOOL_SUBMIT_VERDICT",
    "resolve_tools",
]
