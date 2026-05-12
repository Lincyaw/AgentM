"""Extractor tool profiles — symmetric to the auditor profile system.

The v3.1 extractor uses a single ``submit_events`` tool, so today only
one profile exists. The knob is exposed for symmetry and future A/B
(e.g. a profile that splits submission into incremental
``register_event`` + ``add_edge`` for very small models).
"""

from __future__ import annotations

TOOL_SUBMIT_EVENTS = "submit_events"


PROFILES: dict[str, tuple[str, ...]] = {
    "minimal": (TOOL_SUBMIT_EVENTS,),
}

DEFAULT_PROFILE = "minimal"


def resolve_tools(
    *,
    profile: str | None,
    tools: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Resolve a (profile, tools) pair into the actual tool tuple.

    Same contract as
    :func:`llmharness.audit.auditor.profiles.resolve_tools`.
    ``submit_events`` is always present.
    """
    if tools is not None:
        resolved = tuple(t for t in tools if isinstance(t, str) and t)
    else:
        key = profile if isinstance(profile, str) and profile else DEFAULT_PROFILE
        resolved = PROFILES.get(key, PROFILES[DEFAULT_PROFILE])

    if TOOL_SUBMIT_EVENTS not in resolved:
        resolved = (TOOL_SUBMIT_EVENTS, *resolved)
    return resolved


__all__ = [
    "DEFAULT_PROFILE",
    "PROFILES",
    "TOOL_SUBMIT_EVENTS",
    "resolve_tools",
]
