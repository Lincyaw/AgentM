"""Fail-stop tests for the auditor profile + prompt-variant machinery.

The default ``minimal`` profile mounts only ``submit_verdict``. The
``with_drill_down`` profile additionally mounts ``get_event_detail``
and ``get_turn``. Prompt variants are loaded from
``agents/auditor/prompts/auditor_<name>.md``.

Why fail-stop: if minimal accidentally pulls in drill-down tools, the
small-model SFT target diverges from the deployed inference surface.
If prompt loading drifts, training/inference framing diverges silently.
"""

from __future__ import annotations

from llmharness.agents.auditor.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    TOOL_GET_EVENT_DETAIL,
    TOOL_GET_TURN,
    TOOL_SUBMIT_VERDICT,
    resolve_tools,
)
from llmharness.agents.auditor.prompt import load_auditor_prompt

# --- profile resolution -----------------------------------------------------


def test_default_profile_is_minimal_and_only_has_submit() -> None:
    assert DEFAULT_PROFILE == "minimal"
    assert PROFILES["minimal"] == (TOOL_SUBMIT_VERDICT,)


def test_resolve_unknown_profile_falls_back_to_default() -> None:
    assert resolve_tools(profile="does-not-exist", tools=None) == (TOOL_SUBMIT_VERDICT,)


def test_resolve_explicit_tools_overrides_profile() -> None:
    out = resolve_tools(profile="with_drill_down", tools=[TOOL_GET_TURN])
    assert TOOL_SUBMIT_VERDICT in out
    assert TOOL_GET_TURN in out
    assert TOOL_GET_EVENT_DETAIL not in out


def test_submit_verdict_is_force_included_even_when_omitted() -> None:
    out = resolve_tools(profile=None, tools=[])
    assert TOOL_SUBMIT_VERDICT in out


# --- prompt-file loading ----------------------------------------------------


def test_minimal_prompt_does_not_reference_drill_down_tools() -> None:
    text = load_auditor_prompt("minimal")
    assert "get_event_detail" not in text
    assert "get_turn" not in text
    assert "DEGRADED MODE" not in text


def test_unknown_prompt_name_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        load_auditor_prompt("totally-bogus-variant")
