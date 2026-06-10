"""Fail-stop tests for the auditor profile + prompt-variant machinery."""

from __future__ import annotations

from llmharness.agents.auditor.prompt import load_auditor_prompt
from llmharness.agents.auditor.tools import (
    GET_EVENT_DETAIL_TOOL_NAME,
    GET_TURN_TOOL_NAME,
    PROFILES,
    SUBMIT_VERDICT_TOOL_NAME,
    _resolve_tools,
)


def test_default_profile_is_minimal_and_only_has_submit() -> None:
    assert PROFILES["minimal"] == (SUBMIT_VERDICT_TOOL_NAME,)


def test_resolve_unknown_profile_falls_back_to_default() -> None:
    assert _resolve_tools({"profile": "does-not-exist"}) == (SUBMIT_VERDICT_TOOL_NAME,)


def test_resolve_explicit_tools_overrides_profile() -> None:
    out = _resolve_tools({"profile": "with_drill_down", "tools": [GET_TURN_TOOL_NAME]})
    assert SUBMIT_VERDICT_TOOL_NAME in out
    assert GET_TURN_TOOL_NAME in out
    assert GET_EVENT_DETAIL_TOOL_NAME not in out


def test_submit_verdict_is_force_included_even_when_omitted() -> None:
    out = _resolve_tools({"tools": []})
    assert SUBMIT_VERDICT_TOOL_NAME in out


def test_minimal_prompt_does_not_reference_drill_down_tools() -> None:
    text = load_auditor_prompt("minimal")
    assert "get_event_detail" not in text
    assert "get_turn" not in text
    assert "DEGRADED MODE" not in text


def test_unknown_prompt_name_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        load_auditor_prompt("totally-bogus-variant")
