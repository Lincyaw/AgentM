"""Fail-stop tests for the auditor profile + prompt-variant machinery.

The default ``minimal`` profile mounts only ``submit_verdict``. The
``with_drill_down`` profile additionally mounts ``get_event_detail``
and ``get_turn``. Prompt variants are loaded from
``audit/auditor/prompts/auditor_<name>.md`` so adding a new variant is
pure data — no code change.

Why fail-stop: if minimal accidentally pulls in drill-down tools, the
small-model SFT target diverges from the deployed inference surface.
If prompt loading drifts, training/inference framing diverges silently.
"""

from __future__ import annotations

from llmharness.audit.auditor.extensions import compose_auditor_extensions
from llmharness.audit.auditor.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    TOOL_GET_EVENT_DETAIL,
    TOOL_GET_TURN,
    TOOL_SUBMIT_VERDICT,
    resolve_tools,
)
from llmharness.audit.auditor.prompt import load_auditor_prompt
from llmharness.schema import Edge, EdgeKind, Event, EventKind


def _ev() -> tuple[Event, ...]:
    return (Event(id=1, kind=EventKind.ACT, summary="a", source_turns=[0]),)


def _ed() -> tuple[Edge, ...]:
    return (
        Edge(
            src=1,
            dst=1,
            kind=EdgeKind.DATA,
            reason="r",
            src_turns=(0,),
            dst_turns=(0,),
            cited_entities=("x",),
            cited_quote="",
        ),
    )


# --- profile resolution -----------------------------------------------------


def test_default_profile_is_minimal_and_only_has_submit() -> None:
    assert DEFAULT_PROFILE == "minimal"
    assert PROFILES["minimal"] == (TOOL_SUBMIT_VERDICT,)


def test_resolve_unknown_profile_falls_back_to_default() -> None:
    assert resolve_tools(profile="does-not-exist", tools=None) == (TOOL_SUBMIT_VERDICT,)


def test_resolve_explicit_tools_overrides_profile() -> None:
    out = resolve_tools(profile="with_drill_down", tools=[TOOL_GET_TURN])
    # submit_verdict force-included; the user-supplied tool is kept.
    assert TOOL_SUBMIT_VERDICT in out
    assert TOOL_GET_TURN in out
    assert TOOL_GET_EVENT_DETAIL not in out


def test_submit_verdict_is_force_included_even_when_omitted() -> None:
    out = resolve_tools(profile=None, tools=[])
    assert TOOL_SUBMIT_VERDICT in out


# --- compose wiring ---------------------------------------------------------


def test_minimal_compose_omits_drill_down_modules() -> None:
    exts = compose_auditor_extensions(
        trajectory_snapshot=[{"index": 0, "role": "user", "content": []}],
        events=_ev(),
        edges=_ed(),
        findings=[],
        check_errors={},
        continuation_notes=[],
        # tools=None -> minimal default
    )
    modules = {mod for mod, _cfg in exts}
    assert "llmharness.audit.auditor.submit_tool" in modules
    assert "llmharness.audit.auditor.get_event_detail_tool" not in modules
    assert "llmharness.audit.auditor.get_turn_tool" not in modules


def test_with_drill_down_compose_mounts_all_three_modules() -> None:
    exts = compose_auditor_extensions(
        trajectory_snapshot=[{"index": 0, "role": "user", "content": []}],
        events=_ev(),
        edges=_ed(),
        findings=[],
        check_errors={},
        continuation_notes=[],
        tools=PROFILES["with_drill_down"],
    )
    modules = {mod for mod, _cfg in exts}
    assert "llmharness.audit.auditor.submit_tool" in modules
    assert "llmharness.audit.auditor.get_event_detail_tool" in modules
    assert "llmharness.audit.auditor.get_turn_tool" in modules


# --- prompt-file loading ----------------------------------------------------


def test_minimal_prompt_does_not_reference_drill_down_tools() -> None:
    text = load_auditor_prompt("minimal")
    assert "get_event_detail" not in text
    assert "get_turn" not in text
    assert "DEGRADED MODE" not in text


def test_full_prompt_does_reference_drill_down_tools() -> None:
    text = load_auditor_prompt("full")
    assert "get_event_detail" in text
    assert "get_turn" in text


def test_unknown_prompt_name_raises() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_auditor_prompt("totally-bogus-variant")
