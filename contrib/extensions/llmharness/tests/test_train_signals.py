"""Lock the process-reward contract for external trainers.

These functions are imported by ``rca-autorl``'s GRPO/PPO/DPO drivers
to compute deterministic process rewards from replay-derived tool-event
lists. A regression here silently shifts training signal — that's
load-bearing for the cognitive-audit→RL pipeline, so we pin behavior at
the math level.
"""

from __future__ import annotations

import math

from llmharness import (
    ToolEvent,
    auditor_process_reward,
    extractor_process_reward,
)


def _ok(name: str) -> ToolEvent:
    return ToolEvent(tool_name=name, args={}, is_error=False, error_text=None)


def _err(name: str, msg: str = "boom") -> ToolEvent:
    return ToolEvent(tool_name=name, args={}, is_error=True, error_text=msg)


# --- extractor --------------------------------------------------------------


def test_extractor_happy_path_six_steps_finalize() -> None:
    events: list[ToolEvent] = [
        _ok("upsert_node"),
        _ok("upsert_node"),
        _ok("upsert_edge"),
        _ok("upsert_node"),
        _ok("upsert_edge"),
        _ok("finalize_extraction"),
    ]
    r = extractor_process_reward(events)
    assert r["finalize_success"] == 1.0
    assert r["witness_pass_rate"] == 1.0
    # 6 / 32 budget
    assert math.isclose(r["efficiency_penalty"], 6 / 32)
    # 0.5*1 + 0.3*1 - 0.2*(6/32)
    assert math.isclose(r["reward"], 0.5 + 0.3 - 0.2 * (6 / 32))


def test_extractor_midflight_error_drags_witness_rate() -> None:
    events: list[ToolEvent] = [
        _ok("upsert_node"),
        _err("upsert_edge"),
        _ok("upsert_node"),
        _ok("upsert_edge"),
        _ok("upsert_node"),
        _ok("finalize_extraction"),
    ]
    r = extractor_process_reward(events)
    assert r["finalize_success"] == 1.0
    assert math.isclose(r["witness_pass_rate"], 5 / 6)


def test_extractor_never_finalized() -> None:
    events: list[ToolEvent] = [
        _ok("upsert_node"),
        _ok("upsert_edge"),
        _ok("upsert_node"),
    ]
    r = extractor_process_reward(events)
    assert r["finalize_success"] == 0.0
    assert r["witness_pass_rate"] == 1.0


def test_extractor_finalize_call_failed_doesnt_count() -> None:
    # last call is finalize_extraction but is_error=True → terminal failure
    events: list[ToolEvent] = [_ok("upsert_node"), _err("finalize_extraction")]
    r = extractor_process_reward(events)
    assert r["finalize_success"] == 0.0


def test_extractor_empty_input_all_zero() -> None:
    r = extractor_process_reward([])
    assert r == {
        "reward": 0.0,
        "witness_pass_rate": 0.0,
        "finalize_success": 0.0,
        "efficiency_penalty": 0.0,
    }


# --- auditor ----------------------------------------------------------------


def test_auditor_happy_path_single_submit_verdict() -> None:
    events: list[ToolEvent] = [_ok("submit_verdict")]
    r = auditor_process_reward(events)
    assert r["verdict_submitted"] == 1.0
    assert r["schema_valid_rate"] == 1.0
    assert math.isclose(r["efficiency_penalty"], 1 / 8)
    assert math.isclose(r["reward"], 0.7 + 0.2 - 0.1 * (1 / 8))


def test_auditor_empty_input_all_zero() -> None:
    r = auditor_process_reward([])
    assert r == {
        "reward": 0.0,
        "schema_valid_rate": 0.0,
        "verdict_submitted": 0.0,
        "efficiency_penalty": 0.0,
    }


def test_auditor_schema_error_then_resubmit_terminal_ok() -> None:
    events: list[ToolEvent] = [_err("submit_verdict", "schema"), _ok("submit_verdict")]
    r = auditor_process_reward(events)
    assert r["verdict_submitted"] == 1.0
    assert math.isclose(r["schema_valid_rate"], 1 / 2)
