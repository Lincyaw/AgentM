"""Adaptive IFG intervention metrics and policy behavior."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from policy_engine.compiler import compile_policy_file
from policy_engine.evaluator import PolicyEvaluator
from policy_engine.ifg_interventions import IfgInterventionState, InterventionQuery
from policy_engine.persistence import PolicyPersistence
from policy_engine.projection import (
    PolicyProjector,
    PolicyToolCallProjectionEvent,
    PolicyToolResultProjectionEvent,
)
from policy_engine.source_parser import parse_bash_segments
from policy_engine.source_semantics import analyze_bash_segment
from policy_engine.state import PolicyState


def _base_rules(*names: str):
    policy_path = (
        Path(__file__).parents[1]
        / "contrib/extensions/policy/src/policy_engine/ifg_evidence.yaml"
    )
    rules, _disabled = compile_policy_file(policy_path.read_text(encoding="utf-8"))
    selected = set(names)
    return [rule for rule in rules if rule.rule_id in selected]


def _record(
    state: PolicyState,
    evaluator: PolicyEvaluator,
    tool: str,
    args: dict[str, object],
    result: dict[str, object] | None = None,
) -> None:
    outcome = result or {"text": "ok"}
    state.record_tool_call(tool, args, outcome)
    evaluator.evaluate(
        "tool_result",
        tool,
        args=args,
        result={
            "text": str(outcome.get("text") or ""),
            "error": str(outcome["error"]) if outcome.get("error") else None,
        },
    )


def test_package_runner_with_global_options_is_classified_as_test() -> None:
    segments = parse_bash_segments(
        "cd /repo && pnpm --dir packages/app test src/app.test.ts"
    )
    actions = [analyze_bash_segment(segment).action_kind for segment in segments]

    assert actions == ["control", "test"]


def test_coherent_intervention_closes_without_convergence_signals() -> None:
    state = PolicyState()
    evaluator = PolicyEvaluator(
        _base_rules(
            "exploration-not-converging",
            "mutation-target-drift",
            "unvalidated-intervention",
        ),
        state,
    )
    for index in range(9):
        _record(state, evaluator, "read", {"path": f"src/file_{index}.py"})

    for _index in range(5):
        _record(state, evaluator, "edit", {"path": "src/file_0.py"})

    query = InterventionQuery(state.ifg_interventions)
    assert query.active()
    assert query.mutation_count() == 5
    assert query.distinct_target_files() == 1
    assert query.effective_targets() == 1.0
    assert query.novel_files() == 0
    assert not list(state.effect_log.entries())

    _record(
        state,
        evaluator,
        "bash",
        {"cmd": "cd /repo && pnpm --dir packages/app test src/file_0.test.py"},
        {"text": "1 passed", "exit_code": 0},
    )

    assert not query.active()
    assert query.summary()["validation_attempts"] == 1
    assert not list(state.effect_log.entries())


def test_expanding_frontier_fires_once_from_adaptive_support() -> None:
    state = PolicyState()
    evaluator = PolicyEvaluator(
        _base_rules("exploration-not-converging"),
        state,
    )
    for index in range(9):
        _record(state, evaluator, "read", {"path": f"src/known_{index}.py"})

    _record(state, evaluator, "write", {"path": "src/hypothesis.py"})
    for index in range(4):
        _record(state, evaluator, "read", {"path": f"src/branch_{index}.py"})

    effects = list(state.effect_log.entries())
    assert [effect.rule_id for effect in effects] == ["exploration-not-converging"]
    assert "novel_files: 4" in effects[0].reason
    assert "adaptive_support: 3" in effects[0].reason
    assert "observed_signals: ['expanding']" in effects[0].reason

    _record(state, evaluator, "read", {"path": "src/another_branch.py"})
    assert len(state.effect_log.entries()) == 1


def test_target_drift_uses_effective_targets_instead_of_fixed_file_limit() -> None:
    state = PolicyState()
    evaluator = PolicyEvaluator(_base_rules("mutation-target-drift"), state)
    for index in range(9):
        _record(state, evaluator, "read", {"path": f"src/known_{index}.py"})

    for index in range(3):
        _record(state, evaluator, "edit", {"path": f"src/known_{index}.py"})

    effects = list(state.effect_log.entries())
    assert [effect.rule_id for effect in effects] == ["mutation-target-drift"]
    assert "effective_targets: 3.0" in effects[0].reason


def test_untrusted_pipeline_does_not_close_intervention_epoch() -> None:
    state = IfgInterventionState()
    for index in range(9):
        state.record("read", {"path": f"src/file_{index}.py"}, {"text": "ok"})
    state.record("edit", {"path": "src/file_0.py"}, {"text": "ok"})
    state.record(
        "bash",
        {"cmd": "pytest src/file_0.py 2>&1 | grep FAILED | head"},
        {"text": "", "exit_code": 0},
    )

    query = InterventionQuery(state)
    assert query.active()
    assert query.summary()["validation_attempts"] == 1
    assert query.summary()["untrusted_validations"] == 1

    for _index in range(3):
        state.record("bash", {"cmd": "rg next src"}, {"text": "", "exit_code": 0})

    assert query.became_unvalidated()
    assert query.summary()["adaptive_patience"] == 3


def test_unrelated_targeted_test_does_not_close_intervention_epoch() -> None:
    state = IfgInterventionState()
    state.record("read", {"path": "src/account.py"}, {"text": "ok"})
    state.record("edit", {"path": "src/account.py"}, {"text": "ok"})
    state.record(
        "bash",
        {"cmd": "pytest src/billing_test.py"},
        {"text": "1 passed", "exit_code": 0},
    )

    query = InterventionQuery(state)
    assert query.active()
    assert query.summary()["untrusted_validations"] == 1

    state.record(
        "bash",
        {"cmd": "pytest src/test_account.py"},
        {"text": "1 passed", "exit_code": 0},
    )

    assert not query.active()
    assert query.summary()["completed_epochs"] == 1


def test_projector_persists_live_intervention_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "policy.db"
    persistence = PolicyPersistence(db_path)
    persistence.open()
    projector = PolicyProjector(
        session_id="live-session",
        rules=(),
        persistence=persistence,
    )

    for call_id, tool, path in (
        ("read-1", "read", "src/app.py"),
        ("edit-1", "edit", "src/app.py"),
    ):
        args = {"path": path}
        projector.process(
            PolicyToolCallProjectionEvent(
                turn_index=0,
                tool_call_id=call_id,
                tool_name=tool,
                args=args,
            )
        )
        projector.process(
            PolicyToolResultProjectionEvent(
                turn_index=0,
                tool_call_id=call_id,
                tool_name=tool,
                args=args,
                result_text="ok",
                is_error=False,
                extras={},
            )
        )

    with sqlite3.connect(db_path) as connection:
        raw_summary = connection.execute(
            "SELECT summary_json FROM policy_session_summary WHERE session_id = ?",
            ("live-session",),
        ).fetchone()
    persistence.close()

    assert raw_summary is not None
    intervention = json.loads(raw_summary[0])["intervention"]
    assert intervention["active"] is True
    assert intervention["mutations"] == 1
    assert intervention["known_files_before"] == 1
