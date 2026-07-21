"""Threshold-free structural investigation evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from policy_engine.compiler import compile_policy_file
from policy_engine.evaluator import PolicyEvaluator
from policy_engine.ifg_investigation import IfgInvestigationState, InvestigationQuery
from policy_engine.persistence import PolicyPersistence
from policy_engine.projection import (
    PolicyProjector,
    PolicyToolCallProjectionEvent,
    PolicyToolResultProjectionEvent,
)
from policy_engine.state import PolicyState


def _record(
    state: IfgInvestigationState,
    tool: str,
    args: dict[str, object],
    *,
    exit_code: int = 0,
) -> None:
    state.record(tool, args, {"exit_code": exit_code})


def _base_rules(*names: str):
    policy_path = (
        Path(__file__).parents[1]
        / "contrib/extensions/policy/src/policy_engine/ifg_evidence.yaml"
    )
    rules, _disabled = compile_policy_file(policy_path.read_text(encoding="utf-8"))
    selected = set(names)
    return [rule for rule in rules if rule.rule_id in selected]


def test_reentry_is_emitted_when_an_unchanged_anchor_is_revisited() -> None:
    state = IfgInvestigationState()
    query = InvestigationQuery(state)

    _record(state, "read", {"path": "pkg/alpha.ts"})
    _record(state, "read", {"path": "pkg/beta.ts"})
    assert not query.became_unchanged_reentry()

    _record(state, "read", {"path": "pkg/alpha.ts"})

    assert query.became_unchanged_reentry()
    evidence = query.summary()["current_evidence"]
    assert isinstance(evidence, list)
    assert evidence[0]["kind"] == "unchanged_anchor_reentry"
    assert evidence[0]["paths"] == ["pkg/alpha.ts"]


def test_mutating_an_anchor_does_not_emit_unchanged_reentry() -> None:
    state = IfgInvestigationState()
    query = InvestigationQuery(state)

    _record(state, "read", {"path": "pkg/alpha.ts"})
    _record(state, "read", {"path": "pkg/beta.ts"})
    _record(state, "edit", {"path": "pkg/alpha.ts"})

    assert not query.became_unchanged_reentry()
    assert query.summary()["repository_generation"] == 1


def test_successive_write_first_executions_emit_replacement_without_keywords() -> None:
    state = IfgInvestigationState(cwd="/repo")
    query = InvestigationQuery(state)

    _record(state, "read", {"path": "/repo/pkg/source.ts"})
    _record(state, "write", {"path": "/repo/pkg/a.ts", "content": ""})
    _record(
        state,
        "bash",
        {"cmd": "cd /repo/pkg && npx vitest run a.ts"},
    )
    assert not query.became_artifact_replacement()

    _record(state, "write", {"path": "/repo/pkg/b.ts", "content": ""})
    _record(
        state,
        "bash",
        {"cmd": "cd /repo/pkg && npx vitest run b.ts"},
    )

    assert query.became_artifact_replacement()
    evidence = query.summary()["current_evidence"]
    assert isinstance(evidence, list)
    replacement = next(
        item for item in evidence if item["kind"] == "created_artifact_replacement"
    )
    assert replacement["paths"] == ["/repo/pkg/a.ts", "/repo/pkg/b.ts"]


def test_repository_mutation_breaks_created_artifact_replacement_chain() -> None:
    state = IfgInvestigationState(cwd="/repo")
    query = InvestigationQuery(state)

    _record(state, "read", {"path": "/repo/pkg/source.ts"})
    _record(state, "write", {"path": "/repo/pkg/a.ts", "content": ""})
    _record(state, "bash", {"cmd": "cd /repo/pkg && npx vitest run a.ts"})
    _record(state, "edit", {"path": "/repo/pkg/source.ts"})
    _record(state, "write", {"path": "/repo/pkg/b.ts", "content": ""})
    _record(state, "bash", {"cmd": "cd /repo/pkg && npx vitest run b.ts"})

    assert not query.became_artifact_replacement()


def test_repeated_phase_state_emits_cycle_and_exact_focus_relation() -> None:
    state = IfgInvestigationState(cwd="/repo")
    query = InvestigationQuery(state)

    _record(state, "read", {"path": "/repo/pkg/source.ts"})
    _record(state, "write", {"path": "/repo/pkg/a.ts", "content": ""})
    _record(state, "bash", {"cmd": "cd /repo/pkg && npx vitest run a.ts"})

    _record(state, "read", {"path": "/repo/pkg/source.ts"})
    _record(state, "write", {"path": "/repo/pkg/b.ts", "content": ""})
    _record(state, "bash", {"cmd": "cd /repo/pkg && npx vitest run b.ts"})

    assert query.became_state_cycle()
    summary = query.summary()
    latest_phase = summary["latest_phase"]
    assert isinstance(latest_phase, dict)
    assert latest_phase["focus_relation"] == "same"
    assert summary["evidence_counts"] == {
        "created_artifact_replacement": 1,
        "unchanged_investigation_state_cycle": 1,
    }


def test_repository_index_fact_prevents_existing_write_from_becoming_created() -> None:
    state = IfgInvestigationState(cwd="/repo")
    state.configure_repository(
        contains_file=lambda path: path == "/repo/pkg/existing.ts",
    )

    _record(state, "write", {"path": "/repo/pkg/existing.ts", "content": ""})

    summary = state.summary()
    assert summary["repository_artifacts"] == 1
    assert summary["created_artifacts"] == 0
    assert summary["repository_generation"] == 1


def test_base_rules_persist_neutral_investigation_evidence() -> None:
    state = PolicyState(cwd="/repo")
    evaluator = PolicyEvaluator(
        _base_rules(
            "created-artifact-replacement",
            "unchanged-investigation-state-cycle",
        ),
        state,
    )

    def record(tool: str, args: dict[str, object]) -> None:
        result = {"exit_code": 0}
        state.record_tool_call(tool, args, result)
        evaluator.evaluate(
            "tool_result",
            tool,
            args=args,
            result={"text": "", "error": None},
        )

    record("read", {"path": "/repo/pkg/source.ts"})
    record("write", {"path": "/repo/pkg/a.ts", "content": ""})
    record("bash", {"cmd": "cd /repo/pkg && npx vitest run a.ts"})
    record("read", {"path": "/repo/pkg/source.ts"})
    record("write", {"path": "/repo/pkg/b.ts", "content": ""})
    record("bash", {"cmd": "cd /repo/pkg && npx vitest run b.ts"})

    effects = list(state.effect_log.entries())
    assert [effect.rule_id for effect in effects] == [
        "created-artifact-replacement",
        "unchanged-investigation-state-cycle",
    ]
    assert all(effect.mode == "observe" for effect in effects)
    assert all(effect.effect == "notify" for effect in effects)


def test_effect_log_persists_structured_investigation_evidence(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "policy.db"
    persistence = PolicyPersistence(db_path)
    persistence.open()
    projector = PolicyProjector(
        session_id="investigation-session",
        rules=_base_rules("created-artifact-replacement"),
        persistence=persistence,
    )

    def process(call_id: str, tool: str, args: dict[str, object]) -> None:
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
                result_text="",
                is_error=False,
                extras={"exit_code": 0},
            )
        )

    process("read", "read", {"path": "source.ts"})
    process("write-a", "write", {"path": "a.ts", "content": ""})
    process("run-a", "bash", {"cmd": "npx vitest run a.ts"})
    process("write-b", "write", {"path": "b.ts", "content": ""})
    process("run-b", "bash", {"cmd": "npx vitest run b.ts"})
    projector.finish()

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT context_json FROM event_log WHERE session_id = ? AND rule_id = ?",
            ("investigation-session", "created-artifact-replacement"),
        ).fetchone()
    persistence.close()

    assert row is not None
    context = json.loads(row[0])
    assert context["channel"] == "tool_result"
    current = context["evidence"]["current_evidence"]
    replacement = next(
        item for item in current if item["kind"] == "created_artifact_replacement"
    )
    assert replacement["paths"] == ["a.ts", "b.ts"]
