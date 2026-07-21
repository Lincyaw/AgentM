"""Policy DSL behavior over active IFG source regions."""

from __future__ import annotations

from pathlib import Path

from policy_engine.compiler import compile_policy_file, compile_rule
from policy_engine.evaluator import PolicyEvaluator
from policy_engine.ifg_interventions import IfgInterventionState
from policy_engine.ifg_regions import IfgQuery, IfgRegionState, parse_source_region
from policy_engine.state import PolicyState


def _result(start: int, end: int, *, content_hash: str = "same") -> dict[str, object]:
    lines = "\n".join(f"{line}\tline {line}" for line in range(start, end + 1))
    return {"text": lines, "content_hash": content_hash}


def _base_rule(name: str):
    policy_path = (
        Path(__file__).parents[1]
        / "contrib/extensions/policy/src/policy_engine/ifg_evidence.yaml"
    )
    rules, _disabled = compile_policy_file(policy_path.read_text(encoding="utf-8"))
    return next(rule for rule in rules if rule.rule_id == name)


def test_source_region_parser_is_shared_with_ifg_numbered_results() -> None:
    parsed = parse_source_region("header\n8\talpha\n9\tbeta")

    assert parsed is not None
    assert parsed.content_text == "alpha\nbeta"
    assert parsed.line_range == {
        "start_line": 8,
        "end_line": 9,
        "matched_lines": [8, 9],
        "partial": True,
    }


def test_edit_invalidates_only_its_region_and_write_invalidates_file() -> None:
    regions = IfgRegionState()
    query = IfgQuery(regions, IfgInterventionState()).region_reads
    path = "src/app.py"

    regions.record("read", {"path": path}, _result(1, 100), turn=1)
    regions.record("read", {"path": path}, _result(60, 65), turn=2)
    assert query.overlap_count(min_ratio=1.0, min_lines=6) == 1

    regions.record(
        "edit",
        {"path": path, "old_string": "old", "new_string": "new"},
        _result(60, 65, content_hash="edited"),
        turn=3,
    )
    regions.record(
        "read", {"path": path}, _result(1, 10, content_hash="edited"), turn=4
    )
    assert list(regions.entries())[-1].overlap_ratio == 1.0

    regions.record(
        "read", {"path": path}, _result(60, 65, content_hash="edited"), turn=4
    )
    assert list(regions.entries())[-1].overlap_lines == 0

    regions.record(
        "read", {"path": path}, _result(60, 65, content_hash="edited"), turn=5
    )
    assert list(regions.entries())[-1].overlap_ratio == 1.0

    regions.record(
        "write", {"path": path, "content": "replacement"}, {"text": "ok"}, turn=6
    )
    regions.record(
        "read", {"path": path}, _result(1, 100, content_hash="written"), turn=7
    )
    assert list(regions.entries())[-1].overlap_lines == 0


def test_changed_content_hash_invalidates_stale_read_coverage() -> None:
    regions = IfgRegionState()
    path = "src/app.py"

    regions.record("read", {"path": path}, _result(10, 20, content_hash="v1"), turn=1)
    regions.record("read", {"path": path}, _result(10, 20, content_hash="v2"), turn=2)

    assert list(regions.entries())[-1].overlap_lines == 0


def test_repeated_region_rule_fires_from_dsl_namespace() -> None:
    rule = compile_rule(
        {
            "name": "test-region-loop",
            "on": "tool_call_post",
            "match": {"tool": "read"},
            "when": (
                "ifg.region_reads.overlap_count(last=4, min_ratio=0.8, "
                "min_lines=10) >= 3"
            ),
            "effect": "escalate",
            "mode": "observe",
            "escalate_context": (
                "ifg.region_reads.summary(last=4, min_ratio=0.8, min_lines=10)"
            ),
            "reason": "Repeated source region reads.",
        }
    )
    state = PolicyState()
    evaluator = PolicyEvaluator([rule], state)

    for _turn in range(4):
        args = {"path": "src/app.py"}
        result = _result(1, 20)
        state.record_tool_call("read", args, result)
        evaluator.evaluate("tool_result", "read", args=args, result={"text": "ok"})

    effects = list(state.effect_log.entries())
    assert len(effects) == 1
    assert effects[0].rule_id == "test-region-loop"
    assert "repeated_reads: 3" in effects[0].reason


def test_base_policy_compiles_repeated_region_rule() -> None:
    policy_path = (
        Path(__file__).parents[1]
        / "contrib/extensions/policy/src/policy_engine/ifg_evidence.yaml"
    )
    rules, disabled = compile_policy_file(policy_path.read_text(encoding="utf-8"))

    assert not disabled
    rule_ids = {rule.rule_id for rule in rules}
    assert "repeated-region-reading" in rule_ids
    assert "exploration-not-converging" in rule_ids
    assert "mutation-target-drift" in rule_ids
    assert "unvalidated-intervention" in rule_ids
    assert "exploration-without-action" not in rule_ids
    assert "file-churn" not in rule_ids
