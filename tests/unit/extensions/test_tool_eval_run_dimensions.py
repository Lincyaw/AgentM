from __future__ import annotations

from agentm.extensions.builtin.tool_eval_run import (
    _dimension_means,
    _normalize_grade,
)


def test_normalize_grade_preserves_numeric_dimensions() -> None:
    grade = _normalize_grade(
        {
            "score": 0.7,
            "dimensions": {
                "fpg_sql_executable_ratio": 1.0,
                "bad": "not-a-number",
            },
            "feedback_text": "ok",
            "module_feedback": {"evidence_sql": "all executable"},
            "failure_kind": "correctness",
        }
    )

    assert grade["dimensions"] == {"fpg_sql_executable_ratio": 1.0}
    assert grade["module_feedback"] == {"evidence_sql": "all executable"}


def test_dimension_means_are_stable_for_eval_task_records() -> None:
    means = _dimension_means(
        {
            "fpg_sql_all_executable": [1.0, 0.0],
            "fpg_sql_executable_ratio": [1.0, 0.5],
            "empty": [],
        }
    )

    assert means == {
        "fpg_sql_all_executable": 0.5,
        "fpg_sql_executable_ratio": 0.75,
    }
