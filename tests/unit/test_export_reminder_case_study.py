from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from scripts import export_reminder_case_study


def test_effect_marks_secondary_metric_regression_as_harmed_partial() -> None:
    baseline = export_reminder_case_study.JudgeResult(
        correct=False,
        detail={"f1": 0.0, "any_service_hit": True, "fault_kind_accuracy": 0.0},
    )
    fork = export_reminder_case_study.JudgeResult(
        correct=False,
        detail={"f1": 0.0, "any_service_hit": False, "fault_kind_accuracy": None},
    )

    assert export_reminder_case_study._effect(baseline, fork) == "harmed_partial"


def test_final_payload_reads_tool_args() -> None:
    payload = {
        "root_causes": [
            {"service": "ts-route-plan-service", "fault_kind": "http_slow"}
        ]
    }

    assert (
        export_reminder_case_study._final_payload(
            [{"args": payload}], session_id="session-1"
        )
        == payload
    )


def test_discover_variants_uses_trace_index_and_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_trace_index(
        agentm_cmd: list[str],
        *,
        children_of: str,
    ) -> list[dict[str, str]]:
        assert agentm_cmd == ["agentm"]
        assert children_of == "root-1"
        return [
            {"session_id": "child-a"},
            {"session_id": "child-b"},
            {"session_id": "child-c"},
            {"session_id": "child-d"},
        ]

    def fake_session_info(agentm_cmd: list[str], session_id: str) -> dict[str, Any]:
        assert agentm_cmd == ["agentm"]
        return {
            "child-a": {
                "header": {
                    "config": {
                        "lineage": {
                            "kind": "fork",
                            "source_session_id": "root-1",
                        },
                        "experiment": {"variant": "named-reminder"},
                    }
                }
            },
            "child-b": {
                "header": {
                    "config": {
                        "lineage": {
                            "kind": "fork",
                            "source_session_id": "root-1",
                        },
                    }
                }
            },
            "child-c": {
                "header": {
                    "config": {
                        "lineage": {
                            "kind": "sub_agent",
                            "source_session_id": "root-1",
                        }
                    }
                }
            },
            "child-d": {
                "header": {
                    "config": {
                        "lineage": {
                            "kind": "fork",
                            "source_session_id": "other-root",
                        }
                    }
                }
            },
        }[session_id]

    monkeypatch.setattr(
        export_reminder_case_study,
        "_run_trace_index",
        fake_run_trace_index,
    )
    monkeypatch.setattr(
        export_reminder_case_study,
        "_session_info",
        fake_session_info,
    )

    assert export_reminder_case_study._discover_variants(["agentm"], "root-1") == [
        export_reminder_case_study.Variant(
            name="named-reminder",
            session_id="child-a",
        ),
        export_reminder_case_study.Variant(
            name="fork-child-b",
            session_id="child-b",
        ),
    ]


def test_typer_cli_help_exposes_variant_option() -> None:
    result = CliRunner().invoke(export_reminder_case_study.app, ["--help"])

    assert result.exit_code == 0
    assert "--variant" in result.stdout
    assert "COMMAND" not in result.stdout
