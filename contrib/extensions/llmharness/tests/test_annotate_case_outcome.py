"""Lock the case_outcome.json grading rule.

The downstream outcome reward + DPO pair builder reads
``case_outcome.json``; a regression in the substring-match rule silently
shifts the outcome reward signal. We keep the rule in lockstep with
``contrib/scenarios/rca/eval/baseline/grader.py`` (replicated, not
imported — the canonical grader is coupled to the on-disk observability
trace path).
"""

from __future__ import annotations

import json
from pathlib import Path

from llmharness.distill.cli import main as cli_main


def _make_bundle(
    tmp_path: Path,
    *,
    case_id: str,
    datapack_name: str,
    ground_truth: list[str],
    fault_type: str,
    submission: dict | None,
    observability_shape: bool = False,
) -> Path:
    bundle = tmp_path / case_id
    bundle.mkdir()
    (bundle / "case_metadata.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "datapack_name": datapack_name,
                "ground_truth": ground_truth,
                "fault_type": fault_type,
            }
        )
    )
    main_jsonl = bundle / "main.jsonl"
    rows: list[dict] = []
    if submission is not None:
        if observability_shape:
            rows.append(
                {
                    "kind": "event.dispatch",
                    "name": "emit:tool_call",
                    "attributes": {
                        "event": {
                            "tool_name": "submit_final_report",
                            "args": submission,
                        }
                    },
                }
            )
        else:
            rows.append(
                {
                    "type": "tool_call",
                    "payload": {
                        "tool_name": "submit_final_report",
                        "args": submission,
                    },
                }
            )
    main_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return bundle


def test_annotate_full_hit(tmp_path: Path) -> None:
    bundle = _make_bundle(
        tmp_path,
        case_id="case-7",
        datapack_name="dp-7",
        ground_truth=["checkoutservice"],
        fault_type="cpu_stress",
        submission={
            "root_causes": [
                {"service": "checkoutservice", "fault_kind": "cpu_stress"}
            ]
        },
    )
    rc = cli_main(["annotate-case-outcome", "--bundle", str(bundle)])
    assert rc == 0
    out = json.loads((bundle / "case_outcome.json").read_text())
    assert out["case_id"] == "case-7"
    assert out["datapack_name"] == "dp-7"
    assert out["service_hit"] == 1.0
    assert out["fault_kind_hit"] == 1.0
    assert abs(out["composite_score"] - 1.0) < 1e-9
    assert out["submission_seen"] is True


def test_annotate_service_only(tmp_path: Path) -> None:
    bundle = _make_bundle(
        tmp_path,
        case_id="case-svc",
        datapack_name="dp",
        ground_truth=["paymentservice"],
        fault_type="oom",
        submission={
            "root_causes": [
                {"service": "paymentservice-1", "fault_kind": "latency"}
            ]
        },
    )
    cli_main(["annotate-case-outcome", "--bundle", str(bundle)])
    out = json.loads((bundle / "case_outcome.json").read_text())
    assert out["service_hit"] == 1.0
    assert out["fault_kind_hit"] == 0.0
    assert abs(out["composite_score"] - 0.7) < 1e-9


def test_annotate_observability_shape(tmp_path: Path) -> None:
    bundle = _make_bundle(
        tmp_path,
        case_id="case-obs",
        datapack_name="dp",
        ground_truth=["cartservice"],
        fault_type="oom",
        submission={"root_causes": [{"service": "cartservice", "fault_kind": "oom"}]},
        observability_shape=True,
    )
    cli_main(["annotate-case-outcome", "--bundle", str(bundle)])
    out = json.loads((bundle / "case_outcome.json").read_text())
    assert out["service_hit"] == 1.0
    assert out["fault_kind_hit"] == 1.0


def test_annotate_no_submission(tmp_path: Path) -> None:
    bundle = _make_bundle(
        tmp_path,
        case_id="case-noop",
        datapack_name="dp",
        ground_truth=["x"],
        fault_type="y",
        submission=None,
    )
    cli_main(["annotate-case-outcome", "--bundle", str(bundle)])
    out = json.loads((bundle / "case_outcome.json").read_text())
    assert out["submission_seen"] is False
    assert out["composite_score"] == 0.0
