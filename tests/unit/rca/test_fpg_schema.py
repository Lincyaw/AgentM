from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from rca import SCENARIO_ROOT
from rca.fpg_schema import (
    model_output_model,
    model_output_tool_schema,
    resolve_profile_path,
)
from agentm.extensions.validate import validate_atom_file


def _walk(value: Any) -> list[Any]:
    items = [value]
    if isinstance(value, dict):
        for child in value.values():
            items.extend(_walk(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(_walk(child))
    return items


def test_fpg_tool_schema_has_inlined_refs() -> None:
    profile = resolve_profile_path(None, scenario_dir=str(SCENARIO_ROOT))
    schema = model_output_tool_schema(profile)

    assert "$defs" not in schema
    assert not any(isinstance(item, dict) and "$ref" in item for item in _walk(schema))
    assert set(schema["required"]) == {"nodes", "root_causes"}


def test_profile_bound_model_accepts_fpg_output() -> None:
    profile = resolve_profile_path(None, scenario_dir=str(SCENARIO_ROOT))
    model = model_output_model(profile)
    now = datetime(2026, 6, 19, 12, tzinfo=UTC)

    output = model.model_validate(
        {
            "nodes": [
                {
                    "id": "n1",
                    "subject": "svc:checkout",
                    "predicate": "latency_degraded",
                    "time": {"start": now.isoformat(), "end": now.isoformat()},
                    "evidence": [
                        {
                            "query": {
                                "language": "sql",
                                "statement": "select 1 as observed_latency",
                            },
                            "explanation": "p99 latency rose during the incident",
                        }
                    ],
                }
            ],
            "edges": [],
            "root_causes": ["n1"],
        }
    )

    assert output.root_causes == ["n1"]


def test_profile_bound_model_accepts_link_root_cause() -> None:
    profile = resolve_profile_path(None, scenario_dir=str(SCENARIO_ROOT))
    model = model_output_model(profile)
    now = datetime(2026, 6, 19, 12, tzinfo=UTC)

    output = model.model_validate(
        {
            "nodes": [
                {
                    "id": "link:checkout->payment",
                    "subject": "link:checkout->payment",
                    "predicate": "network_degraded",
                    "time": {"start": now.isoformat(), "end": now.isoformat()},
                    "evidence": [
                        {
                            "query": {
                                "language": "sql",
                                "statement": "select 1 as link_latency_delta",
                            },
                            "explanation": "caller-side latency rose on the payment path",
                        }
                    ],
                },
                {
                    "id": "checkout",
                    "subject": "svc:checkout",
                    "predicate": "latency_degraded",
                    "time": {"start": now.isoformat(), "end": now.isoformat()},
                    "evidence": [
                        {
                            "query": {
                                "language": "sql",
                                "statement": "select 1 as checkout_latency_delta",
                            },
                            "explanation": "checkout slowed while waiting on payment",
                        }
                    ],
                },
            ],
            "edges": [
                {
                    "src": "link:checkout->payment",
                    "dst": "checkout",
                    "mechanism": "network_path_effect",
                    "verification": "consistency-checked",
                }
            ],
            "root_causes": ["link:checkout->payment"],
        }
    )

    assert output.root_causes == ["link:checkout->payment"]


def test_profile_bound_model_rejects_old_rcabench_output() -> None:
    profile = resolve_profile_path(None, scenario_dir=str(SCENARIO_ROOT))
    model = model_output_model(profile)

    with pytest.raises(Exception):
        model.model_validate(
            {
                "root_causes": [
                    {
                        "service": "checkout",
                        "fault_kind": "http_slow",
                        "evidence": [
                            {
                                "kind": "trace",
                                "sql": "select 1",
                                "claim": "checkout is slow",
                            }
                        ],
                    }
                ],
                "propagation": [],
            }
        )


def test_fpg_rca_atoms_satisfy_section_11_contract() -> None:
    atom_paths = [
        SCENARIO_ROOT / "src/rca/default/finalize.py",
        SCENARIO_ROOT / "src/rca/default/fpg_contract.py",
        SCENARIO_ROOT / "src/rca/default/rcabench_contract.py",
    ]

    issues = []
    for path in atom_paths:
        issues.extend(
            validate_atom_file(
                path,
                module_path=f"rca.default.{path.stem}",
                known_extension_names={"finalize", "fpg_contract", "rcabench_contract"},
            )
        )

    assert issues == [], "\n".join(
        f"{issue.module_path} [{issue.rule}]: {issue.message}" for issue in issues
    )
