"""Verifier v2 anomaly attribution must stay service-local."""

from __future__ import annotations

from contrib.scenarios.verifier_v2.gaps import evaluate_gaps
from contrib.scenarios.verifier_v2.state import Case, GraphState


def test_unrelated_service_endpoint_cannot_explain_frontend_anomaly() -> None:
    case = Case(
        injections=[],
        graph={},
        infra_set=set(),
        entry_services={"frontend"},
        seeds=set(),
        data_dir="/nonexistent",
        fault_docs={},
        data_profile={},
        anomaly_inventory=[
            {
                "id": "checkout-latency",
                "status": "changed",
                "subject": "svc:frontend",
                "component": "/checkout",
            }
        ],
        window={},
    )
    state = GraphState(case=case, log=lambda _message: None)
    state.nodes["unrelated-backend"] = {
        "kind": "hop",
        "affected_endpoints": ["/checkout"],
    }

    report = evaluate_gaps(case, state)

    assert [gap.id for gap in report.gaps] == ["unexplained_anomaly:checkout-latency"]
