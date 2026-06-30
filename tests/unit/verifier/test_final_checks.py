from __future__ import annotations

import json

from contrib.scenarios.verifier.lib.final_checks import run_final_checks


def test_causal_graph_upstream_overrides_endpoint_name_prior(tmp_path):
    (tmp_path / "causal_graph.json").write_text(
        json.dumps(
            {
                "component_to_service": {
                    "span|profile::profile.Profile/GetProfiles": "profile",
                    "span|frontend::profile.Profile/GetProfiles": "frontend",
                    "span|frontend::HTTP /recommendations": "frontend",
                },
                "edges": [
                    {
                        "source": "span|profile::profile.Profile/GetProfiles",
                        "target": "span|frontend::profile.Profile/GetProfiles",
                    },
                    {
                        "source": "span|frontend::profile.Profile/GetProfiles",
                        "target": "span|frontend::HTTP /recommendations",
                    },
                ],
                "path_terminal_alarm_nodes": [
                    {
                        "component": "span|frontend::HTTP /recommendations",
                        "state": ["erroring"],
                    }
                ],
            }
        )
    )

    report = run_final_checks(
        data_dir=str(tmp_path),
        graph={
            "profile": [["frontend"]],
            "recommendation": [["frontend"]],
        },
        data_profile={
            "structure": {
                "services": ["frontend", "profile", "recommendation"],
            }
        },
        anomaly_inventory=[],
        entry_services={"frontend"},
        seeds={"profile"},
        confirmed_seed_ids={"profile"},
        nodes={"profile": {}, "frontend": {}},
        adj={"profile": ["frontend"]},
    )

    assert report["passed"] is True
    anomaly = report["frontend_anomalies"][0]
    assert anomaly["causal_upstream_services"] == ["profile"]
