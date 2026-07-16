"""Verifier final output invariants must be total over valid graph shapes."""

from __future__ import annotations

from contrib.scenarios.verifier.lib.final_checks import run_final_checks


def test_confirmed_local_only_seed_is_not_reported_as_a_gap() -> None:
    report = run_final_checks(
        data_dir="/nonexistent",
        graph={},
        data_profile={},
        anomaly_inventory=[],
        entry_services={"frontend"},
        seeds={"database"},
        confirmed_seed_ids={"database"},
        nodes={"database": {}},
        adj={},
    )

    assert report["passed"] is True
    assert report["seed_reachability"] == {"database": []}
