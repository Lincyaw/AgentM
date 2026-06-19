from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SCENARIOS = Path(__file__).resolve().parents[3] / "contrib/scenarios"
if str(_SCENARIOS) not in sys.path:
    sys.path.insert(0, str(_SCENARIOS))

_injection = importlib.import_module("verifier.lib.injection")
_fpg = importlib.import_module("verifier.lib.fpg")


def test_network_peer_injection_is_reified_as_link_entity() -> None:
    entry = {
        "app": "search",
        "chaos_type": "NetworkDelay",
        "direction": "to",
        "latency": 1490,
        "target_service": "rate",
        "namespace": "hs1",
        "duration": 5,
    }

    normalized = _injection.enrich_injection_entry(entry)

    assert normalized["target"] == "search"
    assert normalized["node_id"] == "link:search->rate"
    assert normalized["target_entity"] == "link:search->rate"
    assert normalized["effect_target"] == "search"
    assert normalized["params"] == "direction=to, latency=1490, target_service=rate"


def test_fpg_injection_records_keep_link_target_entity() -> None:
    entry = {
        "app": "search",
        "chaos_type": "NetworkPartition",
        "direction": "both",
        "target_service": "rate",
        "namespace": "hs1",
        "duration": 5,
    }
    meta = {
        "window": {
            "start": "2026-06-19T12:00:00+00:00",
            "end": "2026-06-19T12:05:00+00:00",
        },
        "engine": [entry],
    }

    records = _fpg.build_injection_records(meta)

    assert records == [
        {
            "node_id": "link:search->rate",
            "fault_type": "NetworkPartition",
            "target_entity": "link:search->rate",
            "parameters": {
                "direction": "both",
                "target_service": "rate",
            },
            "time": meta["window"],
            "replay_count": 0,
        }
    ]


def test_http_peer_injection_is_reified_as_link_entity() -> None:
    entry = {
        "app": "checkout",
        "chaos_type": "HTTPResponseStatusModified",
        "method": "POST",
        "path": "/pay",
        "target_service": "payment",
    }

    normalized = _injection.enrich_injection_entry(entry)

    assert normalized["target"] == "checkout"
    assert normalized["node_id"] == "link:checkout->payment"
    assert normalized["target_entity"] == "link:checkout->payment"
    assert normalized["effect_target"] == "checkout"
    assert normalized["params"] == (
        "method=POST, path=/pay, target_service=payment"
    )
