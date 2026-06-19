from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

_SCENARIOS = Path(__file__).resolve().parents[3] / "contrib/scenarios"
if str(_SCENARIOS) not in sys.path:
    sys.path.insert(0, str(_SCENARIOS))

_fpg = importlib.import_module("verifier.lib.fpg")
_feedback = importlib.import_module("verifier.lib.finalize_feedback")


def test_http_display_config_fpg_injection_keeps_link_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "injection.json").write_text(json.dumps({
        "benchmark": "clickhouse",
        "display_config": json.dumps({
            "duration": 4,
            "namespace": "ts",
            "injection_point": {
                "app_name": "ts-preserve-service",
                "method": "POST",
                "route": "/api/v1/basicservice/basic/travel",
                "server_address": "ts-basic-service",
                "server_port": "8080",
            },
        }),
        "end_time": "2025-07-20T19:04:43Z",
        "engine_config": json.dumps({"name": "InjectionConf"}),
        "fault_type": "http_aborted",
        "injection_name": "ts1-ts-preserve-service-request-abort-rd9nzw",
        "start_time": "2025-07-20T19:00:44Z",
    }))

    meta = _fpg.load_injection_meta(tmp_path)
    records = _fpg.build_injection_records(meta)

    assert records == [
        {
            "node_id": "link:ts-preserve-service->ts-basic-service",
            "fault_type": "http_aborted",
            "target_entity": "link:ts-preserve-service->ts-basic-service",
            "parameters": {"target_service": "ts-basic-service"},
            "time": {
                "start": "2025-07-20T19:00:44Z",
                "end": "2025-07-20T19:04:43Z",
            },
            "replay_count": 0,
        }
    ]


def test_sql_validation_feedback_explains_zero_row_evidence() -> None:
    payload = _feedback.sql_validation_error_payload([
        {
            "location": "evidence[1]",
            "error": "0 rows",
            "sql": "SELECT * FROM abnormal_logs WHERE level = 'ERROR'",
        }
    ])

    [failure] = payload["failures"]
    assert payload["error"] == "sql_validation_failed"
    assert "0-row result is not accepted" in payload["actionable_hints"][1]
    assert "returned no rows" in failure["why"]
    assert "SELECT count(*) AS matching_rows" in failure["fix"]
