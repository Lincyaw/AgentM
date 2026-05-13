"""Smoke test for ``llmharness.serve`` — read-only HTTP API."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from llmharness.serve import make_server


@pytest.fixture
def cases_root(tmp_path: Path) -> Path:
    """Build a minimal cases/ directory with one fully-populated case."""
    case = tmp_path / "case-alpha"
    (case / "extractor").mkdir(parents=True)
    (case / "auditor").mkdir(parents=True)
    (case / "event_graph").mkdir(parents=True)

    (case / "meta.json").write_text(
        json.dumps(
            {
                "case_id": "case-alpha",
                "extractor_firings": 1,
                "auditor_firings": 1,
                "surfaced_reminders": 0,
                "silent_verdicts": 1,
            }
        ),
        encoding="utf-8",
    )
    (case / "main_agent.jsonl").write_text(
        '{"role":"user","content":"hi"}\n{"role":"assistant","content":"hello"}\n',
        encoding="utf-8",
    )
    (case / "verdicts.jsonl").write_text('{"sequence":1,"surface_reminder":false}\n', encoding="utf-8")
    (case / "trajectory.jsonl").write_text('{"turn":1}\n', encoding="utf-8")
    (case / "extractor" / "001_turn_001.json").write_text(
        '{"status":"ok","latency_ms":10}', encoding="utf-8"
    )
    (case / "auditor" / "001_turn_005.json").write_text(
        '{"status":"ok","latency_ms":20}', encoding="utf-8"
    )
    (case / "event_graph" / "after_extractor_001.json").write_text(
        '{"events":[],"edges":[]}', encoding="utf-8"
    )

    # A non-case directory must be ignored by /api/cases.
    (tmp_path / "scratch").mkdir()
    return tmp_path


@pytest.fixture
def server(cases_root: Path):
    srv = make_server(cases_root, host="127.0.0.1", port=0, allow_origin="*")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    host, port = srv.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        yield base
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2)


def _get(base: str, path: str) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(f"{base}{path}")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, dict(r.headers), r.read()


def _get_json(base: str, path: str):
    status, headers, body = _get(base, path)
    assert status == 200, (status, body)
    assert "application/json" in headers.get("Content-Type", "")
    return json.loads(body)


def test_health(server: str, cases_root: Path) -> None:
    payload = _get_json(server, "/api/health")
    assert payload["case_count"] == 1
    assert Path(payload["root"]) == cases_root.resolve()


def test_list_cases(server: str) -> None:
    payload = _get_json(server, "/api/cases")
    assert [c["case_id"] for c in payload["cases"]] == ["case-alpha"]
    assert payload["cases"][0]["meta"]["extractor_firings"] == 1


def test_case_files(server: str) -> None:
    assert _get_json(server, "/api/cases/case-alpha/meta")["case_id"] == "case-alpha"

    status, headers, body = _get(server, "/api/cases/case-alpha/main_agent")
    assert status == 200
    assert "ndjson" in headers["Content-Type"]
    assert body.decode().count("\n") == 2


def test_firings_index_and_file(server: str) -> None:
    files = _get_json(server, "/api/cases/case-alpha/firings/extractor")["files"]
    assert files == ["001_turn_001.json"]

    firing = _get_json(server, "/api/cases/case-alpha/firings/extractor/001_turn_001.json")
    assert firing["status"] == "ok"


def test_snapshot_present_and_missing(server: str) -> None:
    snap = _get_json(server, "/api/cases/case-alpha/snapshots/1")
    assert snap == {"events": [], "edges": []}

    req = urllib.request.Request(f"{server}/api/cases/case-alpha/snapshots/99")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 404


def test_cors_preflight(server: str) -> None:
    req = urllib.request.Request(f"{server}/api/cases", method="OPTIONS")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 204
        assert r.headers["Access-Control-Allow-Origin"] == "*"


def test_path_traversal_rejected(server: str) -> None:
    req = urllib.request.Request(f"{server}/api/cases/..%2Fetc/meta")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code in (400, 404)


def test_unknown_case_is_404(server: str) -> None:
    req = urllib.request.Request(f"{server}/api/cases/does-not-exist/meta")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 404
