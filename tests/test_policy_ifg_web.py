"""Contracts for the live IFG web projection and HTTP boundary."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

from sqlalchemy import create_engine

from policy_engine.ifg import IfgToolEvent, persist_ifg_tool_events
from policy_engine.ifg.web import create_ifg_web_server, load_ifg_web_snapshot


def _event(
    *,
    call_id: str,
    tool: str,
    args: dict[str, object],
    result: str,
    timestamp: float,
) -> IfgToolEvent:
    return IfgToolEvent(
        session_id="web-session",
        turn=0,
        event_id=int(timestamp),
        tool_call_id=call_id,
        phase="post",
        tool_name=tool,
        args=args,
        result={"content": [{"type": "text", "text": result}]},
        processed={},
        state={},
        cwd="/repo",
        ts=timestamp,
        source="test",
        raw_evidence={"tool": tool},
    )


def _build_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "policy.db"
    engine = create_engine(f"sqlite:///{db_path}")
    events = (
        _event(
            call_id="read-1",
            tool="read",
            args={"path": "src/app.py"},
            result="1\tdef foo():\n2\t    return 1",
            timestamp=1.0,
        ),
        _event(
            call_id="bash-1",
            tool="bash",
            args={"cmd": "cd /repo && rg foo src/app.py | head"},
            result="src/app.py:1:def foo():",
            timestamp=2.0,
        ),
        _event(
            call_id="write-1",
            tool="write",
            args={"path": "src/new.py", "content": "def bar():\n    return 2\n"},
            result="Created src/new.py",
            timestamp=3.0,
        ),
    )
    with engine.begin() as connection:
        persist_ifg_tool_events(
            connection,
            events,
            session_id="web-session",
            update_summary=True,
        )
    engine.dispose()
    return db_path


def test_web_snapshot_collapses_evidence_into_three_lanes(tmp_path: Path) -> None:
    snapshot = load_ifg_web_snapshot(_build_db(tmp_path), "web-session")

    assert snapshot["stats"]["tools"] == {"bash": 1, "read": 1, "write": 1}
    assert {node["type"] for node in snapshot["nodes"]} == {
        "action",
        "file",
        "symbol",
    }
    assert {edge["kind"] for edge in snapshot["edges"]} >= {
        "action-file",
        "action-symbol",
        "symbol-file",
    }

    node_ids = {node["id"] for node in snapshot["nodes"]}
    assert all(
        edge["source"] in node_ids and edge["target"] in node_ids
        for edge in snapshot["edges"]
    )

    app_file = next(
        node
        for node in snapshot["nodes"]
        if node["type"] == "file" and node["path"] == "/repo/src/app.py"
    )
    app_component = next(
        component
        for component in snapshot["components"]
        if component["id"] == app_file["componentId"]
    )
    assert app_component["tools"] == {"bash": 1, "read": 1}
    assert app_component["symbols"] >= 1


def test_ifg_http_server_serves_snapshot_and_local_assets(tmp_path: Path) -> None:
    db_path = _build_db(tmp_path)
    server = create_ifg_web_server(db_path, "web-session", port=0, refresh_ms=750)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"
    try:
        with urlopen(f"{base_url}/api/graph", timeout=3) as response:  # noqa: S310
            payload = json.load(response)
        with urlopen(  # noqa: S310
            f"{base_url}/api/graph?revision={payload['revision']}",
            timeout=3,
        ) as response:
            unchanged_status = response.status
        with urlopen(f"{base_url}/", timeout=3) as response:  # noqa: S310
            html = response.read().decode()
        with urlopen(  # noqa: S310
            f"{base_url}/vendor/cytoscape.min.js",
            timeout=3,
        ) as response:
            vendor = response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert payload["sessionId"] == "web-session"
    assert payload["refreshMs"] == 750
    assert payload["nodes"]
    assert unchanged_status == 204
    assert not any(edge["kind"] == "symbol-symbol" for edge in payload["edges"])
    assert "<title>AgentM IFG</title>" in html
    assert len(vendor) > 400_000
    assert b"cytoscape" in vendor.lower()
