"""Behavior contracts for policy IFG source units and symbol bridging."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine

from policy_engine.ifg import (
    IFG_EXTRACTOR_VERSION,
    IfgToolEvent,
    extract_ifg_from_tool_events,
    persist_ifg_tool_events,
    rebuild_ifg_projection,
)
from policy_engine.ifg.schema import ensure_ifg_schema
from policy_engine.paths import default_policy_db_path, resolve_policy_db_path
from policy_engine.trace_view import PolicyTraceViewProvider, load_policy_trace_rows


def _event(
    *,
    turn: int,
    event_id: int,
    tool_call_id: str,
    tool_name: str,
    args: object,
    text: str,
    cwd: str = "/tmp/repo",
    session_id: str = "session-ifg",
) -> IfgToolEvent:
    return IfgToolEvent(
        session_id=session_id,
        turn=turn,
        event_id=event_id,
        tool_call_id=tool_call_id,
        phase="post",
        tool_name=tool_name,
        args=args,
        result={"content": [{"type": "text", "text": text}]},
        processed={},
        state={},
        cwd=cwd,
        ts=float(turn),
        source="test",
        raw_evidence={"tool_name": tool_name},
    )


def _source_events(session_id: str = "session-ifg") -> tuple[IfgToolEvent, ...]:
    return (
        _event(
            session_id=session_id,
            turn=1,
            event_id=1,
            tool_call_id="read-1",
            tool_name="read",
            args={"path": "src/app.py"},
            text="(3 lines total)\n1\timport os\n2\tdef foo():\n3\t    return os.getcwd()",
        ),
        _event(
            session_id=session_id,
            turn=2,
            event_id=2,
            tool_call_id="bash-1",
            tool_name="bash",
            args={"cmd": "rg foo src/app.py"},
            text="Stdout:\nsrc/app.py:1:def foo():\n",
        ),
        _event(
            session_id=session_id,
            turn=3,
            event_id=3,
            tool_call_id="edit-1",
            tool_name="edit",
            args={
                "path": "src/app.py",
                "old_string": "def foo():\n    return os.getcwd()",
                "new_string": "def foo():\n    return os.curdir",
            },
            text="Updated 'src/app.py':\n1\tdef foo():\n2\t    return os.curdir",
        ),
        _event(
            session_id=session_id,
            turn=4,
            event_id=4,
            tool_call_id="write-1",
            tool_name="write",
            args={
                "path": "src/util.py",
                "content": "class Util:\n    def method(self):\n        pass\n",
            },
            text="Created 'src/util.py' (45 bytes)",
        ),
    )


def test_ifg_source_units_bridge_tool_io_to_code_symbols() -> None:
    rows = extract_ifg_from_tool_events(_source_events())

    assert {action.action_kind for action in rows.actions} == {"read", "edit", "write"}
    assert {(unit.kind, unit.relation, unit.path) for unit in rows.source_units} == {
        ("read_result", "read", "/tmp/repo/src/app.py"),
        ("bash_segment", "read", None),
        ("bash_search_result", "supports", "/tmp/repo/src/app.py"),
        ("edit_old", "read", "/tmp/repo/src/app.py"),
        ("edit_new", "edit", "/tmp/repo/src/app.py"),
        ("edit_result_snippet", "edit", "/tmp/repo/src/app.py"),
        ("write_content", "write", "/tmp/repo/src/util.py"),
    }
    assert {edge.path for edge in rows.file_edges} == {
        "/tmp/repo/src/app.py",
        "/tmp/repo/src/util.py",
    }

    symbols = {
        (symbol.kind, symbol.qualified_name, symbol.path) for symbol in rows.symbols
    }
    assert ("module", "os", "/tmp/repo/src/app.py") in symbols
    assert ("function", "foo", "/tmp/repo/src/app.py") in symbols
    assert ("class", "Util", "/tmp/repo/src/util.py") in symbols
    assert ("method", "Util.method", "/tmp/repo/src/util.py") in symbols
    assert all(symbol.qualified_name not in {"def", "pass"} for symbol in rows.symbols)

    assert {mention.symbol_text for mention in rows.symbol_mentions} == {"foo"}
    foo_symbol = next(
        symbol
        for symbol in rows.symbols
        if symbol.kind == "function" and symbol.qualified_name == "foo"
    )
    mention_edges = [
        edge
        for edge in rows.action_symbol_edges
        if edge.symbol_id == foo_symbol.symbol_id
        and edge.metadata.get("resolution") == "path"
    ]
    assert mention_edges
    assert any(
        edge.to_node_id == f"node:{foo_symbol.symbol_id}"
        and edge.from_node_id.startswith("source:")
        for edge in rows.graph_edges
    )


def test_ifg_persistence_uses_source_unit_tables() -> None:
    engine = create_engine("sqlite:///:memory:")
    session_id = "session-persist"
    with engine.begin() as conn:
        result = persist_ifg_tool_events(
            conn,
            _source_events(session_id),
            session_id=session_id,
            update_summary=True,
        )

        assert result.source_units == 7
        assert result.symbol_mentions == 2
        assert result.unresolved_symbol_mentions == 0
        assert result.symbols >= 3

        table_counts = {
            table: conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar_one()
            for table in (
                "ifg_actions",
                "ifg_action_file_edges",
                "ifg_source_units",
                "ifg_symbol_mentions",
                "ifg_symbols",
                "ifg_action_symbol_edges",
                "ifg_file_symbol_edges",
                "ifg_nodes",
                "ifg_edges",
            )
        }
        assert table_counts["ifg_source_units"] == 7
        assert table_counts["ifg_symbol_mentions"] == 2
        assert table_counts["ifg_symbols"] >= 3
        assert table_counts["ifg_nodes"] > table_counts["ifg_actions"]

        summary_raw = conn.exec_driver_sql(
            "SELECT summary_json FROM ifg_session_summary WHERE session_id = ?",
            (session_id,),
        ).scalar_one()
    summary = json.loads(summary_raw)
    assert summary["source_units"] == 7
    assert summary["symbol_mentions"] == 2
    assert summary["unresolved_symbol_mentions"] == 0


def test_ifg_projection_can_be_deferred_until_session_shutdown() -> None:
    engine = create_engine("sqlite:///:memory:")
    session_id = "session-deferred"
    with engine.begin() as conn:
        persist_ifg_tool_events(
            conn,
            _source_events(session_id),
            session_id=session_id,
            rebuild_projection=False,
        )
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM ifg_actions").scalar_one()
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM ifg_nodes").scalar_one() == 0

        projection = rebuild_ifg_projection(conn, session_id)

        assert projection.graph_nodes
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM ifg_nodes").scalar_one()
        assert (
            conn.exec_driver_sql(
                "SELECT COUNT(*) FROM ifg_symbol_symbol_edges WHERE relation = 'imports'"
            ).scalar_one()
            == 0
        )

    engine.dispose()


def test_policy_trace_view_only_shows_current_ifg_model(tmp_path: Path) -> None:
    db_path = tmp_path / "policy.db"
    engine = create_engine(f"sqlite:///{db_path}")
    session_id = "session-trace-view"
    with engine.begin() as conn:
        persist_ifg_tool_events(
            conn,
            _source_events(session_id),
            session_id=session_id,
            extractor_version="legacy-ifg-v1",
            update_summary=True,
        )
        current = persist_ifg_tool_events(
            conn,
            _source_events(session_id),
            session_id=session_id,
            update_summary=True,
        )
    engine.dispose()

    category_counts = {
        "ifg_actions": current.actions,
        "ifg_source_units": current.source_units,
        "ifg_files": current.files,
        "ifg_symbols": current.symbols,
    }
    for category, expected_count in category_counts.items():
        rows = load_policy_trace_rows(
            db_path,
            session_id,
            categories=frozenset({category}),
        )
        assert len(rows) == expected_count
        assert {row.metadata.get("extractor_version") for row in rows} == {
            IFG_EXTRACTOR_VERSION
        }

    summary_rows = load_policy_trace_rows(
        db_path,
        session_id,
        categories=frozenset({"summary"}),
    )
    projection_counts = json.loads(summary_rows[0].content)["counts"]
    assert projection_counts["ifg_actions"] == current.actions
    assert projection_counts["ifg_source_units"] == current.source_units
    assert any(row.title == "IFG Model Summary" for row in summary_rows)


def test_policy_trace_view_exposes_only_primary_policy_and_ifg_tabs() -> None:
    specs = PolicyTraceViewProvider().trace_view_specs()
    assert [spec.id for spec in specs] == [
        "policy",
        "policy-effects",
        "policy-ifg-actions",
        "policy-ifg-source-units",
        "policy-ifg-files",
        "policy-ifg-symbols",
    ]
    assert [spec.title for spec in specs] == [
        "Policy",
        "Effects",
        "Actions",
        "Source Units",
        "Files",
        "Symbols",
    ]


def test_policy_db_resolution_finds_repo_run_by_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("AGENTM_HOME", raising=False)
    (tmp_path / "agentm.toml").write_text("[trajectory]\n", encoding="utf-8")
    unrelated = tmp_path / ".agentm" / "run-a" / "policy_state" / "policy.db"
    matching = tmp_path / ".agentm" / "run-b" / "policy_state" / "policy.db"
    for path, session_id in ((unrelated, "other"), (matching, "wanted")):
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE policy_tool_events (session_id TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO policy_tool_events (session_id) VALUES (?)",
                (session_id,),
            )

    nested_cwd = tmp_path / "src" / "package"
    nested_cwd.mkdir(parents=True)
    assert resolve_policy_db_path(session_id="wanted", cwd=nested_cwd) == matching


def test_policy_db_resolution_honors_explicit_agentm_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agentm_home = tmp_path / "explicit-home"
    monkeypatch.setenv("AGENTM_HOME", str(agentm_home))

    expected = agentm_home / "policy_state" / "policy.db"
    assert default_policy_db_path() == expected
    assert resolve_policy_db_path(session_id="any", cwd=tmp_path) == expected


def test_bash_directories_and_patterns_stay_unresolved_without_anchors() -> None:
    event = _event(
        turn=1,
        event_id=1,
        tool_call_id="bash-path-scopes",
        tool_name="bash",
        args={
            "cmd": (
                "find packages -maxdepth 2 -type f; "
                "ls node_modules/.pnpm; "
                "rg foo packages 'packages/*/src'"
            )
        },
        text="",
    )

    rows = extract_ifg_from_tool_events((event,))

    assert rows.file_edges == ()
    assert {(item.path_text, item.path_kind) for item in rows.path_candidates} == {
        ("packages", "directory"),
        ("node_modules/.pnpm", "directory"),
        ("packages/*/src", "pattern"),
    }


def test_later_read_anchor_reconnects_an_earlier_bash_candidate(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "policy.db"
    engine = create_engine(f"sqlite:///{db_path}")
    session_id = "session-late-anchor"
    bash_event = _event(
        session_id=session_id,
        turn=1,
        event_id=1,
        tool_call_id="bash-before-read",
        tool_name="bash",
        args={"cmd": "ls src/app.py"},
        text="",
    )
    read_event = _event(
        session_id=session_id,
        turn=2,
        event_id=2,
        tool_call_id="read-after-bash",
        tool_name="read",
        args={"path": "src/app.py"},
        text="(1 lines total)\n1\tdef foo(): pass",
    )

    with engine.begin() as conn:
        before = persist_ifg_tool_events(
            conn,
            (bash_event,),
            session_id=session_id,
        )
        assert before.files == 0
        assert before.path_candidates == 1
        assert before.unresolved_path_candidates == 1

        after = persist_ifg_tool_events(
            conn,
            (read_event,),
            session_id=session_id,
        )
        resolved = (
            conn.exec_driver_sql(
                """
            SELECT e.path, e.is_anchor, e.metadata_json
            FROM ifg_action_file_edges e
            JOIN ifg_actions a ON a.action_id = e.action_id
            WHERE e.session_id = ? AND a.tool_name = 'bash'
            """,
                (session_id,),
            )
            .mappings()
            .one()
        )

    engine.dispose()
    assert after.files == 1
    assert after.file_edges == 2
    assert after.unresolved_path_candidates == 0
    assert resolved["path"] == "/tmp/repo/src/app.py"
    assert resolved["is_anchor"] == 0
    assert json.loads(resolved["metadata_json"])["resolution"] == "trajectory_anchor"


def test_repository_existence_does_not_promote_bash_evidence(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "app.py"
    source_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    event = _event(
        turn=1,
        event_id=1,
        tool_call_id="bash-repository",
        tool_name="bash",
        args={"cmd": f"rg foo {source_path}"},
        text="",
        cwd=str(tmp_path),
    )

    rows = extract_ifg_from_tool_events((event,))

    assert rows.file_edges == ()
    assert rows.symbols == ()
    assert {candidate.path_text for candidate in rows.path_candidates} == {
        str(source_path)
    }
    assert {mention.symbol_text for mention in rows.symbol_mentions} == {"foo"}


def test_ifg_schema_adds_anchor_column_before_its_index() -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        ensure_ifg_schema(conn)
        conn.exec_driver_sql("DROP INDEX idx_ifg_action_file_edges_anchor")
        conn.exec_driver_sql("ALTER TABLE ifg_action_file_edges DROP COLUMN is_anchor")

        ensure_ifg_schema(conn)

        columns = {
            row[1]
            for row in conn.exec_driver_sql(
                "PRAGMA table_info(ifg_action_file_edges)"
            ).fetchall()
        }
        indexes = {
            row[1]
            for row in conn.exec_driver_sql(
                "PRAGMA index_list(ifg_action_file_edges)"
            ).fetchall()
        }
    engine.dispose()
    assert "is_anchor" in columns
    assert "idx_ifg_action_file_edges_anchor" in indexes
