from __future__ import annotations

import json

from sqlalchemy.engine import Connection

from agentm.presenter.trajectory import TraceMetrics, TraceQuery, TraceSnapshot
from agentm.storage.sql import create_sqlite_engine
from agentm.core.lib.tokens import count_text_tokens
from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
)
from agentm.core.abi.termination import ModelEndTurn
from agentm.core.abi.trajectory import Outcome, Round, ToolRecord, Turn
from agentm.core.abi.trigger import UserInput
from policy_engine import _content_stats
from policy_engine.compiler import compile_policy_file
from policy_engine.persistence import PolicyPersistence
from policy_engine.projection import events_from_turns, project_events
from policy_engine.trace_view import (
    build_policy_trace_view_registry,
    load_policy_trace_rows,
)
from policy_engine.types import (
    EffectRecord,
    EntityRecord,
    Evidence,
    FileStateEntry,
    ToolLogEntry,
)


def _count(conn: Connection, table: str) -> int:
    row = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def test_context_content_stats_count_tokens_for_text_and_tool_calls() -> None:
    tool_call = ToolCallBlock(
        type="tool_call",
        id="call-1",
        name="bash",
        arguments={"cmd": "echo hi"},
    )
    tool_result = ToolResultBlock(
        type="tool_result",
        tool_call_id="call-1",
        content=(TextContent(type="text", text="ok"),),
    )

    chars, tokens, text_blocks, image_blocks = _content_stats(
        (
            TextContent(type="text", text="hello world"),
            tool_call,
            tool_result,
        ),
        model_name=None,
    )

    rendered_call = json.dumps(
        {"arguments": {"cmd": "echo hi"}, "name": "bash"},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    assert chars == len("hello world") + len(rendered_call) + len("ok")
    assert tokens == (
        count_text_tokens("hello world")
        + count_text_tokens(rendered_call)
        + count_text_tokens("ok")
    )
    assert text_blocks == 2
    assert image_blocks == 0


def test_policy_projection_queries_missing_state_and_effect_log_records(
    tmp_path,
) -> None:
    policy = """
version: 1
rules:
  - name: slot-backed-effect
    on: tool_call_pre
    match: {tool: bash}
    when: "event.args.get('cmd', '') == 'echo seed'"
    effect: notify
    mode: observe
    reason: "seed effect"
  - name: effect-log-query-fired
    on: tool_call_pre
    match: {tool: bash}
    when: >
      event.args.get('cmd', '') == 'echo followup'
      and effect_log.exists({'rule_id': 'slot-backed-effect'})
    effect: notify
    mode: observe
    reason: "effect log query worked"
  - name: missing-context-should-not-fire
    on: tool_call_pre
    match: {tool: bash}
    when: >
      lookup('context_state', session.turn_count, 'context_usage_pct') > 80
    effect: notify
    mode: observe
    reason: "missing context should not compare truthy"
"""
    rules, _disabled = compile_policy_file(policy)
    turns = (
        _turn_with_tool(
            turn_index=0,
            tool_name="bash",
            args={"cmd": "echo seed"},
            result_text="seeded",
        ),
        _turn_with_tool(
            turn_index=1,
            tool_name="bash",
            args={"cmd": "echo followup"},
            result_text="followed",
        ),
    )

    db_path = tmp_path / "policy.db"
    persistence = PolicyPersistence(db_path)
    persistence.open()
    try:
        result = project_events(
            session_id="session-query",
            events=events_from_turns(turns),
            rules=rules,
            persistence=persistence,
        )
    finally:
        persistence.close()

    assert result.effects == 2
    rows = load_policy_trace_rows(
        db_path,
        "session-query",
        categories=frozenset({"effects"}),
    )
    assert {row.title for row in rows} == {
        "slot-backed-effect",
        "effect-log-query-fired",
    }


def test_policy_persistence_records_intermediate_state(tmp_path) -> None:
    db_path = tmp_path / "policy.db"
    persistence = PolicyPersistence(db_path)
    persistence.open()

    file_entry = FileStateEntry(
        path="/tmp/app.py",
        content_hash="abc123",
        first_read_turn=0,
        last_read_turn=1,
        last_write_turn=1,
        read_count=2,
        write_count=1,
        reverts_to_prior_hash=True,
    )
    entity = EntityRecord(
        entity="/tmp/app.py",
        entity_type="path",
        first_seen_turn=0,
        last_seen_turn=1,
        occurrence_count=2,
    )
    entity.evidence.records.append(
        Evidence(type="structural", turn=0, detail="path argument")
    )
    tool_entry = ToolLogEntry(
        turn=1,
        tool="bash",
        args_hash="deadbeef",
        cmd="make test",
        exit_code=1,
        error="x" * 3000,
        error_fingerprint="fp-1",
        error_category="runtime",
        duration_ms=321,
        result_length=123,
        is_repeat=True,
        repeat_count=1,
    )
    effect = EffectRecord(
        rule_id="stuck-loop",
        mode="observe",
        channel="tool_result",
        effect="notify",
        reason="diagnostic",
        turn=1,
    )

    persistence.queue_tool_event(
        session_id="session-1",
        turn=1,
        phase="pre",
        tool_call_id="call-1",
        tool_name="bash",
        args={
            "cmd": "echo hi",
            "api_key": "secret-value",
            "items": list(range(105)),
        },
        taint_labels=("secret",),
    )
    persistence.queue_tool_event(
        session_id="session-1",
        turn=1,
        phase="post",
        tool_call_id="call-1",
        tool_name="bash",
        args={"cmd": "make test"},
        entry=tool_entry,
        result={"is_error": True, "error": "boom"},
        processed={
            "is_error": True,
            "error": "boom",
            "result_content_hash": "result-hash-1",
            "duration_ms": 321,
        },
    )
    persistence.queue_file_state_snapshot("session-1", (file_entry,))
    persistence.queue_entity_state_snapshot("session-1", (entity,))
    persistence.queue_context_state(
        "session-1",
        1,
        {"total_context_tokens": 42, "access_token": "raw-token"},
    )
    persistence.queue_turn_summary(
        "session-1",
        1,
        {"tool_calls_count": 2, "tool_names_set": {"bash", "read"}},
    )
    persistence.queue_session_summary(
        "session-1",
        {"turn_count": 1, "tool_call_count": 1},
    )
    persistence.queue_eval_error(
        session_id="session-1",
        turn=1,
        rule_id="bad-rule",
        channel="tool_call",
        tool_name="bash",
        error="NameError: missing",
    )
    persistence.queue_effect("session-1", effect)
    persistence.flush()
    persistence.close()

    engine = create_sqlite_engine(db_path)
    try:
        with engine.connect() as conn:
            assert _count(conn, "event_log") == 1
            assert _count(conn, "policy_tool_events") == 2
            assert _count(conn, "policy_file_state") == 1
            assert _count(conn, "policy_entity_state") == 1
            assert _count(conn, "policy_context_state") == 1
            assert _count(conn, "policy_turn_summary") == 1
            assert _count(conn, "policy_session_summary") == 1
            assert _count(conn, "policy_eval_error") == 1

            pre = (
                conn.exec_driver_sql(
                    "SELECT args_json, state_json FROM policy_tool_events WHERE phase = 'pre'"
                )
                .mappings()
                .fetchone()
            )
            assert pre is not None
            pre_args = json.loads(pre["args_json"])
            assert pre_args["api_key"] == "secret-value"
            assert len(pre_args["items"]) == 105
            assert json.loads(pre["state_json"])["taint_labels"] == ["secret"]

            post = (
                conn.exec_driver_sql(
                    """
                SELECT state_json, result_json, processed_json, exit_code, duration_ms,
                       result_content_hash
                FROM policy_tool_events WHERE phase = 'post'
                """
                )
                .mappings()
                .fetchone()
            )
            assert post is not None
            post_state = json.loads(post["state_json"])
            assert post_state["tool_log_entry"]["error_fingerprint"] == "fp-1"
            assert len(post_state["tool_log_entry"]["error"]) == 3000
            assert json.loads(post["result_json"])["is_error"] is True
            post_processed = json.loads(post["processed_json"])
            assert post_processed["result_content_hash"] == "result-hash-1"
            assert post["exit_code"] == 1
            assert post["duration_ms"] == 321
            assert post["result_content_hash"] == "result-hash-1"

            context = (
                conn.exec_driver_sql("SELECT context_json FROM policy_context_state")
                .mappings()
                .fetchone()
            )
            assert context is not None
            assert json.loads(context["context_json"])["access_token"] == "raw-token"
    finally:
        engine.dispose()


def test_policy_trace_file_history_and_stream_from_tool_events(tmp_path) -> None:
    db_path = tmp_path / "policy.db"
    persistence = PolicyPersistence(db_path)
    persistence.open()
    try:
        persistence.queue_tool_event(
            session_id="session-file",
            turn=0,
            phase="pre",
            tool_call_id="read-1",
            tool_name="read",
            args={"path": "/tmp/app.py"},
        )
        persistence.queue_tool_event(
            session_id="session-file",
            turn=0,
            phase="post",
            tool_call_id="read-1",
            tool_name="read",
            args={"path": "/tmp/app.py"},
            result={"is_error": False, "text": "print('hi')"},
            processed={"is_error": False, "text_length": 11},
        )
        persistence.queue_tool_event(
            session_id="session-file",
            turn=1,
            phase="pre",
            tool_call_id="edit-1",
            tool_name="edit",
            args={"path": "/tmp/app.py"},
        )
        persistence.queue_tool_event(
            session_id="session-file",
            turn=1,
            phase="post",
            tool_call_id="edit-1",
            tool_name="edit",
            args={"path": "/tmp/app.py"},
            result={"is_error": False},
            processed={
                "is_error": False,
                "content_hash": "hash-after",
                "previous_content_hash": "hash-before",
            },
        )
        persistence.queue_tool_event(
            session_id="session-file",
            turn=2,
            phase="post",
            tool_call_id="glob-1",
            tool_name="glob",
            args={"path": "/tmp/generated.list"},
            result={"is_error": False},
            processed={"is_error": False},
        )
        persistence.queue_tool_event(
            session_id="session-file",
            turn=3,
            phase="post",
            tool_call_id="bash-1",
            tool_name="bash",
            args={"cmd": "cat package.json"},
            result={"is_error": False},
            processed={"is_error": False},
        )
        persistence.flush()
    finally:
        persistence.close()

    stream_rows = load_policy_trace_rows(
        db_path,
        "session-file",
        categories=frozenset({"file_stream"}),
    )
    assert [(row.title, row.cause, row.metadata["phase"]) for row in stream_rows] == [
        ("/tmp/app.py", "ok:-", "pre+post"),
        ("/tmp/app.py", "ok:-", "pre+post"),
        ("/tmp/generated.list", "ok:-", "post"),
    ]
    assert [row.metadata["operation"] for row in stream_rows] == [
        "read",
        "write",
        "read",
    ]
    assert [row.metadata["result"] for row in stream_rows] == ["ok:-", "ok:-", "ok:-"]
    assert [row.metadata["file_result"] for row in stream_rows] == [
        "ok:-",
        "ok:-",
        "ok:-",
    ]
    assert [row.metadata["tool_result"] for row in stream_rows] == [
        "ok:-",
        "ok:-",
        "ok:-",
    ]
    assert [row.display_name for row in stream_rows] == [
        "/tmp/app.py",
        "/tmp/app.py",
        "/tmp/generated.list",
    ]
    assert all(row.metadata["category"] == "file_stream" for row in stream_rows)
    assert stream_rows[2].metadata["source"] == "args.path"
    assert [row.metadata["evidence"] for row in stream_rows] == [
        "structured",
        "structured",
        "structured",
    ]

    file_tool_rows = load_policy_trace_rows(
        db_path,
        "session-file",
        categories=frozenset({"file_tool_stream"}),
    )
    assert [(row.tool_name, row.title) for row in file_tool_rows] == [
        ("read", "/tmp/app.py"),
        ("edit", "/tmp/app.py"),
    ]
    assert all(row.metadata["category"] == "file_tool_stream" for row in file_tool_rows)

    bash_rows = load_policy_trace_rows(
        db_path,
        "session-file",
        categories=frozenset({"bash_file_stream"}),
    )
    assert [(row.title, row.metadata["operation"]) for row in bash_rows] == [
        ("package.json", "read")
    ]
    assert bash_rows[0].metadata["category"] == "bash_file_stream"
    assert bash_rows[0].metadata["evidence"] == "shell-op"
    assert bash_rows[0].metadata["source"] == "cmd.reader"

    file_rows = load_policy_trace_rows(
        db_path,
        "session-file",
        categories=frozenset({"files"}),
    )
    app_row = next(row for row in file_rows if row.title == "/tmp/app.py")
    app_detail = json.loads(app_row.content)
    assert app_row.display_name is None
    assert app_row.cause == "r:1 w:1 ref:0 ev:2"
    assert app_row.metadata["reads"] == 1
    assert app_row.metadata["writes"] == 1
    assert app_row.metadata["events"] == 2
    assert [event["operation"] for event in app_detail["history"]] == ["read", "write"]
    assert [event["phase"] for event in app_detail["history"]] == [
        "pre+post",
        "pre+post",
    ]
    assert app_detail["history"][1]["content_hash"] == "hash-after"

    snapshot = TraceSnapshot(
        session_id="session-file",
        turns=(),
        rows=(),
        metrics=TraceMetrics(),
    )
    specs = build_policy_trace_view_registry(db_path).specs()
    file_view = next(spec for spec in specs if spec.id == "policy-files").build(
        snapshot,
        TraceQuery(),
    )
    stream_view = next(spec for spec in specs if spec.id == "policy-file-stream").build(
        snapshot,
        TraceQuery(),
    )
    file_tool_view = next(
        spec for spec in specs if spec.id == "policy-file-tool-stream"
    ).build(
        snapshot,
        TraceQuery(),
    )
    bash_stream_view = next(
        spec for spec in specs if spec.id == "policy-bash-file-stream"
    ).build(
        snapshot,
        TraceQuery(),
    )
    assert any(row.title == "/tmp/app.py" for row in file_view.rows)
    assert [row.title for row in stream_view.rows] == [
        "/tmp/app.py",
        "/tmp/app.py",
        "/tmp/generated.list",
    ]
    assert [row.title for row in file_tool_view.rows] == [
        "/tmp/app.py",
        "/tmp/app.py",
    ]
    assert [row.title for row in bash_stream_view.rows] == ["package.json"]


def test_bash_file_stream_ignores_heredoc_source_tokens(tmp_path) -> None:
    db_path = tmp_path / "policy.db"
    persistence = PolicyPersistence(db_path)
    persistence.open()
    try:
        persistence.queue_tool_event(
            session_id="session-bash",
            turn=0,
            phase="post",
            tool_call_id="bash-heredoc",
            tool_name="bash",
            args={
                "cmd": (
                    "cat > /tmp/repro.any << 'EOF'\n"
                    'import x from "better-auth/plugins/jwt";\n'
                    'fetch("/oauth2/token", { headers: { "content-type": '
                    '"application/json" } });\n'
                    "EOF\n"
                    "cat package.json 2>/dev/null"
                )
            },
            result={"is_error": False},
            processed={"is_error": False},
        )
        persistence.queue_tool_event(
            session_id="session-bash",
            turn=1,
            phase="post",
            tool_call_id="bash-error",
            tool_name="bash",
            args={"cmd": "cat README.md"},
            result={"is_error": True},
            processed={
                "is_error": True,
                "exit_code": 7,
                "error_category": "runtime",
            },
        )
        persistence.flush()
    finally:
        persistence.close()

    stream_rows = load_policy_trace_rows(
        db_path,
        "session-bash",
        categories=frozenset({"bash_file_stream"}),
    )

    assert [
        (
            row.title,
            row.metadata["operation"],
            row.metadata["file_result"],
            row.metadata["evidence"],
            row.metadata["source"],
        )
        for row in stream_rows
    ] == [
        ("/tmp/repro.any", "write", "observed", "shell-op", "cmd.redirect.write"),
        ("package.json", "read", "observed", "shell-op", "cmd.reader"),
        ("README.md", "read", "observed", "shell-op", "cmd.reader"),
    ]
    assert {row.metadata["category"] for row in stream_rows} == {"bash_file_stream"}
    assert stream_rows[2].metadata["tool_result"] == "exit:7 runtime"
    assert stream_rows[2].metadata["tool_exit_code"] == 7
    assert stream_rows[2].metadata["command"] == "cat README.md"
    structured_rows = load_policy_trace_rows(
        db_path,
        "session-bash",
        categories=frozenset({"file_stream"}),
    )
    assert structured_rows[0].title == "No Policy File Stream"
    assert "better-auth/plugins/jwt" not in {row.title for row in stream_rows}
    assert "/oauth2/token" not in {row.title for row in stream_rows}
    assert "application/json" not in {row.title for row in stream_rows}
    assert "/dev/null" not in {row.title for row in stream_rows}


def test_policy_projection_backfills_existing_turns_idempotently(tmp_path) -> None:
    policy = """
version: 1
rules:
  - name: destructive-command-observed
    on: tool_call_pre
    match: {tool: bash}
    when: "'rm -rf' in event.args.get('cmd', '')"
    effect: notify
    mode: observe
    reason: "destructive command"
"""
    rules, _disabled = compile_policy_file(policy)
    turn = _turn_with_tool(
        turn_index=0,
        tool_name="bash",
        args={"cmd": "rm -rf /tmp/nope"},
        result_text="blocked elsewhere",
        is_error=True,
        extras={
            "exit_code": 2,
            "duration_ms": 45,
            "agentm_runtime": {"result_content_hash": "result-hash-2"},
        },
    )

    db_path = tmp_path / "policy.db"
    persistence = PolicyPersistence(db_path)
    persistence.open()
    try:
        first = project_events(
            session_id="session-2",
            events=events_from_turns((turn,)),
            rules=rules,
            persistence=persistence,
        )
        persistence.delete_session("session-2")
        second = project_events(
            session_id="session-2",
            events=events_from_turns((turn,)),
            rules=rules,
            persistence=persistence,
        )
    finally:
        persistence.close()

    assert first.turns == 1
    assert first.tool_calls == 1
    assert first.effects == 1
    assert second.effects == 1

    engine = create_sqlite_engine(db_path)
    try:
        with engine.connect() as conn:
            assert _count(conn, "event_log") == 1
            assert _count(conn, "policy_tool_events") == 2
            assert _count(conn, "policy_turn_summary") == 1
            assert _count(conn, "policy_session_summary") == 1
            event = (
                conn.exec_driver_sql("SELECT * FROM event_log").mappings().fetchone()
            )
            assert event is not None
            assert event["session_id"] == "session-2"
            assert event["rule_id"] == "destructive-command-observed"
            post = (
                conn.exec_driver_sql(
                    """
                SELECT exit_code, duration_ms, result_content_hash, processed_json
                FROM policy_tool_events WHERE phase = 'post'
                """
                )
                .mappings()
                .fetchone()
            )
            assert post is not None
            assert post["exit_code"] == 2
            assert post["duration_ms"] == 45
            assert post["result_content_hash"] == "result-hash-2"
            assert json.loads(post["processed_json"])["exit_code"] == 2
    finally:
        engine.dispose()

    rows = load_policy_trace_rows(
        db_path,
        "session-2",
        categories=frozenset({"summary", "effects", "tools", "files", "entities"}),
    )
    assert any(row.title == "Projection Counts" for row in rows)
    assert any(row.title == "destructive-command-observed" for row in rows)
    assert any(row.tool_name == "bash" for row in rows)
    tool_rows = [row for row in rows if row.key.startswith("policy:tool:")]
    assert {row.display_name for row in tool_rows} == {"pre bash", "post bash"}
    assert any(row.cause == "pre taint:-" for row in tool_rows)
    assert any(str(row.cause).startswith("post err:") for row in tool_rows)
    assert all("hash:" in row.preview for row in tool_rows)
    assert any("fp:" in row.preview for row in tool_rows)


def _turn_with_tool(
    *,
    turn_index: int,
    tool_name: str,
    args: dict[str, object],
    result_text: str,
    is_error: bool = False,
    extras: dict[str, object] | None = None,
) -> Turn:
    call = ToolCallBlock(
        type="tool_call",
        id=f"call-{turn_index}",
        name=tool_name,
        arguments=args,  # type: ignore[arg-type]
    )
    result = ToolResultBlock(
        type="tool_result",
        tool_call_id=call.id,
        content=(TextContent(type="text", text=result_text),),
        is_error=is_error,
        extras=extras,
    )
    return Turn(
        index=turn_index,
        id=f"turn-{turn_index}",
        trigger=UserInput(
            content=(TextContent(type="text", text=f"turn {turn_index}"),)
        ),
        rounds=(
            Round(
                response=AssistantMessage(
                    role="assistant",
                    content=(call,),
                    timestamp=1.0,
                ),
                tool_results=(ToolRecord(call=call, result=result),),
            ),
        ),
        outcome=Outcome(cause=ModelEndTurn()),
        timestamp=2.0,
    )
