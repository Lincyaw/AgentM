"""Fail-stop tests for ``core.runtime.trace_reader``.

TraceReader is the single seam every reader in the tree depends on after
PR-G. The on-disk wire format is locked down by ``test_otel_export.py``;
these tests pin the *read* contract — span/log iteration, attribute
unwrap, convenience accessors — against synthetic OTLP/JSON ndjson.

Tests deliberately use hand-built fixtures (not the real SDK exporter)
so they exercise edge cases the exporter would never produce: split
batches per line, intValue strings, nested kvlists, malformed lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.core.abi import LogRecord, Span, TraceReader, attr


def _write_lines(path: Path, lines: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(line, separators=(",", ":")) for line in lines) + "\n",
        encoding="utf-8",
    )


def _span_line(
    *,
    name: str,
    attributes: list[dict] | None = None,
    span_id: str = "AAAAAAAA",
    parent_span_id: str | None = None,
    trace_id: str = "TTTTTTTT",
    kind: str = "SPAN_KIND_INTERNAL",
    start_ns: int = 1_000_000,
    end_ns: int = 2_000_000,
) -> dict:
    span: dict = {
        "name": name,
        "traceId": trace_id,
        "spanId": span_id,
        "kind": kind,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": attributes or [],
    }
    if parent_span_id is not None:
        span["parentSpanId"] = parent_span_id
    return {
        "resource": {"attributes": []},
        "scopeSpans": [{"scope": {"name": "agentm"}, "spans": [span]}],
    }


def _log_line(
    *,
    event_name: str,
    body: dict | None = None,
    body_value: dict | None = None,
    attributes: list[dict] | None = None,
    span_id: str | None = None,
    time_ns: int = 1_500_000,
) -> dict:
    record: dict = {
        "eventName": event_name,
        "severityNumber": "SEVERITY_NUMBER_INFO",
        "severityText": "INFO",
        "timeUnixNano": str(time_ns),
        "attributes": attributes or [],
    }
    if body_value is not None:
        record["body"] = body_value
    elif body is not None:
        record["body"] = {
            "kvlistValue": {
                "values": [
                    {"key": k, "value": {"stringValue": v} if isinstance(v, str) else v}
                    for k, v in body.items()
                ]
            }
        }
    if span_id is not None:
        record["spanId"] = span_id
    return {
        "resource": {"attributes": []},
        "scopeLogs": [{"scope": {"name": "agentm"}, "logRecords": [record]}],
    }


def _kv(key: str, value: dict) -> dict:
    return {"key": key, "value": value}


def test_iter_spans_filters_by_name_and_attribute(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_lines(
        path,
        [
            _span_line(
                name="chat m1",
                attributes=[
                    _kv("gen_ai.request.model", {"stringValue": "m1"}),
                    _kv("agentm.turn.index", {"intValue": "3"}),
                ],
            ),
            _span_line(name="execute_tool read", span_id="BBBBBBBB"),
            _span_line(
                name="chat m2",
                attributes=[_kv("gen_ai.request.model", {"stringValue": "m2"})],
                span_id="CCCCCCCC",
            ),
        ],
    )

    reader = TraceReader(path)
    chat_names = [s.name for s in reader.iter_spans(name="chat m1")]
    assert chat_names == ["chat m1"]

    chats = list(reader.iter_spans(name="chat m1"))
    assert chats[0].attributes["agentm.turn.index"] == 3  # intValue parsed
    assert chats[0].attributes["gen_ai.request.model"] == "m1"
    assert chats[0].span_id == "AAAAAAAA"
    assert chats[0].trace_id == "TTTTTTTT"

    by_attr = list(
        reader.iter_spans(attribute_filters={"gen_ai.request.model": "m2"})
    )
    assert [s.name for s in by_attr] == ["chat m2"]


def test_iter_log_records_unwraps_kvlist_body(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_lines(
        path,
        [
            _log_line(
                event_name="agentm.session.fingerprint",
                body={"scenario": "general_purpose"},
            ),
            _log_line(
                event_name="agentm.message.appended",
                body={"id": "msg-1", "type": "message"},
                attributes=[_kv("agentm.message.id", {"stringValue": "msg-1"})],
                span_id="SP",
            ),
        ],
    )

    reader = TraceReader(path)
    fp = reader.load_session_fingerprint()
    assert fp == {"scenario": "general_purpose"}

    msgs = reader.load_messages()
    assert msgs == [{"id": "msg-1", "type": "message"}]

    by_parent = list(reader.iter_log_records(parent_span_id="SP"))
    assert len(by_parent) == 1
    assert by_parent[0].event_name == "agentm.message.appended"
    assert by_parent[0].attributes["agentm.message.id"] == "msg-1"

    no_match = list(reader.iter_log_records(parent_span_id="MISSING"))
    assert no_match == []


def test_iter_all_interleaves_spans_and_logs_in_file_order(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_lines(
        path,
        [
            _log_line(event_name="agentm.session.header", body={"id": "s1"}),
            _span_line(name="chat m1"),
            _log_line(event_name="agentm.turn.summary", body={"tool_call_count": "1"}),
        ],
    )
    items = list(TraceReader(path).iter_all())
    kinds = [type(item).__name__ for item in items]
    assert kinds == ["LogRecord", "Span", "LogRecord"]
    assert isinstance(items[0], LogRecord)
    assert items[0].event_name == "agentm.session.header"
    assert isinstance(items[1], Span)
    assert items[1].name == "chat m1"


def test_chat_calls_yields_only_chat_spans(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_lines(
        path,
        [
            _span_line(name="chat m1"),
            _span_line(name="execute_tool read"),
            _span_line(name="chat m2", span_id="BB"),
            _span_line(name="invoke_agent general_purpose"),
        ],
    )
    names = [s.name for s in TraceReader(path).chat_calls()]
    assert names == ["chat m1", "chat m2"]


def test_load_session_header_returns_latest(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_lines(
        path,
        [
            _log_line(event_name="agentm.session.header", body={"id": "s1", "cwd": "/a"}),
            _log_line(event_name="agentm.session.header", body={"id": "s1", "cwd": "/b"}),
        ],
    )
    header = TraceReader(path).load_session_header()
    assert header == {"id": "s1", "cwd": "/b"}


def test_load_turn_summaries_in_order(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_lines(
        path,
        [
            _log_line(
                event_name="agentm.turn.summary",
                body={"input_tokens": {"intValue": "10"}},
            ),
            _log_line(
                event_name="agentm.turn.summary",
                body={"input_tokens": {"intValue": "20"}},
            ),
        ],
    )
    summaries = TraceReader(path).load_turn_summaries()
    assert [s["input_tokens"] for s in summaries] == [10, 20]


def test_attr_helper_returns_default_for_missing(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_lines(
        path,
        [
            _span_line(
                name="chat m1",
                attributes=[_kv("gen_ai.request.model", {"stringValue": "m1"})],
            )
        ],
    )
    span = next(TraceReader(path).iter_spans())
    assert attr(span, "gen_ai.request.model") == "m1"
    assert attr(span, "missing", "fallback") == "fallback"


def test_malformed_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps(_log_line(event_name="agentm.session.header", body={"id": "s1"}))
        + "\nnot-json-at-all\n"
        + json.dumps(_log_line(event_name="agentm.message.appended", body={"id": "m1", "type": "message"}))
        + "\n",
        encoding="utf-8",
    )
    reader = TraceReader(path)
    assert reader.load_session_header() == {"id": "s1"}
    assert reader.load_messages() == [{"id": "m1", "type": "message"}]


def test_missing_file_returns_empty_iterators(tmp_path: Path) -> None:
    reader = TraceReader(tmp_path / "does-not-exist.jsonl")
    assert list(reader.iter_spans()) == []
    assert list(reader.iter_log_records()) == []
    assert reader.load_messages() == []
    assert reader.load_session_header() is None
    assert reader.load_session_fingerprint() is None


def test_iterators_are_lazy_generators(tmp_path: Path) -> None:
    """Calling an iterator method must not load the whole file."""
    path = tmp_path / "trace.jsonl"
    _write_lines(path, [_span_line(name="chat m1"), _span_line(name="chat m2")])
    reader = TraceReader(path)
    iterator = reader.iter_spans()
    # types.GeneratorType — single-shot, not a list.
    import types

    assert isinstance(iterator, types.GeneratorType)
    first = next(iterator)
    assert first.name == "chat m1"


def test_arrayvalue_and_nested_kvlist_unwrap(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    nested_body = {
        "kvlistValue": {
            "values": [
                {
                    "key": "tools",
                    "value": {
                        "arrayValue": {
                            "values": [
                                {"stringValue": "read"},
                                {"stringValue": "write"},
                            ]
                        }
                    },
                },
                {
                    "key": "nested",
                    "value": {
                        "kvlistValue": {
                            "values": [
                                {"key": "k", "value": {"intValue": "42"}},
                            ]
                        }
                    },
                },
            ]
        }
    }
    _write_lines(
        path,
        [_log_line(event_name="agentm.session.ready", body_value=nested_body)],
    )
    record = next(
        TraceReader(path).iter_log_records(name="agentm.session.ready")
    )
    assert record.body == {"tools": ["read", "write"], "nested": {"k": 42}}


def test_tool_calls_pairs_args_and_result_by_span_id(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_lines(
        path,
        [
            _span_line(name="execute_tool read", span_id="SP1"),
            _log_line(
                event_name="agentm.tool.call.arguments",
                body={"path": "/x"},
                span_id="SP1",
            ),
            _log_line(
                event_name="agentm.tool.call.result",
                body={"ok": "yes"},
                span_id="SP1",
            ),
            _span_line(name="execute_tool write", span_id="SP2"),
        ],
    )
    triples = list(TraceReader(path).tool_calls())
    assert len(triples) == 2
    span1, args1, result1 = triples[0]
    assert span1.name == "execute_tool read"
    assert args1 is not None and args1.body == {"path": "/x"}
    assert result1 is not None and result1.body == {"ok": "yes"}
    span2, args2, result2 = triples[1]
    assert span2.name == "execute_tool write"
    assert args2 is None and result2 is None


@pytest.mark.parametrize(
    "wrapped,expected",
    [
        ({"stringValue": "abc"}, "abc"),
        ({"intValue": "12"}, 12),
        ({"intValue": 12}, 12),
        ({"doubleValue": "1.5"}, 1.5),
        ({"boolValue": True}, True),
        ({"arrayValue": {"values": [{"stringValue": "x"}]}}, ["x"]),
    ],
)
def test_span_attributes_are_unwrapped(
    tmp_path: Path, wrapped: dict, expected
) -> None:
    path = tmp_path / "trace.jsonl"
    _write_lines(
        path,
        [_span_line(name="chat m1", attributes=[_kv("k", wrapped)])],
    )
    span = next(TraceReader(path).iter_spans())
    assert span.attributes["k"] == expected
