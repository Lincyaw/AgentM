"""Fail-stop tests for the ``get_event_detail([ids])`` drill-down tool.

Same fail-stop rationale as ``test_get_turn_tool``: the tool runs inside
the auditor child loop. If it raises on bad input instead of returning
a structured error, the entire audit firing collapses without a verdict
entry.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agentm.core.abi import ToolResult

from llmharness.schema import Edge, EdgeKind, Event, EventKind


def _install_and_capture(*, events: list[Event], edges: list[Edge]) -> Any:
    """Return the registered tool's ``fn`` callback."""
    from llmharness.audit.auditor.get_event_detail_tool import install

    captured: list[Any] = []

    class _CapturAPI:
        def register_tool(self, tool: Any) -> None:
            captured.append(tool)

    install(_CapturAPI(), {"events": events, "edges": edges})  # type: ignore[arg-type]
    assert len(captured) == 1
    return captured[0].fn


def _ev(i: int) -> Event:
    return Event(id=i, kind=EventKind.ACT, summary=f"e{i}", source_turns=[i])


def _edge(src: int, dst: int) -> Edge:
    return Edge(
        src=src,
        dst=dst,
        kind=EdgeKind.DATA,
        reason="r",
        src_turns=(src,),
        dst_turns=(dst,),
        cited_entities=("x",),
        cited_quote="x",
    )


def test_happy_path_returns_event_with_outgoing_and_incoming_edges() -> None:
    events = [_ev(1), _ev(2), _ev(3)]
    edges = [_edge(1, 2), _edge(2, 3), _edge(3, 1)]
    fn = _install_and_capture(events=events, edges=edges)

    result: ToolResult = asyncio.run(fn({"event_ids": [1, 2]}))

    assert isinstance(result, ToolResult)
    assert result.is_error is False
    payload = json.loads(result.content[0].text)

    assert "1" in payload
    assert "2" in payload
    assert payload["1"]["event"]["id"] == 1
    # outgoing(1) = edge(1,2); incoming(1) = edge(3,1)
    out_dsts_1 = [e["dst"] for e in payload["1"]["outgoing_edges"]]
    in_srcs_1 = [e["src"] for e in payload["1"]["incoming_edges"]]
    assert out_dsts_1 == [2]
    assert in_srcs_1 == [3]
    # outgoing(2) = edge(2,3); incoming(2) = edge(1,2)
    out_dsts_2 = [e["dst"] for e in payload["2"]["outgoing_edges"]]
    in_srcs_2 = [e["src"] for e in payload["2"]["incoming_edges"]]
    assert out_dsts_2 == [3]
    assert in_srcs_2 == [1]
    assert "missing" not in payload


def test_unknown_id_recorded_in_missing_list() -> None:
    events = [_ev(1)]
    edges: list[Edge] = []
    fn = _install_and_capture(events=events, edges=edges)

    result: ToolResult = asyncio.run(fn({"event_ids": [1, 999]}))

    assert result.is_error is False
    payload = json.loads(result.content[0].text)
    assert "1" in payload
    assert payload["1"]["event"]["id"] == 1
    assert payload.get("missing") == [999]
    # Unknown id NOT under a stringified key.
    assert "999" not in payload


def test_negative_id_returns_structured_error() -> None:
    fn = _install_and_capture(events=[_ev(1)], edges=[])

    result: ToolResult = asyncio.run(fn({"event_ids": [-1]}))

    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert "negative id" in result.content[0].text


def test_empty_event_ids_list_returns_structured_error() -> None:
    fn = _install_and_capture(events=[_ev(1)], edges=[])

    result: ToolResult = asyncio.run(fn({"event_ids": []}))

    assert result.is_error is True
    assert "non-empty" in result.content[0].text


def test_non_int_id_returns_structured_error() -> None:
    fn = _install_and_capture(events=[_ev(1)], edges=[])

    result: ToolResult = asyncio.run(fn({"event_ids": ["1"]}))

    assert result.is_error is True
    assert "must" in result.content[0].text
