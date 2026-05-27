"""Fail-stop tests for the ``get_event_detail([ids])`` drill-down tool.

Same fail-stop rationale as ``test_get_turn_tool``: the tool runs inside
the auditor child loop. If it raises on bad input instead of returning
a structured error, the entire audit firing collapses without a verdict
entry.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentm.core.abi import ToolResult

from llmharness.schema import Edge, EdgeKind, Event, EventKind


def _install_and_capture(*, events: list[Event], edges: list[Edge]) -> Any:
    """Return the registered tool's ``fn`` callback."""
    from llmharness.audit.auditor.get_event_detail import build_get_event_detail_tool

    return build_get_event_detail_tool(events, edges).fn


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


def test_negative_id_returns_structured_error() -> None:
    fn = _install_and_capture(events=[_ev(1)], edges=[])

    result: ToolResult = asyncio.run(fn({"event_ids": [-1]}))

    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert "negative id" in result.content[0].text
