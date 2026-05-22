"""Pin the v19 ``finalize_extraction`` tool contract.

Two load-bearing paths:

* **happy** — state has accepted events, finalize returns ToolTerminate
  carrying the agreed reason string. Without this, the audit child
  loop never knows the firing is done and the adapter records a
  no-call failure.

* **passthrough** — state has a linear (in=1, out=1) chain, finalize
  returns a ToolResult whose ``options`` section names the specific
  in/out-neighbour ids. Without this, the witness retry behaviour
  observed on the live OpenAI run returns — generic "merge or promote"
  with no concrete tool call.
"""

from __future__ import annotations

import pytest
from agentm.core.abi import ToolResult, ToolTerminate

from llmharness.audit.extractor.finalize_extraction import (
    FINALIZE_EXTRACTION_REASON,
    build_finalize_extraction_tool,
)
from llmharness.audit.extractor.state import ExtractionState
from llmharness.schema import Edge, EdgeKind, Event, EventKind


@pytest.mark.asyncio
async def test_finalize_extraction_happy_path_terminates() -> None:
    state = ExtractionState()
    # Inject a one-event firing directly into the legacy snapshot so
    # finalize's degree-check exemption (single node) applies.
    state._events_pending = [
        Event(id=1, kind=EventKind.TASK, summary="root", source_turns=(0,))
    ]
    tool = build_finalize_extraction_tool(state)

    outcome = await tool.fn({})

    assert isinstance(outcome, ToolTerminate)
    assert outcome.reason == FINALIZE_EXTRACTION_REASON
    assert state.committed is True
    assert len(state.events) == 1


@pytest.mark.asyncio
async def test_finalize_extraction_passthrough_names_neighbours() -> None:
    state = ExtractionState()
    # Linear chain 1 -> 2 -> 3 makes event 2 a passthrough (in=1, out=1).
    state._events_pending = [
        Event(id=1, kind=EventKind.TASK, summary="root", source_turns=(0,)),
        Event(id=2, kind=EventKind.ACT, summary="mid", source_turns=(1,)),
        Event(id=3, kind=EventKind.EVID, summary="leaf", source_turns=(2,)),
    ]
    state._edges_pending = [
        Edge(
            src=1, dst=2, kind=EdgeKind.REF, reason="r",
            src_turns=(0,), dst_turns=(1,),
            cited_entities=(), cited_quote="x",
        ),
        Edge(
            src=2, dst=3, kind=EdgeKind.REF, reason="r",
            src_turns=(1,), dst_turns=(2,),
            cited_entities=(), cited_quote="y",
        ),
    ]
    tool = build_finalize_extraction_tool(state)

    outcome = await tool.fn({})

    assert isinstance(outcome, ToolResult)
    assert outcome.is_error is True
    text = outcome.content[0].text  # type: ignore[union-attr]
    # Three-section template present.
    assert "what you tried:" in text
    assert "current graph:" in text
    assert "next options:" in text
    # Options must name the passthrough's actual in/out neighbours
    # AND the merge / promote tool calls — concrete, not generic.
    assert "delete_node(2)" in text
    assert "upsert_node" in text
    # The recovery option names dst=2 (the passthrough id) to boost out-deg.
    assert "dst=2" in text
    # State remains uncommitted so the loop can recover with more edits.
    assert state.committed is False
