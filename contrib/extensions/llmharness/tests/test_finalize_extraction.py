"""Pin the v4 ``finalize_extraction`` tool contract.

V4 (2026-05-24): finalize ALWAYS commits on a witness-valid graph.
Two load-bearing paths:

* **happy (clean graph)** — finalize returns ToolTerminate carrying
  the agreed reason string, with a digest payload and NO advisory
  note. Without this, the audit child loop never knows the firing
  is done and the adapter records a no-call failure.

* **happy (chain shape)** — finalize still returns ToolTerminate
  (NOT a blocking ToolResult), and the success text carries a
  "Graph committed. Note: …" advisory naming the chain-link event
  ids. The model gets a hint for the next firing without being
  trapped in the current one.
"""

from __future__ import annotations

import pytest
from agentm.core.abi import ToolTerminate

from llmharness.audit.extractor.finalize_extraction import (
    FINALIZE_EXTRACTION_REASON,
    build_finalize_extraction_tool,
)
from llmharness.audit.extractor.state import ExtractionState
from llmharness.schema import Edge, EdgeKind, Event, EventKind


@pytest.mark.asyncio
async def test_finalize_extraction_happy_path_terminates() -> None:
    state = ExtractionState()
    # Inject a one-event firing directly into the legacy snapshot. A
    # single-event graph has no chain links so no advisory is attached.
    state._events_pending = [
        Event(id=1, kind=EventKind.TASK, summary="root", source_turns=(0,))
    ]
    tool = build_finalize_extraction_tool(state)

    outcome = await tool.fn({})

    assert isinstance(outcome, ToolTerminate)
    assert outcome.reason == FINALIZE_EXTRACTION_REASON
    assert state.committed is True
    assert len(state.events) == 1
    text = outcome.result.content[0].text  # type: ignore[union-attr]
    assert "Note:" not in text  # no chain-link advisory on a clean graph


@pytest.mark.asyncio
async def test_finalize_extraction_chain_shape_commits_with_advisory() -> None:
    state = ExtractionState()
    # Linear chain 1 -> 2 -> 3 makes event 2 a chain link (in=1, out=1).
    # Under v4 this commits cleanly and the success text carries a
    # soft warning naming event[2].
    state._events_pending = [
        Event(id=1, kind=EventKind.TASK, summary="root", source_turns=(0,)),
        Event(id=2, kind=EventKind.ACT, summary="mid", source_turns=(1,)),
        Event(id=3, kind=EventKind.ACT, summary="leaf", source_turns=(2,)),
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

    # Firing terminates regardless of chain shape.
    assert isinstance(outcome, ToolTerminate)
    assert outcome.reason == FINALIZE_EXTRACTION_REASON
    assert state.committed is True

    text = outcome.result.content[0].text  # type: ignore[union-attr]
    # Advisory appended to success result names the chain-link id.
    assert "Graph committed." in text
    assert "Note:" in text
    assert "event[2]" in text
