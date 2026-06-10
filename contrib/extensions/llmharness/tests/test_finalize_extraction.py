"""Pin the v4 ``finalize_extraction`` tool contract.

V4 (2026-05-24): finalize ALWAYS commits on a witness-valid graph from
the op log (``apply_*`` surface). ``finalize_extraction`` is an OPTIONAL
fast-exit — the runner commits on stop from the same op log regardless —
so these tests pin its still-supported early-termination contract, not a
commit gate. Each case drives the real production path
(``apply_node_upsert`` / ``apply_edge_upsert``), never the removed legacy
``_events_pending`` shortcut.

Load-bearing paths:

* **happy (clean graph)** — finalize returns ToolTerminate carrying the
  agreed reason string, a digest payload, and NO advisory note.
* **happy (chain shape)** — a 1->2->3 chain (node 2 is a chain link:
  in=1, out=1) still returns ToolTerminate (NOT a blocking ToolResult),
  and the success text carries a "Graph committed. Note: …" advisory
  naming the chain-link event. The model gets a hint for the next firing
  without being trapped in the current one.
* **already finalized** — a second finalize returns an is_error ToolResult
  (the one remaining failure mode), not a terminate.
"""

from __future__ import annotations

import pytest
from agentm.core.abi import ToolResult, ToolTerminate

from llmharness.agents.extractor.state import ExtractionState
from llmharness.agents.extractor.tools.finalize_extraction import (
    FINALIZE_EXTRACTION_REASON,
    build_finalize_extraction_tool,
)


def _node(state: ExtractionState, node_id: int, summary: str) -> None:
    """Apply one node via the v4 op-log surface; assert it was accepted."""
    out = state.apply_node_upsert(
        {"id": node_id, "kind": "act", "summary": summary, "source_turns": [0]}
    )
    assert isinstance(out, dict), out


def _data_edge(state: ExtractionState, src: int, dst: int) -> None:
    """Apply one ``data`` edge (no witness needed); assert it was accepted."""
    out = state.apply_edge_upsert(
        {"src": src, "dst": dst, "kind": "data", "reason": "r", "cited_entities": ["svc"]}
    )
    assert isinstance(out, dict), out


@pytest.mark.asyncio
async def test_finalize_extraction_happy_path_terminates() -> None:
    state = ExtractionState()
    _node(state, 1, "root")
    tool = build_finalize_extraction_tool(state)

    outcome = await tool.fn({})

    assert isinstance(outcome, ToolTerminate)
    assert outcome.reason == FINALIZE_EXTRACTION_REASON
    assert state.committed is True
    assert len(state.events) == 1
    text = outcome.result.content[0].text  # type: ignore[union-attr]
    assert "Note:" not in text  # no chain-link advisory on a single-node graph


@pytest.mark.asyncio
async def test_finalize_extraction_chain_shape_terminates_with_advisory() -> None:
    state = ExtractionState()
    _node(state, 1, "task")
    _node(state, 2, "middle")
    _node(state, 3, "leaf")
    _data_edge(state, 1, 2)
    _data_edge(state, 2, 3)
    tool = build_finalize_extraction_tool(state)

    outcome = await tool.fn({})

    assert isinstance(outcome, ToolTerminate)
    assert outcome.reason == FINALIZE_EXTRACTION_REASON
    assert state.committed is True
    assert len(state.events) == 3
    text = outcome.result.content[0].text  # type: ignore[union-attr]
    # Node 2 is a chain link (in=1, out=1) so the soft advisory fires.
    assert "Graph committed. Note:" in text
    assert "event[2]" in text


@pytest.mark.asyncio
async def test_finalize_extraction_second_call_is_error() -> None:
    state = ExtractionState()
    _node(state, 1, "root")
    tool = build_finalize_extraction_tool(state)

    first = await tool.fn({})
    assert isinstance(first, ToolTerminate)

    second = await tool.fn({})
    assert isinstance(second, ToolResult)
    assert second.is_error is True
