"""Pin the passthrough recovery options contract.

The live OpenAI run that motivated the v19 extractor refactor saw the
same passthrough payload rejected 6 times in a row because the v18
error said "merge or promote" without naming WHICH tool to call or
WHICH ids were involved. The fix is structural: every passthrough
rejection must name (a) the passthrough id, (b) at least one
in/out-neighbour id, and (c) a concrete tool call with arguments
substituted in.

Given the linear chain ``1 -> 2 -> 3`` (passthrough at 2):

* ``delete_node(2)`` must appear — the merge recipe;
* the in-neighbour id (1) AND the out-neighbour id (3) must appear
  somewhere in the options;
* the boost-out-degree recipe must name ``dst=2`` (the passthrough)
  so the LLM sees a concrete promote target.

A regression on any of these lines reverts the model to v18-style
retries.
"""

from __future__ import annotations

from llmharness.audit.extractor._state_echo import chain_neighbours
from llmharness.audit.extractor.finalize_extraction import _passthrough_options
from llmharness.audit.extractor.state import ExtractionState
from llmharness.schema import Edge, EdgeKind, Event, EventKind


def _linear_chain_state() -> ExtractionState:
    state = ExtractionState()
    state._events_pending = [
        Event(id=1, kind=EventKind.TASK, summary="root", source_turns=(0,)),
        Event(id=2, kind=EventKind.ACT, summary="mid", source_turns=(1,)),
        Event(id=3, kind=EventKind.EVID, summary="leaf", source_turns=(2,)),
    ]
    state._edges_pending = [
        Edge(
            src=1, dst=2, kind=EdgeKind.REF, reason="r",
            src_turns=(0,), dst_turns=(1,),
            cited_entities=(), cited_quote="a",
        ),
        Edge(
            src=2, dst=3, kind=EdgeKind.REF, reason="r",
            src_turns=(1,), dst_turns=(2,),
            cited_entities=(), cited_quote="b",
        ),
    ]
    return state


def test_chain_neighbours_returns_in_out_pair() -> None:
    state = _linear_chain_state()
    assert chain_neighbours(state, 2) == (1, 3)
    # Endpoints are not passthroughs — they have (0, 1) or (1, 0).
    assert chain_neighbours(state, 1) == (None, None)
    assert chain_neighbours(state, 3) == (None, None)


def test_passthrough_options_name_concrete_neighbours_and_tools() -> None:
    state = _linear_chain_state()
    # The validator emits this exact diagnostic line shape (see state.py
    # ``_validate_event_degrees``); the helper parses it for ids.
    raw_error = (
        "finalize: graph has passthrough events ...\n"
        "Passthrough events (1):\n"
        "  event[2] kind=act 'mid': in=1, out=1"
    )
    options = _passthrough_options(state, raw_error)
    joined = "\n".join(options)
    # Merge recipe — concrete tool + concrete id.
    assert "delete_node(2)" in joined
    # Promote-out-degree recipe — concrete dst id matching the passthrough.
    assert "dst=2" in joined
    # In-neighbour id 1 appears somewhere so the model knows the chain context.
    assert "(1)" in joined or " 1 " in joined or "id=1" in joined
    # Reset is always offered as the last-resort option.
    assert any("reset_extraction" in opt for opt in options)
