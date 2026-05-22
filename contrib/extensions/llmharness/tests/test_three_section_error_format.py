"""Pin the three-section render contract for ``format_witness_error``.

Every extractor tool error funnels through this helper. The shape is
the SFT-quality fix that motivated the v19 extractor refactor (see the
helper's module docstring), so a structural drift would silently
degrade live trajectories. We pin:

* sections appear in order: symptom, what-you-tried, current-graph,
  next-options;
* options carry alphabetical labels ``(a)``, ``(b)``, ...;
* empty ``options`` raises (no unactionable errors slip through).
"""

from __future__ import annotations

import pytest

from llmharness.audit._witness_errors import format_witness_error


def test_renders_all_three_sections_in_order() -> None:
    text = format_witness_error(
        symptom="passthrough event in chain",
        attempt="upsert_node(id=3, kind=evid)",
        state_echo="3 nodes, 2 edges; last: id=2 kind=act",
        options=[
            "delete_node(2) then upsert_node(id=3, ...) merging act+evid",
            "upsert_edge(src=4, dst=2, kind=ref, ...) to promote node 2",
        ],
    )
    # Symptom is the first line.
    lines = text.splitlines()
    assert lines[0] == "passthrough event in chain"
    # what-you-tried + current-graph block in order, then next-options.
    tried_idx = next(i for i, ln in enumerate(lines) if "what you tried:" in ln)
    graph_idx = next(i for i, ln in enumerate(lines) if "current graph:" in ln)
    options_idx = next(i for i, ln in enumerate(lines) if "next options:" in ln)
    assert tried_idx < graph_idx < options_idx
    # Option labels are (a), (b) in order.
    option_lines = [ln for ln in lines if ln.lstrip().startswith("(")]
    assert option_lines[0].lstrip().startswith("(a)")
    assert option_lines[1].lstrip().startswith("(b)")


def test_placeholders_when_attempt_or_state_missing() -> None:
    text = format_witness_error(
        symptom="malformed id",
        attempt=None,
        state_echo=None,
        options=["re-call with corrected id"],
    )
    assert "what you tried:    —" in text
    assert "current graph:     (empty)" in text


def test_empty_options_rejected() -> None:
    with pytest.raises(ValueError):
        format_witness_error(
            symptom="x", attempt=None, state_echo=None, options=[]
        )
