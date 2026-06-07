"""Hypothesis-driven, falsification-gated RCA scenario package.

Phase 1 surface: the L1 ``HypothesisGraph`` schema (``schema``) and the
single-writer store atom (``atoms.rca_hgraph_store``). Later phases add the
falsification gate, evidence tools, FSM policy, brief builder, and finalize
guard — see ``.claude/designs/hypothesis-driven-rca.md``.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
