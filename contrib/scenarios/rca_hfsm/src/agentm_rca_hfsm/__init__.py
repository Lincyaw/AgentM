"""Hypothesis-driven, falsification-gated RCA scenario package.

Phase 1 surface: the L1 ``HypothesisGraph`` schema (``schema``) and the
single-writer store atom (``atoms.rca_hgraph_store``). Later phases add the
falsification gate, evidence tools, FSM policy, brief builder, and finalize
guard — see ``.claude/plans/2026-05-13-rca-hfsm-phase1.md``.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
