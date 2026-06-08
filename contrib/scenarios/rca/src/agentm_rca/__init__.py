"""RCA scenario — §11 atoms + HFSM strategy.

Shared atoms (baseline + multi-agent + harness):
    duckdb_sql, finalize, prompt_loader, rcabench_contract

Multi-agent additions:
    hypothesis_tools (requires artifact_store), worker_finalize

Overlays (orthogonal, via manifest includes):
    runtime_context, worker_skills

HFSM strategy (self-contained sub-package):
    hfsm/
"""

from __future__ import annotations

from pathlib import Path

SCENARIO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
"""Absolute path to ``contrib/scenarios/rca/``.

Atoms use this to locate sibling resources (prompts/, skills/)
instead of fragile ``__file__``-relative parent chains.
"""
