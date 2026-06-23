"""RCA scenario atoms.

default/    shared atoms (baseline + multi-agent + harness)
hfsm/       hypothesis-falsification state machine strategy
"""

from __future__ import annotations

from pathlib import Path

SCENARIO_ROOT: Path = Path(__file__).parents[2]
"""Absolute path to ``contrib/scenarios/rca/``."""
