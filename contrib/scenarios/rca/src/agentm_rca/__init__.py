"""RCA scenario package built on top of the AgentM SDK."""

from __future__ import annotations

from pathlib import Path

SCENARIO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
"""Absolute path to the ``contrib/scenarios/rca/`` directory.

Derived from the installed package location (stable across editable and
wheel installs). Atoms use this to locate sibling resources (prompts/,
skills/, etc.) instead of fragile ``__file__``-relative parent chains.
"""
