"""Host-side tools for the llmharness cognitive-audit pipeline.

These are NOT part of the ``llmharness`` src package — they are host-side
drivers, CLI entry points, and offline orchestrators that import from
``llmharness`` (the installed package) and ``agentm.core.runtime.*``
(the AgentM SDK runtime).

Subdirectories:
- ``replay/``     — replay engine, single-firing and chain replay
- ``distill/``    — offline labeling and SFT/RL data export
- ``aggregate/``  — case-level aggregation from replay sidecars
- ``eval/``       — benchmark evaluation drivers
- ``extensions/`` — pure check functions (no longer atoms)
"""
