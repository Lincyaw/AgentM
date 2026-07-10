"""Measurement-first rescue-window harness (see DESIGN.md / doc.md).

Layered by the doc's design intent:

- ``model``    — problem-definition data model (doc §3, §6.2, §7).
- ``harness``  — replayable benchmark machinery (doc §6, §7, §13.1).
- ``analysis`` — opportunity / window / ladder / report (doc §3.3, §4, §9, §10).
- ``critic``   — bounded critic baselines (doc §3.4 / E4).
"""

from .model import (
    ActionType,
    ContentLevel,
    EvalUnit,
    EvalUnitStore,
    ForkPoint,
    Intervention,
    LadderRung,
    PrefixPoint,
    Treatment,
)

__all__ = [
    "ActionType",
    "ContentLevel",
    "EvalUnit",
    "EvalUnitStore",
    "ForkPoint",
    "Intervention",
    "LadderRung",
    "PrefixPoint",
    "Treatment",
]


# ---------------------------------------------------------------------------
# Benchmark registration — mounts the rescue-window CLI as a subcommand
# of agentm-eval, wrapping it with experiment lifecycle.
# ---------------------------------------------------------------------------

import typer

from agentm_eval.registry import register


class RescueWindowBenchmark:
    name = "rescue-window"
    description = "Fork-at-prefix branching experiments"

    def create_cli(self) -> typer.Typer:
        from .cli import app as rw_app

        return rw_app


register("rescue-window", RescueWindowBenchmark.description, RescueWindowBenchmark)
