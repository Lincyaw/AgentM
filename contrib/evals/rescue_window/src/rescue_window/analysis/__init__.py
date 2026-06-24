"""Measurement layer (doc §3.3, §4, §9, §10).

Pure functions over ``EvalUnit`` rows: the outcome aggregator (Q/Δ/G*/G^C/gap +
adaptive-K), the Rescue Window estimator, the Rescuability Ladder, and the report
generator. No rollouts happen here — re-deriving metrics never re-runs the actor.
"""

from .aggregate import (
    AggregateResult,
    CellStat,
    PrefixStat,
    aggregate,
    undecided_cells,
)
from .ladder import PrefixLadder, ladder_gaps
from .report import build_report, to_markdown, write_report
from .window import WindowMetrics, estimate_windows

__all__ = [
    "AggregateResult",
    "CellStat",
    "PrefixLadder",
    "PrefixStat",
    "WindowMetrics",
    "aggregate",
    "build_report",
    "estimate_windows",
    "ladder_gaps",
    "to_markdown",
    "undecided_cells",
    "write_report",
]
