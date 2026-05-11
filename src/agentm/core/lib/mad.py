"""Median-absolute-deviation (MAD) confidence helper.

Pure utility belonging to ``core/lib/`` alongside ``frontmatter`` — no
I/O, no harness imports, no atom dependencies.

Used by ``tool_propose_change`` (B-PRB) as an *advisory* signal layered
on top of the existing 2-sigma noise floor in the 4-floor deployment
gate. MAD is robust to outliers and friendly to small-N evaluation
sets where 2-sigma over a 3-point grade is a poor noise estimator.

The function does NOT decide accept/reject — it produces a tier label
that gets recorded on the activation record so reflection / observers
have visibility into the noise quality of each decision. A follow-up
will migrate floor 2 from sigma to MAD once activations.jsonl shows
what the tier distribution looks like in practice.
"""

from __future__ import annotations

import statistics
from typing import Any


def mad_confidence(
    values: list[float], baseline: float, candidate: float
) -> dict[str, Any] | None:
    """Compute the MAD-confidence tier for ``candidate`` against
    ``baseline`` given a reference pool ``values``.

    Returns ``None`` when ``len(values) < 3`` (too small to estimate
    dispersion) or when computed MAD == 0 (the pool is degenerate /
    all identical, no signal). Otherwise returns
    ``{"ratio": float, "tier": "real" | "marginal" | "noise"}`` where:

    - ``ratio = abs(candidate - baseline) / MAD(values)``
    - ``ratio >= 2.0``  -> ``"real"``
    - ``1.0 <= ratio < 2.0`` -> ``"marginal"``
    - ``ratio < 1.0``  -> ``"noise"``

    MAD here is the population median absolute deviation (no scaling
    constant); the choice of 1.0 / 2.0 thresholds is a heuristic
    informed by the 2-sigma incumbent floor and is documented in
    ``.claude/designs/per-task-evolution-loop.md`` once PR-B is merged.
    """
    if len(values) < 3:
        return None
    median = statistics.median(values)
    deviations = [abs(v - median) for v in values]
    mad = statistics.median(deviations)
    if mad == 0:
        return None
    ratio = abs(candidate - baseline) / mad
    if ratio >= 2.0:
        tier = "real"
    elif ratio >= 1.0:
        tier = "marginal"
    else:
        tier = "noise"
    return {"ratio": float(ratio), "tier": tier}
