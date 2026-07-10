"""Outcome aggregator (doc §3.2/§3.3/§9.3).

Pure functions over ``EvalUnit`` rows. Per (prefix, treatment) we estimate
Q_t(u); per prefix we derive Δ_t(u), the oracle opportunity G*_t, the bounded
realization G^C_t, and the gap. CONTINUE is the per-trajectory baseline outcome
(DESIGN §5), so Q_t(CONTINUE) is a point and Δ is the treatment posterior shifted
by it.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

from ..model import ContentLevel, EvalUnit

# Levels that count as real intervention opportunity for G* (controls excluded).
_CONTROL_LEVELS = {ContentLevel.CONTINUE, ContentLevel.PLACEBO, ContentLevel.GENERIC}
_DEFAULT_ORACLE_LEVELS = tuple(
    level for level in ContentLevel if level not in _CONTROL_LEVELS
)
# Levels a bounded critic is allowed to realize (no GT). Oracle levels excluded.
_DEFAULT_CRITIC_LEVELS = (
    ContentLevel.GENERIC,
    ContentLevel.TYPE,
    ContentLevel.TYPE_TARGET,
    ContentLevel.EVIDENCE,
)


@dataclass(frozen=True, slots=True)
class CellStat:
    """Q_t(u) for one (prefix, treatment) cell."""

    prefix_id: str
    treatment_id: str
    content_level: ContentLevel
    n: int
    mean_score: float | None
    success_rate: float | None
    delta: float | None
    p_delta_gt_eps: float | None


@dataclass(frozen=True, slots=True)
class PrefixStat:
    """Per-prefix opportunity / realization summary."""

    prefix_id: str
    trajectory_id: str
    case_id: str
    repository_id: str
    progress: float
    stratum: str
    continue_score: float | None
    continue_success: bool | None
    cells: dict[str, CellStat] = field(default_factory=dict)
    g_star: float | None = None
    g_star_treatment: str | None = None
    g_critic: float | None = None
    gap: float | None = None
    min_delta: float | None = None
    oracle_actionable: bool = False
    harm_sensitive: bool = False


@dataclass(frozen=True, slots=True)
class AggregateResult:
    epsilon: float
    prefixes: list[PrefixStat]


def aggregate(
    rows: list[EvalUnit],
    *,
    epsilon: float = 0.0,
    oracle_levels: tuple[ContentLevel, ...] = _DEFAULT_ORACLE_LEVELS,
    critic_levels: tuple[ContentLevel, ...] = _DEFAULT_CRITIC_LEVELS,
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> AggregateResult:
    by_prefix: dict[str, list[EvalUnit]] = {}
    for row in rows:
        by_prefix.setdefault(row.prefix_id, []).append(row)

    prefixes: list[PrefixStat] = []
    for prefix_id, prefix_rows in sorted(by_prefix.items()):
        prefixes.append(
            _aggregate_prefix(
                prefix_id,
                prefix_rows,
                epsilon=epsilon,
                oracle_levels=oracle_levels,
                critic_levels=critic_levels,
                bootstrap_n=bootstrap_n,
                seed=seed,
            )
        )
    return AggregateResult(epsilon=epsilon, prefixes=prefixes)


def undecided_cells(
    result: AggregateResult,
    *,
    band: tuple[float, float] = (0.2, 0.8),
    oracle_levels: tuple[ContentLevel, ...] = _DEFAULT_ORACLE_LEVELS,
) -> list[tuple[str, str]]:
    """Decision-relevant (prefix, treatment) cells whose P(Δ>ε) is undecided.

    Drives adaptive-K (doc §9.3): escalate rollouts only where the oracle-best vs
    CONTINUE call is still uncertain.
    """

    low, high = band
    out: list[tuple[str, str]] = []
    oracle_set = set(oracle_levels)
    for prefix in result.prefixes:
        for cell in prefix.cells.values():
            if cell.content_level not in oracle_set:
                continue
            if cell.p_delta_gt_eps is None:
                continue
            if low <= cell.p_delta_gt_eps <= high:
                out.append((prefix.prefix_id, cell.treatment_id))
    return out


def _aggregate_prefix(
    prefix_id: str,
    rows: list[EvalUnit],
    *,
    epsilon: float,
    oracle_levels: tuple[ContentLevel, ...],
    critic_levels: tuple[ContentLevel, ...],
    bootstrap_n: int,
    seed: int,
) -> PrefixStat:
    first = rows[0]
    cont_rows = [r for r in rows if r.content_level is ContentLevel.CONTINUE]
    continue_score = _mean(
        [r.normalized_score for r in cont_rows if r.normalized_score is not None]
    )
    continue_success = cont_rows[0].binary_success if cont_rows else None

    by_treatment: dict[str, list[EvalUnit]] = {}
    for row in rows:
        if row.content_level is ContentLevel.CONTINUE:
            continue
        if row.status != "succeeded":
            continue
        by_treatment.setdefault(row.treatment_id, []).append(row)

    cells: dict[str, CellStat] = {}
    rng = random.Random(f"{prefix_id}:{seed}")
    for treatment_id, treatment_rows in by_treatment.items():
        scores = [r.normalized_score for r in treatment_rows if r.normalized_score is not None]
        successes = [r.binary_success for r in treatment_rows if r.binary_success is not None]
        mean_score = _mean(scores)
        delta = (
            mean_score - continue_score
            if mean_score is not None and continue_score is not None
            else None
        )
        cells[treatment_id] = CellStat(
            prefix_id=prefix_id,
            treatment_id=treatment_id,
            content_level=treatment_rows[0].content_level,
            n=len(treatment_rows),
            mean_score=mean_score,
            success_rate=_mean([1.0 if s else 0.0 for s in successes]) if successes else None,
            delta=delta,
            p_delta_gt_eps=_p_delta_gt_eps(
                scores, continue_score, epsilon, bootstrap_n, rng
            ),
        )

    oracle_set = set(oracle_levels)
    critic_set = set(critic_levels)
    oracle_deltas = [
        (c.delta, c.treatment_id)
        for c in cells.values()
        if c.content_level in oracle_set and c.delta is not None
    ]
    critic_deltas = [
        c.delta for c in cells.values() if c.content_level in critic_set and c.delta is not None
    ]
    all_deltas = [c.delta for c in cells.values() if c.delta is not None]

    g_star, g_star_treatment = (max(oracle_deltas) if oracle_deltas else (None, None))
    g_critic = max(critic_deltas) if critic_deltas else None
    gap = g_star - g_critic if g_star is not None and g_critic is not None else None
    min_delta = min(all_deltas) if all_deltas else None

    return PrefixStat(
        prefix_id=prefix_id,
        trajectory_id=first.trajectory_id,
        case_id=first.case_id,
        repository_id=first.repository_id,
        progress=first.progress,
        stratum=str(first.metadata.get("stratum", "")),
        continue_score=continue_score,
        continue_success=continue_success,
        cells=cells,
        g_star=g_star,
        g_star_treatment=g_star_treatment,
        g_critic=g_critic,
        gap=gap,
        min_delta=min_delta,
        oracle_actionable=bool(g_star is not None and g_star > epsilon),
        harm_sensitive=bool(min_delta is not None and min_delta < -epsilon),
    )


def _p_delta_gt_eps(
    scores: list[float],
    continue_score: float | None,
    epsilon: float,
    bootstrap_n: int,
    rng: random.Random,
) -> float | None:
    if not scores or continue_score is None:
        return None
    if len(scores) == 1:
        return 1.0 if (scores[0] - continue_score) > epsilon else 0.0
    n = len(scores)
    hits = 0
    for _ in range(bootstrap_n):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        if (statistics.fmean(sample) - continue_score) > epsilon:
            hits += 1
    return hits / bootstrap_n


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None
