"""Rescue Window estimator (doc §4.1).

Assemble each trajectory's G*_t over its sampled prefixes into the temporal
window {t : G*_t > gamma} and report opening/closing time, width, peak, and
opportunity area. Time axis = normalized progress in [0,1].
"""

from __future__ import annotations

from dataclasses import dataclass

from .aggregate import AggregateResult, PrefixStat


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    trajectory_id: str
    case_id: str
    n_points: int
    exists: bool
    t_open: float | None
    t_close: float | None
    width: float | None
    peak: float | None
    peak_progress: float | None
    area: float


def estimate_windows(
    result: AggregateResult, *, gamma: float = 0.0
) -> list[WindowMetrics]:
    by_traj: dict[str, list[PrefixStat]] = {}
    for prefix in result.prefixes:
        by_traj.setdefault(prefix.trajectory_id, []).append(prefix)

    windows: list[WindowMetrics] = []
    for trajectory_id, prefixes in sorted(by_traj.items()):
        ordered = sorted(prefixes, key=lambda p: p.progress)
        points = [(p.progress, p.g_star) for p in ordered if p.g_star is not None]
        in_window = [(prog, g) for prog, g in points if g > gamma]
        peak_progress, peak = (
            max(points, key=lambda item: item[1]) if points else (None, None)
        )
        windows.append(
            WindowMetrics(
                trajectory_id=trajectory_id,
                case_id=ordered[0].case_id,
                n_points=len(points),
                exists=bool(in_window),
                t_open=min(prog for prog, _ in in_window) if in_window else None,
                t_close=max(prog for prog, _ in in_window) if in_window else None,
                width=(
                    max(prog for prog, _ in in_window)
                    - min(prog for prog, _ in in_window)
                    if in_window
                    else None
                ),
                peak=peak,
                peak_progress=peak_progress,
                area=_opportunity_area(points),
            )
        )
    return windows


def _opportunity_area(points: list[tuple[float, float]]) -> float:
    """Trapezoidal integral of max(G*_t, 0) over progress."""

    if len(points) < 2:
        return 0.0
    ordered = sorted(points, key=lambda item: item[0])
    area = 0.0
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        area += (x1 - x0) * (max(y0, 0.0) + max(y1, 0.0)) / 2.0
    return area
