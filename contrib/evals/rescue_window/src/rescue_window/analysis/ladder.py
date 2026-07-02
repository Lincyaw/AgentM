"""Rescuability Ladder (doc §4.2), RCA-scoped.

In pure-diagnosis RCA several rungs collapse (state is read-only), so the
load-bearing decomposition is CHANNEL vs ACTOR:

  CHANNEL = best Δ a typed, channel-respecting nudge achieves (the content ladder
            and ORACLE_GROUNDED).
  ACTOR   = best Δ when the answer is handed over directly (ORACLE_DIAG).

A large ACTOR-over-CHANNEL gap means the natural-language channel is the
bottleneck, not the actor's recovery ability.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ContentLevel, LadderRung
from .aggregate import AggregateResult, PrefixStat

_RUNG_OF: dict[ContentLevel, LadderRung] = {
    ContentLevel.ORACLE_DIAG: LadderRung.ACTOR,
}


def _rung_for(level: ContentLevel) -> LadderRung:
    return _RUNG_OF.get(level, LadderRung.CHANNEL)


@dataclass(frozen=True, slots=True)
class PrefixLadder:
    prefix_id: str
    rung_best: dict[str, float]
    channel_to_actor_gap: float | None


def ladder_gaps(result: AggregateResult) -> list[PrefixLadder]:
    out: list[PrefixLadder] = []
    for prefix in result.prefixes:
        rung_best = _rung_best(prefix)
        channel = rung_best.get(LadderRung.CHANNEL.value)
        actor = rung_best.get(LadderRung.ACTOR.value)
        gap = actor - channel if channel is not None and actor is not None else None
        out.append(
            PrefixLadder(
                prefix_id=prefix.prefix_id,
                rung_best=rung_best,
                channel_to_actor_gap=gap,
            )
        )
    return out


def _rung_best(prefix: PrefixStat) -> dict[str, float]:
    best: dict[str, float] = {}
    for cell in prefix.cells.values():
        if cell.delta is None:
            continue
        rung = _rung_for(cell.content_level).value
        if rung not in best or cell.delta > best[rung]:
            best[rung] = cell.delta
    return best
