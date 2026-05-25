"""Emit the online-RL training pool from replay records.

One JSONL row per phase firing — stripped :class:`ReplayRecord` shape, no
teacher targets. The downstream trainer (rca-autorl GRPO/PPO) samples a
row, re-hydrates it via :meth:`ReplayRecord.from_dict`, feeds it straight
into :func:`replay_extractor_record` / :func:`replay_auditor_record` to
produce a fresh rollout, and scores the rollout with
:mod:`llmharness.train_signals`. The trainer never sees the teacher's
output trajectory.

Stripped fields (would leak teacher behavior into the policy's input)::

    output                    - the teacher's final submit_* args
    status                    - teacher rollout terminal status
    error                     - teacher rollout error string
    latency_ms                - teacher wall-clock
    raw_assistant_messages    - teacher's content blocks

Every other ReplayRecord field is preserved: ``phase``, ``turn_index``,
``session_id``, ``trace_id``, ``ts_ns``, ``compose_kwargs``, ``payload``,
``provider``, ``extras``. The result is ready for
``ReplayRecord.from_dict(row)`` and is the **public contract** consumed by
rca-autorl from this commit forward.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any, Literal

from ..replay.record import ReplayRecord

Phase = Literal["extractor", "auditor"]

__all__ = ["STRIPPED_FIELDS", "rl_rows_from_replay"]

#: Field names removed from each ReplayRecord before emission. Locked as
#: part of the public contract — changing this set is a breaking schema
#: change for downstream consumers.
STRIPPED_FIELDS: frozenset[str] = frozenset(
    {
        "output",
        "status",
        "error",
        "latency_ms",
        "raw_assistant_messages",
    }
)


def _strip(record_dict: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record_dict.items() if k not in STRIPPED_FIELDS}


def rl_rows_from_replay(
    replay_records: Iterable[dict[str, Any]],
    *,
    phases: tuple[Phase, ...] = ("extractor", "auditor"),
) -> Iterator[dict[str, Any]]:
    """Yield stripped-ReplayRecord dicts for every matching record.

    Skips records whose phase isn't in ``phases``. The yielded dict is
    JSON-serializable and round-trips via ``ReplayRecord.from_dict``.
    """
    wanted = set(phases)
    for rec in replay_records:
        phase = rec.get("phase")
        if phase not in wanted:
            continue
        yield _strip(rec)


def serialize_row(row: dict[str, Any]) -> str:
    """JSON-encode one rl row (the on-disk JSONL representation)."""
    return json.dumps(row, ensure_ascii=False, default=str)


def hydrate_row(row: dict[str, Any]) -> ReplayRecord:
    """Re-hydrate an rl-prompts row into a ``ReplayRecord``.

    Fills stripped fields with neutral defaults (``status='ok'``, empty
    output/error/messages) so the result is a structurally-valid record
    that can be passed straight into ``replay_extractor_record`` /
    ``replay_auditor_record`` for a fresh policy rollout.
    """
    filled = dict(row)
    filled.setdefault("output", None)
    filled.setdefault("status", "ok")
    filled.setdefault("error", None)
    filled.setdefault("latency_ms", 0)
    filled.setdefault("raw_assistant_messages", [])
    return ReplayRecord.from_dict(filled)
