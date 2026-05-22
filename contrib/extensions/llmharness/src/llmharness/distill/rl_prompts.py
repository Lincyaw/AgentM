"""Emit the online-RL prompt pool from replay records.

One JSONL row per phase firing — prompt-only, no targets. The downstream
trainer (rca-autorl GRPO/PPO) samples from this pool, runs the policy
to produce a rollout, scores it with :mod:`llmharness.train_signals`,
and never sees a teacher trajectory. Locking the row schema here is what
lets the trainer be implemented independently.

Row schema::

    {
      "phase": "extractor" | "auditor",
      "sample_id": "<case>:<firing>:<phase>",
      "source_case_id": "<replay-derived; root_session_id or meta sample_id>",
      "firing_index": <turn_index>,
      "input": {"system": "...", "user": "..."},
      "meta": {"root_session_id": ..., "turn_index": ..., "ts_ns": ...}
    }
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from .export import child_prompt_for_record

Phase = Literal["extractor", "auditor"]

__all__ = ["RlPromptRow", "rl_prompts_from_replay"]


@dataclass(frozen=True)
class RlPromptRow:
    """One prompt-only row for the RL sampling pool."""

    phase: Phase
    sample_id: str
    source_case_id: str
    firing_index: int
    input_system: str
    input_user: str
    meta: dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "phase": self.phase,
                "sample_id": self.sample_id,
                "source_case_id": self.source_case_id,
                "firing_index": self.firing_index,
                "input": {"system": self.input_system, "user": self.input_user},
                "meta": self.meta,
            },
            ensure_ascii=False,
            default=str,
        )


def rl_prompts_from_replay(
    replay_records: Iterable[dict[str, Any]],
    *,
    source_case_id: str,
    phases: tuple[Phase, ...] = ("extractor", "auditor"),
) -> Iterator[RlPromptRow]:
    """Yield one :class:`RlPromptRow` per matching replay record.

    Skips records whose phase isn't in ``phases`` and records whose
    prompt can't be reconstructed (unknown phase / no payload). No
    target / output fields are emitted — this is the prompt pool only.
    """
    wanted = set(phases)
    for rec in replay_records:
        phase = rec.get("phase")
        if phase not in wanted:
            continue
        prompt = child_prompt_for_record(rec)
        if prompt is None:
            continue
        input_system, input_user = prompt
        firing_index = int(rec.get("turn_index") or 0)
        root_session_id = str(rec.get("root_session_id") or "")
        # phase is narrowed by ``wanted ⊆ {"extractor", "auditor"}`` —
        # cast for the typed dataclass.
        assert phase in ("extractor", "auditor")
        sample_id = f"{source_case_id}:firing-{firing_index}:{phase}"
        yield RlPromptRow(
            phase=phase,
            sample_id=sample_id,
            source_case_id=source_case_id,
            firing_index=firing_index,
            input_system=input_system,
            input_user=input_user,
            meta={
                "root_session_id": root_session_id,
                "turn_index": firing_index,
                "ts_ns": rec.get("ts_ns"),
            },
        )
