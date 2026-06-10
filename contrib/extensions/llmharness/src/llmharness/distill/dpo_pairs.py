"""DPO pair schema — locked contract for the rca-autorl DPO loader.

The actual pair-building requires Phase 3 fork-and-continue
(spawning a child rollout from a captured firing-time prompt and
scoring multiple alternative trajectories), which is deferred to a
follow-up PR. We lock the **row schema** now so Track B's DPO loader
can be written against a stable contract.

Row schema (one JSONL line per pair)::

    {
      "phase": "extractor" | "auditor",
      "pair_id": "<case_id>:<firing_index>:<i>vs<j>",
      "source_case_id": "<rcabench case id>",
      "firing_index": <int>,
      "prompt": {"system": "...", "user": "..."},
      "chosen":   {"messages": [<assistant trajectory, same shape as SFT target>]},
      "rejected": {"messages": [<assistant trajectory, same shape as SFT target>]},
      "chosen_score":   <float in [0,1]>,
      "rejected_score": <float in [0,1]>,
      "meta": {<provenance: scoring breakdown, ts_ns, etc.>}
    }

The ``messages`` shape under ``chosen`` / ``rejected`` matches
:class:`~llmharness.distill.export.SftRecord.target_messages` —
multi-turn assistant trajectory with ``<think>`` content and OpenAI-
compatible tool_calls — so a DPO trainer can ingest both halves with
the same tokenizer code path as SFT.

When Phase 3 lands, the pair builder will:

1. Read a control replay bundle + its alternative-trajectory siblings
   (siblings come from fork-and-continue runs scored with the same
   process-reward function in :mod:`llmharness.distill.signals`).
2. For each firing, group alternatives by phase+firing_index.
3. Emit ``C(n,2)`` pairs per firing where ``chosen_score >
   rejected_score`` (filter ties at the caller).
4. Score each side with
   :func:`llmharness.distill.signals.extractor_process_reward` /
   :func:`~llmharness.distill.signals.auditor_process_reward` (composite).

This module is the "where to plug Phase 3 in" sentinel.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

__all__ = ["dpo_pairs_from_outcomes"]


def dpo_pairs_from_outcomes(
    control_bundle: Any,
    alternative_bundles: Iterable[Any],
) -> Iterator[dict[str, Any]]:
    """Emit DPO pair rows from a control bundle + scored alternatives.

    *Not yet implemented.* Phase 3 fork-and-continue is required before
    this can produce real pairs. The signature is locked here so the
    rca-autorl DPO loader can be written against a stable contract.

    Args:
        control_bundle: Path-like or dict-like representation of a
            control replay bundle (the canonical rollout, already
            outcome-annotated via ``annotate-case-outcome``).
        alternative_bundles: An iterable of the same shape, each
            representing one fork-and-continue rerun from the same
            case + firing seed. The bundles must share
            ``(source_case_id, firing_index)`` keys with the control.

    Yields:
        DPO pair dicts shaped per the module docstring schema. Pairs
        with equal scores on both sides are filtered upstream by this
        function (no ``chosen_score == rejected_score`` rows escape).

    Raises:
        NotImplementedError: Always — until Phase 3 lands.
    """
    raise NotImplementedError(
        "dpo_pairs_from_outcomes requires Phase 3 fork-and-continue; "
        "see docstring for the locked pair schema."
    )
    # Make this an iterator type even though we never get past the raise.
    yield {}  # pragma: no cover
