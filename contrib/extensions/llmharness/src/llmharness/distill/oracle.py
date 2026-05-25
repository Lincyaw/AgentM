"""Two-stage labeling orchestrator (offline).

Stage A — oracle child (sees GT) — produces a selection over `findings`
and matched event ids. Stage B — rewriter child (NO GT) — checks
graph-only justifiability and emits the final `reminder_text`. If
the rewriter rejects the selection, the sample is dropped.

Both children run as **top-level** AgentM sessions via
:func:`llmharness.tools.engine.run_phase_standalone` with a minimal
four-atom extension list (observability + operations_local +
submit_tool + system_prompt). No cards, no skills.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audit.toolkit.atom_constants import (
    OBSERVABILITY_MODULE,
    OPERATIONS_MODULE,
    SYSTEM_PROMPT_MODULE,
)
from ..schema import Edge, Event, Finding
from ..tools.engine import run_phase_standalone
from ._submit_oracle import SUBMIT_ORACLE_TOOL_NAME
from ._submit_rewriter import SUBMIT_REWRITE_TOOL_NAME
from .causal import CausalSnapshot, causal_mask
from .gt import GroundTruth

_logger = logging.getLogger(__name__)

_ORACLE_SUBMIT_MODULE = "llmharness.distill._submit_oracle"
_REWRITER_SUBMIT_MODULE = "llmharness.distill._submit_rewriter"

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _read_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _minimal_extensions(
    *, submit_module: str, system_prompt: str
) -> list[tuple[str, dict[str, Any]]]:
    """Four-atom audit-child surface — same shape as the live audit children
    but stripped of cards / skills. observability (now also the sole OTLP
    span/log source via Event.to_otel) + operations (substrate freeze) +
    submit + system_prompt.
    """
    return [
        (OBSERVABILITY_MODULE, {}),
        (OPERATIONS_MODULE, {}),
        (submit_module, {}),
        (SYSTEM_PROMPT_MODULE, {"prompt": system_prompt}),
    ]


# ----- dataclasses ----------------------------------------------------------


@dataclass(frozen=True)
class OracleLabel:
    selected_finding_indices: tuple[int, ...]
    matched_event_ids: tuple[int, ...]
    rationale_with_gt: str
    continuation_notes: tuple[str, ...]


@dataclass(frozen=True)
class RewriterOutput:
    justifiable_from_graph: bool
    reminder_text: str
    drop_reason: str
    matched_event_ids: tuple[int, ...]


@dataclass(frozen=True)
class LabeledSample:
    """One auditor firing, fully labeled (or dropped).

    ``input_payload`` is the JSON payload the **student auditor** will
    see at inference — it carries the snapshot but NOT the GT.
    ``target_verdict`` is the V2 ``Verdict``-shaped dict the student is
    trained to produce.
    """

    sample_id: str
    root_session_id: str
    turn_index: int
    input_payload: dict[str, Any]
    target_verdict: dict[str, Any]
    oracle: dict[str, Any]
    rewriter: dict[str, Any]
    drop: bool
    drop_reason: str
    gt_meta: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "sample_id": self.sample_id,
                "root_session_id": self.root_session_id,
                "turn_index": self.turn_index,
                "input_payload": self.input_payload,
                "target_verdict": self.target_verdict,
                "oracle": self.oracle,
                "rewriter": self.rewriter,
                "drop": self.drop,
                "drop_reason": self.drop_reason,
                "gt_meta": self.gt_meta,
            },
            ensure_ascii=False,
            default=str,
        )


# ----- snapshot reconstruction from a replay record -------------------------


def _events_from_replay_compose(compose_kwargs: dict[str, Any]) -> list[Event]:
    out: list[Event] = []
    for raw in compose_kwargs.get("events") or []:
        if not isinstance(raw, dict):
            continue
        try:
            out.append(Event.from_dict(raw))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _edges_from_replay_compose(compose_kwargs: dict[str, Any]) -> list[Edge]:
    out: list[Edge] = []
    for raw in compose_kwargs.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        try:
            out.append(Edge.from_dict(raw))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _findings_from_replay_compose(compose_kwargs: dict[str, Any]) -> list[Finding]:
    out: list[Finding] = []
    for raw in compose_kwargs.get("findings") or []:
        if not isinstance(raw, dict):
            continue
        try:
            out.append(Finding.from_dict(raw))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def snapshot_from_replay(record: dict[str, Any]) -> CausalSnapshot:
    """Reconstruct the causal snapshot for one auditor replay record.

    Replay record schema is defined by
    :class:`llmharness.replay.record.ReplayRecord` — ``compose_kwargs``
    holds the inputs originally passed to
    :func:`compose_auditor_extensions`, ``payload`` holds the user-
    message JSON, and the auditor firing's turn_index is recorded at
    record level.
    """
    compose = record.get("compose_kwargs") or {}
    payload = record.get("payload") or {}
    turn_index = int(record.get("turn_index") or 0)

    events = _events_from_replay_compose(compose)
    edges = _edges_from_replay_compose(compose)
    findings = _findings_from_replay_compose(compose)

    # `trajectory_snapshot` is what the live auditor saw; it's a list of
    # per-turn dicts with an ``index`` key. Fall back to payload graph
    # for compatibility.
    trajectory_raw = compose.get("trajectory_snapshot")
    if not isinstance(trajectory_raw, list):
        trajectory_raw = payload.get("trajectory") or []
    trajectory = [t for t in trajectory_raw if isinstance(t, dict)]

    return causal_mask(
        turn_index=turn_index,
        events=events,
        edges=edges,
        findings=findings,
        trajectory=trajectory,
    )


# ----- oracle + rewriter spawn ---------------------------------------------


async def run_oracle(
    *,
    cwd: str,
    snapshot: CausalSnapshot,
    gt: GroundTruth,
    provider: tuple[str, dict[str, Any]] | None,
) -> OracleLabel | None:
    extensions = _minimal_extensions(
        submit_module=_ORACLE_SUBMIT_MODULE,
        system_prompt=_read_prompt("oracle.md"),
    )
    payload = {
        "snapshot": snapshot.to_prompt_payload(),
        "ground_truth": {
            "root_causes": list(gt.root_causes),
            "fault_type": gt.fault_type,
            "fault_category": gt.fault_category,
        },
    }
    result = await run_phase_standalone(
        cwd=cwd,
        extensions=extensions,
        provider=provider,
        payload=payload,
        terminal_tool=SUBMIT_ORACLE_TOOL_NAME,
        purpose="distill_oracle",
    )
    if result.status != "ok" or not isinstance(result.output, dict):
        _logger.warning(
            "oracle phase non-ok: status=%s error=%s", result.status, result.error
        )
        return None
    args = result.output
    return OracleLabel(
        selected_finding_indices=tuple(
            int(i) for i in args.get("selected_finding_indices") or []
        ),
        matched_event_ids=tuple(int(i) for i in args.get("matched_event_ids") or []),
        rationale_with_gt=str(args.get("rationale_with_gt") or ""),
        continuation_notes=tuple(
            str(n) for n in args.get("continuation_notes") or []
        ),
    )


async def run_rewriter(
    *,
    cwd: str,
    snapshot: CausalSnapshot,
    label: OracleLabel,
    provider: tuple[str, dict[str, Any]] | None,
) -> RewriterOutput | None:
    extensions = _minimal_extensions(
        submit_module=_REWRITER_SUBMIT_MODULE,
        system_prompt=_read_prompt("rewriter.md"),
    )
    payload = {
        "snapshot": snapshot.to_prompt_payload(),
        "selected_finding_indices": list(label.selected_finding_indices),
        "matched_event_ids": list(label.matched_event_ids),
    }
    result = await run_phase_standalone(
        cwd=cwd,
        extensions=extensions,
        provider=provider,
        payload=payload,
        terminal_tool=SUBMIT_REWRITE_TOOL_NAME,
        purpose="distill_rewriter",
    )
    if result.status != "ok" or not isinstance(result.output, dict):
        _logger.warning(
            "rewriter phase non-ok: status=%s error=%s", result.status, result.error
        )
        return None
    args = result.output
    return RewriterOutput(
        justifiable_from_graph=bool(args.get("justifiable_from_graph")),
        reminder_text=str(args.get("reminder_text") or ""),
        drop_reason=str(args.get("drop_reason") or ""),
        matched_event_ids=tuple(int(i) for i in args.get("matched_event_ids") or []),
    )


# ----- per-record driver ----------------------------------------------------


async def label_auditor_record(
    *,
    cwd: str,
    record: dict[str, Any],
    gt: GroundTruth,
    sample_id: str,
    oracle_provider: tuple[str, dict[str, Any]] | None,
    rewriter_provider: tuple[str, dict[str, Any]] | None,
) -> LabeledSample:
    snapshot = snapshot_from_replay(record)
    turn_index = snapshot.turn_index
    root_session_id = str(record.get("root_session_id") or "")
    gt_meta = {"fault_type": gt.fault_type, "fault_category": gt.fault_category}

    # student-visible payload — explicitly NO GT, NO trajectory turns
    # beyond t (already enforced by causal_mask but documented here).
    input_payload = {
        "snapshot": snapshot.to_prompt_payload(),
    }

    label = await run_oracle(
        cwd=cwd, snapshot=snapshot, gt=gt, provider=oracle_provider
    )
    if label is None:
        return LabeledSample(
            sample_id=sample_id,
            root_session_id=root_session_id,
            turn_index=turn_index,
            input_payload=input_payload,
            target_verdict={},
            oracle={},
            rewriter={},
            drop=True,
            drop_reason="oracle_no_call",
            gt_meta=gt_meta,
        )

    # Stay-silent shortcut: oracle picked nothing; no rewriter needed.
    if not label.selected_finding_indices:
        target = {
            "surface_reminder": False,
            "reminder_text": "",
            "continuation_notes": list(label.continuation_notes),
            "matched_event_ids": [],
        }
        return LabeledSample(
            sample_id=sample_id,
            root_session_id=root_session_id,
            turn_index=turn_index,
            input_payload=input_payload,
            target_verdict=target,
            oracle={
                "selected_finding_indices": [],
                "matched_event_ids": [],
                # rationale_with_gt deliberately stored only here (not in
                # input_payload) so it never leaks into training inputs.
                "rationale_with_gt": label.rationale_with_gt,
                "continuation_notes": list(label.continuation_notes),
            },
            rewriter={},
            drop=False,
            drop_reason="",
            gt_meta=gt_meta,
        )

    rewrite = await run_rewriter(
        cwd=cwd, snapshot=snapshot, label=label, provider=rewriter_provider
    )
    if rewrite is None:
        return LabeledSample(
            sample_id=sample_id,
            root_session_id=root_session_id,
            turn_index=turn_index,
            input_payload=input_payload,
            target_verdict={},
            oracle={
                "selected_finding_indices": list(label.selected_finding_indices),
                "matched_event_ids": list(label.matched_event_ids),
                "rationale_with_gt": label.rationale_with_gt,
                "continuation_notes": list(label.continuation_notes),
            },
            rewriter={},
            drop=True,
            drop_reason="rewriter_no_call",
            gt_meta=gt_meta,
        )

    if not rewrite.justifiable_from_graph:
        return LabeledSample(
            sample_id=sample_id,
            root_session_id=root_session_id,
            turn_index=turn_index,
            input_payload=input_payload,
            target_verdict={},
            oracle={
                "selected_finding_indices": list(label.selected_finding_indices),
                "matched_event_ids": list(label.matched_event_ids),
                "rationale_with_gt": label.rationale_with_gt,
                "continuation_notes": list(label.continuation_notes),
            },
            rewriter={
                "justifiable_from_graph": False,
                "reminder_text": rewrite.reminder_text,
                "drop_reason": rewrite.drop_reason,
                "matched_event_ids": list(rewrite.matched_event_ids),
            },
            drop=True,
            drop_reason=rewrite.drop_reason or "rewriter_rejected",
            gt_meta=gt_meta,
        )

    matched = rewrite.matched_event_ids or label.matched_event_ids
    target = {
        "surface_reminder": True,
        "reminder_text": rewrite.reminder_text,
        "continuation_notes": list(label.continuation_notes),
        "matched_event_ids": list(matched),
    }
    return LabeledSample(
        sample_id=sample_id,
        root_session_id=root_session_id,
        turn_index=turn_index,
        input_payload=input_payload,
        target_verdict=target,
        oracle={
            "selected_finding_indices": list(label.selected_finding_indices),
            "matched_event_ids": list(label.matched_event_ids),
            "rationale_with_gt": label.rationale_with_gt,
            "continuation_notes": list(label.continuation_notes),
        },
        rewriter={
            "justifiable_from_graph": True,
            "reminder_text": rewrite.reminder_text,
            "drop_reason": "",
            "matched_event_ids": list(rewrite.matched_event_ids),
        },
        drop=False,
        drop_reason="",
        gt_meta=gt_meta,
    )


__all__ = [
    "LabeledSample",
    "OracleLabel",
    "RewriterOutput",
    "label_auditor_record",
    "run_oracle",
    "run_rewriter",
    "snapshot_from_replay",
]
