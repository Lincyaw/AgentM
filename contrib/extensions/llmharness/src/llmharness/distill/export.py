"""Convert labeled records → SFT JSONL.

Two outputs:

* ``extractor.jsonl`` — straight from replay records, no oracle needed.
  Input is the extractor system prompt + the recorded payload; target
  is the tool-call sequence the replay captured (``register_event`` /
  ``add_edge`` / ``submit_extraction``). The extractor-quality bar is
  already met by the live teacher, so we re-use those trajectories.

* ``auditor.jsonl`` — from :class:`~llmharness.distill.oracle.LabeledSample`
  rows that were not dropped. Input is the student-visible payload
  (causal snapshot, NO GT); target is a single ``submit_verdict`` call
  with the rewriter-approved fields.

Each SFT record carries enough provenance (``sample_id``,
``root_session_id``, ``turn_index``) to back-trace.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..audit.auditor.prompt import AUDITOR_SYSTEM_PROMPT
from ..audit.auditor.submit_tool import SUBMIT_VERDICT_TOOL_NAME
from ..audit.extractor.prompt import EXTRACTOR_SYSTEM_PROMPT
from ..audit.extractor.tools import SUBMIT_EVENTS_TOOL_NAME

Phase = Literal["extractor", "auditor"]


@dataclass(frozen=True)
class SftRecord:
    phase: Phase
    sample_id: str
    root_session_id: str
    turn_index: int
    input_system: str
    input_user: str
    target_tool_calls: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "phase": self.phase,
                "sample_id": self.sample_id,
                "root_session_id": self.root_session_id,
                "turn_index": self.turn_index,
                "input": {"system": self.input_system, "user": self.input_user},
                "target": {"tool_calls": self.target_tool_calls},
                "meta": self.meta,
            },
            ensure_ascii=False,
            default=str,
        )


# ----- extractor side -------------------------------------------------------


def extractor_records_from_replay(
    replay_records: Iterable[dict[str, Any]],
    *,
    sample_id: str,
) -> Iterator[SftRecord]:
    """Yield one SFT record per ok extractor replay record.

    The extractor's replay ``output`` carries the structured events +
    edges that the teacher LLM produced. We re-package them as a
    single ``submit_events`` tool call — the same shape v3.1 of the
    extractor protocol uses (events with embedded ``refs[]`` derived
    from edges).
    """
    for rec in replay_records:
        if rec.get("phase") != "extractor":
            continue
        if rec.get("status") != "ok":
            continue
        out = rec.get("output") or {}
        events = out.get("events") or []
        edges = out.get("edges") or []
        if not events and not edges:
            # Trivial-window extraction; skip — student doesn't need to
            # learn "submit empty".
            continue
        # Re-attach edges as refs[] on events so the SFT target matches
        # the v3.1 ``submit_events`` payload shape (one terminal call).
        refs_by_src: dict[int, list[dict[str, Any]]] = {}
        for ed in edges:
            src = ed.get("src")
            if not isinstance(src, int):
                continue
            refs_by_src.setdefault(src, []).append(
                {
                    "dst": ed.get("dst"),
                    "kind": ed.get("kind"),
                    "reason": ed.get("reason"),
                    "src_turns": ed.get("src_turns"),
                    "dst_turns": ed.get("dst_turns"),
                    "cited_entities": ed.get("cited_entities"),
                    "cited_quote": ed.get("cited_quote"),
                }
            )
        events_with_refs = []
        for ev in events:
            ev_copy = dict(ev)
            ev_id = ev_copy.get("id")
            ev_copy["refs"] = (
                refs_by_src.get(ev_id, []) if isinstance(ev_id, int) else []
            )
            events_with_refs.append(ev_copy)

        yield SftRecord(
            phase="extractor",
            sample_id=sample_id,
            root_session_id=str(rec.get("root_session_id") or ""),
            turn_index=int(rec.get("turn_index") or 0),
            input_system=EXTRACTOR_SYSTEM_PROMPT,
            input_user=json.dumps(rec.get("payload") or {}, ensure_ascii=False),
            target_tool_calls=[
                {
                    "name": SUBMIT_EVENTS_TOOL_NAME,
                    "arguments": {"events": events_with_refs},
                }
            ],
            meta={"replay_ts_ns": rec.get("ts_ns")},
        )


# ----- auditor side ---------------------------------------------------------


def auditor_records_from_labels(
    labeled: Iterable[dict[str, Any]],
) -> Iterator[SftRecord]:
    """Yield SFT records from LabeledSample.to_jsonl()-shaped dicts.

    Dropped samples are skipped (caller can also collect them via
    ``dropped_records_from_labels``).
    """
    for row in labeled:
        if row.get("drop"):
            continue
        target = row.get("target_verdict") or {}
        if not target:
            continue
        yield SftRecord(
            phase="auditor",
            sample_id=str(row.get("sample_id") or ""),
            root_session_id=str(row.get("root_session_id") or ""),
            turn_index=int(row.get("turn_index") or 0),
            input_system=AUDITOR_SYSTEM_PROMPT,
            input_user=json.dumps(
                row.get("input_payload") or {}, ensure_ascii=False
            ),
            target_tool_calls=[
                {
                    "name": SUBMIT_VERDICT_TOOL_NAME,
                    "arguments": {"verdict": target},
                }
            ],
            meta=dict(row.get("gt_meta") or {}),
        )


def dropped_records_from_labels(
    labeled: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Yield drop records for the audit trail (NOT included in SFT)."""
    for row in labeled:
        if not row.get("drop"):
            continue
        yield {
            "sample_id": row.get("sample_id"),
            "root_session_id": row.get("root_session_id"),
            "turn_index": row.get("turn_index"),
            "drop_reason": row.get("drop_reason"),
            "oracle": row.get("oracle"),
            "rewriter": row.get("rewriter"),
        }


# ----- writer ---------------------------------------------------------------


def write_jsonl(path: Path, records: Iterable[SftRecord | dict[str, Any]]) -> int:
    """Write records to JSONL. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            if isinstance(rec, SftRecord):
                fh.write(rec.to_jsonl())
            else:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str))
            fh.write("\n")
            count += 1
    return count


__all__ = [
    "SftRecord",
    "auditor_records_from_labels",
    "dropped_records_from_labels",
    "extractor_records_from_replay",
    "write_jsonl",
]
