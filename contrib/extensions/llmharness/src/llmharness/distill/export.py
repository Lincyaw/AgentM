"""Convert labeled records → SFT JSONL.

Two outputs:

* ``extractor.jsonl`` — straight from replay records, no oracle needed.
  Input is the extractor system prompt + the recorded payload; target
  is a single Qwen / GLM style assistant message — ``content`` carries
  the teacher's ``<think>...</think>`` reasoning trace recovered from
  the replay sidecar's ``raw_assistant_messages``, and ``tool_calls``
  carries a single ``submit_events`` call whose ``arguments`` are
  rebuilt from the witness-filtered ``output.events`` /
  ``output.edges``. The student learns to emit thinking + the same
  validated event graph the live teacher committed; raw, witness-failed
  tool-call args are not surfaced.

* ``auditor.jsonl`` — from :class:`~llmharness.distill.oracle.LabeledSample`
  rows that were not dropped. Input is the student-visible payload
  (causal snapshot, NO GT); target is a single Qwen / GLM style
  assistant message with empty ``content`` and a single
  ``submit_verdict`` tool call. Auditor thinking persistence is a
  separate follow-up; once it lands, ``content`` here gets the same
  ``<think>...</think>`` wrapper.

Each SFT record carries enough provenance (``sample_id``,
``root_session_id``, ``turn_index``) to back-trace.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..audit.auditor.prompt import (
    DEFAULT_PROMPT_NAME as _AUDITOR_DEFAULT_PROMPT_NAME,
)
from ..audit.auditor.prompt import load_auditor_prompt
from ..audit.auditor.submit_verdict import SUBMIT_VERDICT_TOOL_NAME
from ..audit.extractor.prompt import (
    DEFAULT_PROMPT_NAME as _EXTRACTOR_DEFAULT_PROMPT_NAME,
)
from ..audit.extractor.prompt import load_extractor_prompt
from ..audit.extractor.tools import SUBMIT_EVENTS_TOOL_NAME

Phase = Literal["extractor", "auditor"]


@dataclass(frozen=True)
class SftRecord:
    """One SFT training row in Qwen / GLM chat-template shape.

    ``target_messages`` is a list of assistant messages (today always
    length 1) whose ``content`` holds the teacher's reasoning trace as
    ``<think>...</think>`` and whose ``tool_calls`` holds the
    validated terminal call. Trainers that consume Qwen / GLM tokens
    can pass this list straight to the chat template; the ``<think>``
    block is preserved verbatim and ``arguments`` is a JSON string per
    the OpenAI-compatible tool-call convention.
    """

    phase: Phase
    sample_id: str
    root_session_id: str
    turn_index: int
    input_system: str
    input_user: str
    target_messages: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "phase": self.phase,
                "sample_id": self.sample_id,
                "root_session_id": self.root_session_id,
                "turn_index": self.turn_index,
                "input": {"system": self.input_system, "user": self.input_user},
                "target": {"messages": self.target_messages},
                "meta": self.meta,
            },
            ensure_ascii=False,
            default=str,
        )


def _thinking_text_from_raw(raw_assistant_messages: Any) -> str:
    """Concatenate thinking-block text in chronological order.

    Source rows can be the missing field (older sidecars), a non-list
    (corrupt payload), or a list of dicts. Anything that isn't a
    ``{"type": "thinking", "text": str}`` block is ignored — we don't
    want to surface tool_call args or text blocks here, only reasoning.
    """
    if not isinstance(raw_assistant_messages, list):
        return ""
    parts: list[str] = []
    for blk in raw_assistant_messages:
        if not isinstance(blk, dict):
            continue
        if blk.get("type") != "thinking":
            continue
        text = blk.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def _build_target_messages(
    tool_name: str, arguments: dict[str, Any], *, thinking_text: str
) -> list[dict[str, Any]]:
    """Assemble one Qwen / GLM-style assistant message.

    ``content`` carries the thinking block wrapped in ``<think>`` tags;
    when no thinking was captured it collapses to an empty string so
    the chat template doesn't emit a stray pair of empty tags.
    ``tool_calls`` follows the OpenAI-compatible shape (``arguments``
    serialized to a JSON string) so off-the-shelf trainers work without
    a custom adapter.
    """
    content = (
        f"<think>{thinking_text}</think>\n\n" if thinking_text else ""
    )
    return [
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        }
    ]


# ----- extractor side -------------------------------------------------------


def extractor_records_from_replay(
    replay_records: Iterable[dict[str, Any]],
    *,
    sample_id: str,
) -> Iterator[SftRecord]:
    """Yield one SFT record per ok extractor replay record.

    The extractor's replay ``output`` carries the witness-filtered
    events + edges the teacher actually committed; we re-package them
    as a single ``submit_events`` tool call (the v3.1 shape: events
    with embedded ``refs[]`` derived from edges). The teacher's
    ``<think>...</think>`` reasoning trace is recovered from
    ``raw_assistant_messages`` so the student learns reasoning →
    validated graph, not reasoning → raw (possibly invalid) emit.
    Older sidecars without ``raw_assistant_messages`` yield an empty
    thinking string and an empty ``content`` field — the row is still
    a valid SFT target.
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

        thinking_text = _thinking_text_from_raw(rec.get("raw_assistant_messages"))
        yield SftRecord(
            phase="extractor",
            sample_id=sample_id,
            root_session_id=str(rec.get("root_session_id") or ""),
            turn_index=int(rec.get("turn_index") or 0),
            input_system=load_extractor_prompt(_EXTRACTOR_DEFAULT_PROMPT_NAME),
            input_user=json.dumps(rec.get("payload") or {}, ensure_ascii=False),
            target_messages=_build_target_messages(
                SUBMIT_EVENTS_TOOL_NAME,
                {"events": events_with_refs},
                thinking_text=thinking_text,
            ),
            meta={"replay_ts_ns": rec.get("ts_ns")},
        )


# ----- auditor side ---------------------------------------------------------


def auditor_records_from_labels(
    labeled: Iterable[dict[str, Any]],
) -> Iterator[SftRecord]:
    """Yield SFT records from LabeledSample.to_jsonl()-shaped dicts.

    Dropped samples are skipped (caller can also collect them via
    ``dropped_records_from_labels``). Auditor ``content`` is empty
    pending the auditor-side thinking-persistence work — only the
    ``submit_verdict`` call is supervised today.
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
            input_system=load_auditor_prompt(_AUDITOR_DEFAULT_PROMPT_NAME),
            input_user=json.dumps(
                row.get("input_payload") or {}, ensure_ascii=False
            ),
            target_messages=_build_target_messages(
                SUBMIT_VERDICT_TOOL_NAME,
                {"verdict": target},
                thinking_text="",
            ),
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
