"""Convert labeled records → SFT JSONL.

Two outputs:

* ``extractor.jsonl`` — straight from replay records, no oracle needed.
  Input is the extractor system prompt + the recorded payload; target
  is a **multi-turn** assistant trajectory replayed from the recorded
  tool-call sequence (``upsert_node`` / ``upsert_edge`` /
  ``delete_*`` -> ``finalize_extraction``). Each
  assistant message carries ``<think>`` wrapping reconstructed from the
  matching block of ``raw_assistant_messages``. The student learns to
  drive an incremental graph build, not to dump one batch — that was
  the v18 single-shot ``submit_events_batch`` shape, which produced bad
  SFT signal because witness retries dominated the trajectory.

  Replays from the legacy v18 ``submit_events_batch`` toolset are
  skipped (a counter is reported on completion) — under the v19 tool
  surface those tool names are invalid and rebuilding them would
  miscondition the student.

* ``auditor.jsonl`` — from :class:`~llmharness.distill.oracle.LabeledSample`
  rows that were not dropped. Input is the student-visible payload
  (causal snapshot, NO GT); target is a single Qwen / GLM style
  assistant message with empty ``content`` and a single
  ``submit_verdict`` tool call. Auditor thinking persistence is a
  separate follow-up; once it lands, ``content`` here gets the same
  ``<think>...</think>`` wrapper.

Each SFT record carries enough provenance (``sample_id``,
``session_id``, ``turn_index``) to back-trace.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llmharness.agents.auditor.prompt import load_auditor_prompt
from llmharness.agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
from llmharness.agents.extractor.prompt import load_extractor_prompt
from llmharness.agents.extractor.tools import (
    UPSERT_NODE_TOOL_NAME,
    UPSERT_EDGE_TOOL_NAME,
    DELETE_NODE_TOOL_NAME,
    DELETE_EDGE_TOOL_NAME,
    RESET_EXTRACTION_TOOL_NAME,
    FINALIZE_EXTRACTION_TOOL_NAME,
)

# Tool names the extractor can call — used to filter valid steps in SFT export.
EXTRACTOR_TOOL_NAMES: frozenset[str] = frozenset({
    UPSERT_NODE_TOOL_NAME,
    UPSERT_EDGE_TOOL_NAME,
    DELETE_NODE_TOOL_NAME,
    DELETE_EDGE_TOOL_NAME,
    RESET_EXTRACTION_TOOL_NAME,
    FINALIZE_EXTRACTION_TOOL_NAME,
})

# Default prompt names — the new modules don't export DEFAULT_PROMPT_NAME
# as a constant; use the string directly.
_EXTRACTOR_DEFAULT_PROMPT_NAME = "default"
_AUDITOR_DEFAULT_PROMPT_NAME = "minimal"

Phase = Literal["extractor", "auditor"]


# v18 legacy tool names — if a replay record carries any of these, the
# tool sequence is invalid under v19 and the record gets skipped (not
# rebuilt) so we never train the student on a deleted tool surface.
_LEGACY_BATCH_TOOL_NAMES = frozenset({"submit_events", "submit_events_batch"})


@dataclass(frozen=True)
class SftRecord:
    """One SFT training row in Qwen / GLM chat-template shape.

    ``target_messages`` is a list of assistant messages. After the v19
    extractor refactor an extractor row is a **multi-turn trajectory**:
    one assistant message per recorded tool call, each with its own
    ``<think>`` block reconstructed from ``raw_assistant_messages``
    (thinking blocks that sit immediately before the matching
    tool_call). Auditor rows stay single-turn for now.

    Each ``tool_calls[*].function.arguments`` is a JSON string per the
    OpenAI-compatible tool-call convention so off-the-shelf trainers
    work without a custom adapter.
    """

    phase: Phase
    sample_id: str
    session_id: str
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
                "session_id": self.session_id,
                "turn_index": self.turn_index,
                "input": {"system": self.input_system, "user": self.input_user},
                "target": {"messages": self.target_messages},
                "meta": self.meta,
            },
            ensure_ascii=False,
            default=str,
        )


def _assistant_message(
    *, thinking_text: str, tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Assemble one Qwen / GLM-style assistant message.

    ``content`` carries the thinking block wrapped in ``<think>`` tags;
    when no thinking was captured it collapses to an empty string so the
    chat template doesn't emit a stray pair of empty tags. ``tool_calls``
    follows the OpenAI-compatible shape (``arguments`` serialized to a
    JSON string).
    """
    content = f"<think>{thinking_text}</think>\n\n" if thinking_text else ""
    return {
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


def _split_into_steps(
    raw_blocks: list[Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Walk the flattened block list, yielding ``(thinking, tool_call)``.

    The replay sidecar's ``raw_assistant_messages`` is the chronological
    flattening of every assistant message's content blocks (see
    ``_flatten_assistant_blocks`` in the adapter): thinking blocks,
    tool_call blocks, and free-text blocks interleaved. For multi-turn
    SFT we slice it into one step per tool_call, attributing the
    immediately preceding thinking block(s) to that step.

    Returns a list of ``(thinking_concat, tool_call_dict)`` tuples.
    ``tool_call_dict`` carries ``name`` and ``arguments`` (already
    parsed). Blocks of unknown type are ignored — they don't affect
    the SFT shape.
    """
    steps: list[tuple[str, dict[str, Any]]] = []
    pending_thinking: list[str] = []
    for blk in raw_blocks:
        if not isinstance(blk, dict):
            continue
        btype = blk.get("type")
        if btype == "thinking":
            text = blk.get("text")
            if isinstance(text, str) and text:
                pending_thinking.append(text)
        elif btype == "tool_call":
            name = blk.get("name")
            args = blk.get("arguments")
            if not isinstance(name, str):
                # Malformed block — drop the thinking with it so we
                # don't attribute it to the next tool call by mistake.
                pending_thinking = []
                continue
            steps.append(
                (
                    "".join(pending_thinking),
                    {"name": name, "arguments": args if isinstance(args, dict) else {}},
                )
            )
            pending_thinking = []
        # text / other blocks: ignore (the prompt input is the carrier;
        # final SFT content is reconstructed from thinking only).
    return steps


# ----- shared prompt-extraction helpers -------------------------------------


def child_prompt_for_record(rec: dict[str, Any]) -> tuple[str, str] | None:
    """Reconstruct the (system, user) pair the child saw at firing time.

    Returns ``None`` when the record's phase isn't recognized. The system
    prompt is the canonical phase prompt — same string that the live
    extractor / auditor child sees (so SFT and RL prompt pools agree).
    The user prompt is the JSON-serialized ``payload`` field of the
    replay record.

    Shared by both SFT export (:func:`extractor_records_from_replay`)
    and the RL-prompt-pool exporter so the two never drift apart.
    """
    phase = rec.get("phase")
    if phase == "extractor":
        system = load_extractor_prompt(_EXTRACTOR_DEFAULT_PROMPT_NAME)
    elif phase == "auditor":
        system = load_auditor_prompt(_AUDITOR_DEFAULT_PROMPT_NAME)
    else:
        return None
    user = json.dumps(rec.get("payload") or {}, ensure_ascii=False)
    return system, user


# ----- extractor side -------------------------------------------------------


def extractor_records_from_replay(
    replay_records: Iterable[dict[str, Any]],
    *,
    sample_id: str,
) -> Iterator[SftRecord]:
    """Yield one SFT record per ok extractor replay record.

    Drops three kinds of records:

    * non-extractor / non-ok records (status != "ok"),
    * legacy records whose tool sequence contains any v18 batch tool
      name (``submit_events`` / ``submit_events_batch``) — those rebuild
      a deleted tool surface and would miscondition the student. Use
      :func:`legacy_batch_replay_count` to count them ahead of an
      export run.
    * records with no recorded tool calls at all (e.g. spawn_error /
      empty firings) — the student has nothing to learn from an empty
      target trajectory.

    The recorded tool sequence becomes a multi-turn assistant
    trajectory: each tool_call is its own assistant message carrying
    the matching thinking block(s) wrapped in ``<think>``.
    """
    for rec in replay_records:
        if rec.get("phase") != "extractor":
            continue
        if rec.get("status") != "ok":
            continue
        raw_blocks = rec.get("raw_assistant_messages")
        if not isinstance(raw_blocks, list) or not raw_blocks:
            continue
        steps = _split_into_steps(raw_blocks)
        if not steps:
            continue
        # Legacy filter — any v18 batch tool name invalidates the
        # whole trajectory under v19.
        if any(call["name"] in _LEGACY_BATCH_TOOL_NAMES for _think, call in steps):
            continue
        # Optional sanity: ignore steps that call unknown tool names so
        # a malformed sidecar can't slip through.
        valid_steps = [s for s in steps if s[1]["name"] in EXTRACTOR_TOOL_NAMES]
        if not valid_steps:
            continue
        target_messages = [
            _assistant_message(
                thinking_text=thinking,
                tool_name=call["name"],
                arguments=call["arguments"],
            )
            for thinking, call in valid_steps
        ]
        prompt = child_prompt_for_record(rec)
        if prompt is None:
            continue
        input_system, input_user = prompt
        yield SftRecord(
            phase="extractor",
            sample_id=sample_id,
            session_id=str(rec.get("session_id") or ""),
            turn_index=int(rec.get("turn_index") or 0),
            input_system=input_system,
            input_user=input_user,
            target_messages=target_messages,
            meta={"replay_ts_ns": rec.get("ts_ns")},
        )


def legacy_batch_replay_count(
    replay_records: Iterable[dict[str, Any]],
) -> int:
    """Count replay records that would be skipped as legacy-batch.

    Callers exposing a CLI can print this so the operator notices when
    their corpus is dominated by pre-v19 replays — those need a
    re-collection under the new tool surface before SFT is useful.
    """
    count = 0
    for rec in replay_records:
        if rec.get("phase") != "extractor":
            continue
        raw_blocks = rec.get("raw_assistant_messages")
        if not isinstance(raw_blocks, list):
            continue
        for blk in raw_blocks:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "tool_call":
                continue
            if blk.get("name") in _LEGACY_BATCH_TOOL_NAMES:
                count += 1
                break
    return count


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
            session_id=str(row.get("session_id") or ""),
            turn_index=int(row.get("turn_index") or 0),
            input_system=load_auditor_prompt(_AUDITOR_DEFAULT_PROMPT_NAME),
            input_user=json.dumps(row.get("input_payload") or {}, ensure_ascii=False),
            target_messages=[
                _assistant_message(
                    thinking_text="",
                    tool_name=SUBMIT_VERDICT_TOOL_NAME,
                    arguments={"verdict": target},
                )
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
            "session_id": row.get("session_id"),
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
    "child_prompt_for_record",
    "dropped_records_from_labels",
    "extractor_records_from_replay",
    "legacy_batch_replay_count",
    "write_jsonl",
]
