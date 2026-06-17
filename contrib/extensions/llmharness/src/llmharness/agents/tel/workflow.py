"""TEL 2-pass workflow: note-taking → reasoning/verification → submit.

Pass 1 (note): read spans sequentially, record observations with ⚑ flags.
Pass 2 (reason): receive notes, trace each ⚑ to its origin culprit, submit.

Follows the verifier's propagation_workflow pattern — ``async def run(ctx)``
over ``WorkflowContext``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from agentm.extensions.builtin.workflow import AgentResult, WorkflowContext


class TelWorkflowArgs(TypedDict, total=False):
    question: str
    spans: list[dict[str, Any]]
    notepad_dir: str
    instance_id: str


class TelWorkflowResult(TypedDict, total=False):
    predicted_span_ids: list[str]
    reasoning: str
    note_session_id: str
    reason_session_id: str


_TEL_CTX = "llmharness.agents.tel.context"
_TEL_TOOLS = "llmharness.agents.tel.tools"

_SUBMIT_TOOL = "submit_error_spans"
_TEL_SCENARIO = str(Path(__file__).resolve().parent)


async def run(ctx: WorkflowContext) -> TelWorkflowResult:
    args: TelWorkflowArgs = ctx.args  # type: ignore[assignment]
    question = args["question"]
    spans = args["spans"]
    notepad_dir = args.get("notepad_dir", "")
    instance_id = args.get("instance_id", "unknown")
    n_spans = len(spans)

    # Per-instance notepad dir avoids races when running concurrently.
    inst_notepad_dir = str(Path(notepad_dir) / instance_id) if notepad_dir else ""

    tool_allow = ["list_spans", "get_span", "search_spans", "note", _SUBMIT_TOOL]

    # --- Pass 1: Note-taking ---
    ctx.phase("note")
    ctx.log(f"[{instance_id}] pass 1: note-taking")

    await ctx.agent(
        json.dumps({
            "task": "Read the trajectory and take notes on each span.",
            "question": question,
            "n_spans": n_spans,
        }, ensure_ascii=False),
        scenario=_TEL_SCENARIO,
        trace_label=f"tel_2pass_{instance_id}_note",
        atom_config={
            "tel_context": {
                "question": question,
                "spans": spans,
                "stages": {},
                "prompt_name": "notepad",
            },
            "tel_tools": {"notepad_dir": inst_notepad_dir},
        },
        tool_allowlist=tool_allow,
    )

    # Read notes from the instance-specific notepad directory.
    notes = ""
    if inst_notepad_dir:
        np_dir = Path(inst_notepad_dir)
        if np_dir.is_dir():
            for md in sorted(np_dir.glob("*.md"), key=lambda p: p.stat().st_mtime):
                raw = md.read_text(encoding="utf-8")
                lines = [ln for ln in raw.splitlines() if not ln.startswith("# notepad")]
                notes = "\n".join(lines).strip()
                if notes:
                    break

    if not notes:
        ctx.log(f"[{instance_id}] pass 1 produced no notes — aborting")
        return {"predicted_span_ids": [], "reasoning": "no notes from pass 1"}

    ctx.log(f"[{instance_id}] pass 1 done, {len(notes)} chars of notes")

    # --- Pass 2: Reasoning + Submit ---
    ctx.phase("reason")
    ctx.log(f"[{instance_id}] pass 2: reasoning + submit")

    reason_result: AgentResult = await ctx.agent(
        json.dumps({
            "task": "Review the notes, trace each flagged issue to its origin, "
                    "and submit the culprit spans.",
            "question": question,
            "n_spans": n_spans,
        }, ensure_ascii=False),
        scenario=_TEL_SCENARIO,
        trace_label=f"tel_2pass_{instance_id}_reason",
        atom_config={
            "tel_context": {
                "question": question,
                "spans": spans,
                "stages": {},
                "prompt_name": "reason",
                "prior_notes": notes,
            },
            "tel_tools": {"notepad_dir": inst_notepad_dir},
        },
        tool_allowlist=tool_allow,
    )

    # Parse the structured result if available
    predicted: list[str] = []
    reasoning = ""
    if isinstance(reason_result, dict):
        predicted = list(reason_result.get("error_span_ids", []))
        reasoning = str(reason_result.get("reasoning", ""))
    elif isinstance(reason_result, str):
        reasoning = reason_result

    ctx.log(f"[{instance_id}] pass 2 done, predicted {len(predicted)} spans")

    return {
        "predicted_span_ids": predicted,
        "reasoning": reasoning,
    }
