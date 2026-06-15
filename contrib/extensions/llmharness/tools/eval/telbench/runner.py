"""TELBench evaluation driver.

Runs the llmharness cognitive-audit pipeline (extractor + auditor) over
TELBench instances and scores the resulting error-span predictions
against gold annotations.

Three modes:

* **posthoc** — all spans fed as a complete trajectory; extractor fires
  once at the end, auditor fires once on the full graph.
* **online** — spans fed incrementally; extractor + auditor fire every K
  spans; verdicts accumulate across all firings.
* **tel** — direct TEL agent with spanstore + notepad tools; no
  extractor/auditor indirection.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from agentm.core.abi import (
    AgentSessionConfig,
    AssistantMessage,
    ToolCallBlock,
)
from loguru import logger

from llmharness.agents.auditor.context import load_auditor_prompt

from ...replay.offline_driver import replay_pipeline_over_trajectory
from ...replay.runner import AuditorSettings, ExtractorSettings
from .adapter import TelBenchInstance, spans_to_messages
from .scoring import SpanScores, score_instance


@dataclass(frozen=True)
class EvalResult:
    """Outcome of evaluating one TELBench instance."""

    instance_id: str
    predicted_error_indices: set[int]
    gold_error_indices: set[int]
    scores: SpanScores
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    n_spans: int = 0
    session_id: str | None = None


async def evaluate_instance(
    instance: TelBenchInstance,
    *,
    mode: Literal["posthoc", "online"],
    provider: tuple[str, dict[str, Any]] | None,
    cwd: str,
    extractor_interval: int = 5,
    audit_interval: int = 5,
    auditor_prompt: str = "telbench",
) -> EvalResult:
    """Run the llmharness pipeline on one TELBench instance and score."""
    messages = spans_to_messages(instance.spans)
    n_spans = len(messages)

    ext_settings = ExtractorSettings.default()
    aud_settings = AuditorSettings(
        base_prompt=load_auditor_prompt(auditor_prompt),
        summary_threshold=30,
    )

    if mode == "posthoc":
        eff_extractor_interval = n_spans
        eff_audit_interval = n_spans
    else:
        eff_extractor_interval = extractor_interval
        eff_audit_interval = audit_interval

    session_id = f"telbench-{instance.id}"

    result = await replay_pipeline_over_trajectory(
        messages=messages,
        cwd=cwd,
        session_id=session_id,
        provider=provider,
        extractor_settings=ext_settings,
        auditor_settings=aud_settings,
        extractor_interval=eff_extractor_interval,
        audit_interval=eff_audit_interval,
        enable_auditor=True,
        stop_on_first_surface=False,
    )

    # Collect all events from the cumulative graph.
    events, _edges, _phases = result.state.graph_view()
    event_by_id: dict[int, Any] = {e.id: e for e in events}

    # Collect surfaced verdicts from step results.
    surfaced_verdicts: list[dict[str, Any]] = []
    matched_event_ids: set[int] = set()

    for step in result.all_step_results:
        reminder = step.get("surfaced_reminder") if isinstance(step, dict) else None
        auditor_record = step.get("auditor_record") if isinstance(step, dict) else None
        if reminder is not None and auditor_record is not None:
            output = auditor_record.get("output") if isinstance(auditor_record, dict) else None
            if output and isinstance(output, dict) and output.get("surface_reminder"):
                surfaced_verdicts.append(output)
                matched_event_ids.update(output.get("matched_event_ids", []))

    # Map matched event ids -> events' source_turns -> span indices.
    predicted_span_indices: set[int] = set()
    for eid in matched_event_ids:
        event = event_by_id.get(eid)
        if event is not None:
            predicted_span_indices.update(event.source_turns)

    gold = instance.gold_error_indices
    scores = score_instance(predicted_span_indices, gold, n_spans)

    return EvalResult(
        instance_id=instance.id,
        predicted_error_indices=predicted_span_indices,
        gold_error_indices=gold,
        scores=scores,
        verdicts=surfaced_verdicts,
        n_spans=n_spans,
    )


# ---------------------------------------------------------------------------
# TEL agent mode
# ---------------------------------------------------------------------------


async def evaluate_instance_tel(
    instance: TelBenchInstance,
    *,
    provider: tuple[str, dict[str, Any]] | None,
    cwd: str,
    prompt_name: str = "default",
) -> EvalResult:
    """Run the TEL agent directly on one TELBench instance and score.

    The TEL agent receives all spans via its span store tool surface and
    directly identifies error span IDs — no extractor/auditor pipeline.
    """
    from agentm.core.runtime import AgentSession, create_agent_session

    from llmharness.agents.tel.tools import SUBMIT_TOOL_NAME

    _OBS = "agentm.extensions.builtin.observability"
    _OPS = "agentm.extensions.builtin.operations"
    _TEL_CTX = "llmharness.agents.tel.context"
    _TEL_TOOLS = "llmharness.agents.tel.tools"

    notepad_path = f"/tmp/tel_notepad_{instance.id}.md"

    ctx_config: dict[str, Any] = {
        "question": instance.question,
        "spans": instance.spans,
        "stages": instance.annotations.get("stage", {}),
        "prompt_name": prompt_name,
        "notepad_path": notepad_path,
    }

    extensions: list[tuple[str, dict[str, Any]]] = [
        (_OBS, {}),
        (_OPS, {}),
        (_TEL_CTX, ctx_config),
        (_TEL_TOOLS, {}),
    ]

    config = AgentSessionConfig(
        cwd=cwd,
        provider=provider,
        extensions=extensions,
        purpose=f"tel_eval_{instance.id}",
    )

    try:
        session = await create_agent_session(AgentSession, config)
    except Exception as exc:
        logger.error(f"[tel] session creation FAILED for {instance.id}: {exc}")
        raise

    sid = session.session_id
    logger.info(f"[tel] {instance.id}: agentm trace messages --session {sid} --format text")

    payload = json.dumps({
        "task": "Identify error spans in this trajectory.",
        "question": instance.question,
        "n_spans": len(instance.spans),
    }, ensure_ascii=False)

    try:
        messages = await session.prompt(payload)
    except Exception as exc:
        with contextlib.suppress(Exception):
            await session.shutdown()
        logger.error(f"[tel] {instance.id} prompt FAILED (sid={sid}): {exc}")
        gold = instance.gold_error_indices
        scores = score_instance(set(), gold, len(instance.spans))
        return EvalResult(
            instance_id=instance.id,
            predicted_error_indices=set(),
            gold_error_indices=gold,
            scores=scores,
            n_spans=len(instance.spans),
            session_id=sid,
        )

    with contextlib.suppress(Exception):
        await session.shutdown()

    # Extract the submit_error_spans tool call.
    predicted_span_ids: list[str] = []
    reasoning = ""
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == SUBMIT_TOOL_NAME:
                predicted_span_ids = list(block.arguments.get("error_span_ids", []))
                reasoning = str(block.arguments.get("reasoning", ""))
                break
        if predicted_span_ids:
            break

    # Convert span IDs to 0-based indices for scoring.
    id_to_idx = {s["id"]: i for i, s in enumerate(instance.spans)}
    predicted_indices = {id_to_idx[sid] for sid in predicted_span_ids if sid in id_to_idx}

    gold = instance.gold_error_indices
    scores = score_instance(predicted_indices, gold, len(instance.spans))

    return EvalResult(
        instance_id=instance.id,
        predicted_error_indices=predicted_indices,
        gold_error_indices=gold,
        scores=scores,
        verdicts=[{"predicted_span_ids": predicted_span_ids, "reasoning": reasoning}]
        if predicted_span_ids else [],
        n_spans=len(instance.spans),
        session_id=sid,
    )


__all__ = [
    "EvalResult",
    "evaluate_instance",
    "evaluate_instance_tel",
]
