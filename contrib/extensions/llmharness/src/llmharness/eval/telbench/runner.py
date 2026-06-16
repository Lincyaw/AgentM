"""TELBench evaluation driver (post-refactor).

Runs the llmharness cognitive-audit pipeline (extractor + auditor) over
TELBench instances and scores the resulting error-span predictions
against gold annotations.

Uses ``atom._prepare_extractor_data`` for payload construction and spawns
child sessions via ``AgentSession.create()`` with scenario paths.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loguru import logger

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
    from agentm.core.abi import AgentSessionConfig, AssistantMessage, ToolCallBlock
    from agentm.core.runtime import AgentSession, create_agent_session

    from ...agents import auditor_scenario, extractor_scenario
    from ...agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
    from ...agents.extractor.graph import GraphOp, parse_op
    from ...atom import _prepare_extractor_data
    from ...state import CumulativeAuditState

    messages = spans_to_messages(instance.spans)
    n_spans = len(messages)
    cumulative = CumulativeAuditState.fresh()

    if mode == "posthoc":
        eff_extractor_interval = n_spans
        eff_audit_interval = n_spans
    else:
        eff_extractor_interval = extractor_interval
        eff_audit_interval = audit_interval

    ext_scenario = extractor_scenario()
    aud_scenario = auditor_scenario()

    # Walk turns, firing extractor + auditor at cadence.
    for turn_count in range(1, n_spans + 1):
        prefix = messages[:turn_count]
        auditor_due = (turn_count % eff_audit_interval) == 0
        extractor_due = (turn_count % eff_extractor_interval) == 0 or auditor_due

        # --- Extractor ---
        if extractor_due:
            data = _prepare_extractor_data(prefix, cumulative, None)
            if data is not None:
                firing_id = cumulative.firing_id_counter
                ops_dir = Path(cwd) / ".agentm" / "audit_ops"
                ops_dir.mkdir(parents=True, exist_ok=True)
                ops_path = ops_dir / f"telbench-{instance.id}-{firing_id}.jsonl"

                ctx_config: dict[str, Any] = dict(data)
                ctx_config["ops_file"] = str(ops_path)
                ctx_config["prompt_name"] = "default"

                config = AgentSessionConfig(
                    cwd=cwd,
                    provider=provider,
                    scenario=ext_scenario,
                    atom_config_overrides={"extractor_context": ctx_config},
                    purpose="cognitive_audit_extractor",
                )
                try:
                    session = await create_agent_session(AgentSession, config)
                    try:
                        await session.prompt(
                            json.dumps(data, ensure_ascii=False, default=str)
                        )
                    finally:
                        await session.shutdown()

                    # Read ops from the file the extractor wrote.
                    ops: list[GraphOp] = []
                    if ops_path.exists():
                        for line in ops_path.read_text(encoding="utf-8").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                ops.append(parse_op(json.loads(line)))
                            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                                continue
                    cumulative.absorb_extractor_firing(
                        firing_ops=ops,
                        firing_cursor=data["window_hi"],
                        firing_id=firing_id,
                    )
                except Exception as exc:
                    # extractor failure — continue with whatever graph we have
                    logger.warning("telbench: extractor firing failed: {}", exc)

        # --- Auditor ---
        if auditor_due:
            events, edges, phases = cumulative.graph_view()
            if not events:
                continue

            config = AgentSessionConfig(
                cwd=cwd,
                provider=provider,
                scenario=aud_scenario,
                atom_config_overrides={
                    "auditor_context": {
                        "events": [e.to_dict() for e in events],
                        "edges": [ed.to_dict() for ed in edges],
                        "phases": [p.to_dict() for p in phases],
                        "continuation_notes": list(cumulative.last_continuation_notes),
                        "summary_threshold": 30,
                        "prompt_name": auditor_prompt,
                    },
                    "auditor_tools": {},
                },
                purpose="cognitive_audit_auditor",
            )
            try:
                session = await create_agent_session(AgentSession, config)
                try:
                    child_msgs = await session.prompt(
                        json.dumps({
                            "graph": [e.to_dict() for e in events],
                            "continuation_notes_from_prior_firing": list(
                                cumulative.last_continuation_notes
                            ),
                        }, ensure_ascii=False, default=str)
                    )
                finally:
                    await session.shutdown()

                # Extract verdict from terminal tool call.
                if child_msgs:
                    for msg in reversed(child_msgs):
                        if not isinstance(msg, AssistantMessage):
                            continue
                        for block in reversed(msg.content):
                            if (
                                isinstance(block, ToolCallBlock)
                                and block.name == SUBMIT_VERDICT_TOOL_NAME
                            ):
                                verdict_raw = block.arguments.get("verdict")
                                if isinstance(verdict_raw, dict):
                                    cumulative.absorb_auditor_verdict(verdict_raw)
                                break
            except Exception as exc:
                # auditor failure — continue
                logger.warning("telbench: auditor verdict absorption failed: {}", exc)

    # --- Score ---
    events, _edges, _phases = cumulative.graph_view()
    event_by_id: dict[int, Any] = {e.id: e for e in events}

    all_verdicts: list[dict[str, Any]] = []
    matched_event_ids: set[int] = set()
    for v in cumulative.recent_verdicts:
        all_verdicts.append(v)
        matched_event_ids.update(v.get("matched_event_ids", []))

    predicted_span_indices: set[int] = set()
    for eid in matched_event_ids:
        event = event_by_id.get(eid)
        if event is not None:
            predicted_span_indices.update(event.source_turns)

    gold = instance.gold_error_indices
    scores = score_instance(predicted_span_indices, gold, n_spans)

    # Clean up ops files.
    ops_dir = Path(cwd) / ".agentm" / "audit_ops"
    if ops_dir.exists():
        shutil.rmtree(ops_dir, ignore_errors=True)

    return EvalResult(
        instance_id=instance.id,
        predicted_error_indices=predicted_span_indices,
        gold_error_indices=gold,
        scores=scores,
        verdicts=all_verdicts,
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
    import contextlib

    from agentm.core.abi import AgentSessionConfig, AssistantMessage, ToolCallBlock
    from agentm.core.runtime import AgentSession, create_agent_session

    from ...agents.tel.tools import SUBMIT_TOOL_NAME

    _OBS = "agentm.extensions.builtin.observability"
    _OPS = "agentm.extensions.builtin.operations"
    _TEL_CTX = "llmharness.agents.tel.context"
    _TEL_TOOLS = "llmharness.agents.tel.tools"

    notepad_path = str(Path(cwd) / f"tel_notepad_{instance.id}.md")

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
        tool_allowlist=[
            "list_spans",
            "get_span",
            "search_spans",
            "submit_error_spans",
        ],
    )

    try:
        session = await create_agent_session(AgentSession, config)
    except Exception as exc:
        logger.error(f"[tel] session creation FAILED for {instance.id}: {exc}")
        raise

    sid = session.session_id
    # The ``agentm trace`` debug command is logged centrally by
    # create_agent_session (purpose=tel_eval_<id>), so it is not repeated here.

    payload = json.dumps(
        {
            "task": "Identify error spans in this trajectory.",
            "question": instance.question,
            "n_spans": len(instance.spans),
        },
        ensure_ascii=False,
    )

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
        if predicted_span_ids
        else [],
        n_spans=len(instance.spans),
    )


__all__ = [
    "EvalResult",
    "evaluate_instance",
    "evaluate_instance_tel",
]
