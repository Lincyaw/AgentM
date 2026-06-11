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
    from ...agents.extractor.tools import GraphOp, parse_op
    from ...atom import CumulativeAuditState, _prepare_extractor_data

    messages = spans_to_messages(instance.spans)
    n_spans = len(messages)
    cumulative = CumulativeAuditState.fresh()

    if mode == "posthoc":
        eff_extractor_interval = n_spans
        eff_audit_interval = n_spans
    else:
        eff_extractor_interval = extractor_interval
        eff_audit_interval = audit_interval

    session_id = f"telbench-{instance.id}"
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
                except Exception:
                    pass  # extractor failure — continue with whatever graph we have

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
            except Exception:
                pass  # auditor failure — continue

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


__all__ = [
    "EvalResult",
    "evaluate_instance",
]
