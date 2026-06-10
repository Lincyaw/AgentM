"""TELBench evaluation driver.

Runs the llmharness cognitive-audit pipeline (extractor + auditor) over
TELBench instances and scores the resulting error-span predictions
against gold annotations.

Two modes:

* **posthoc** — all spans fed as a complete trajectory; extractor fires
  once at the end, auditor fires once on the full graph.
* **online** — spans fed incrementally; extractor + auditor fire every K
  spans; verdicts accumulate across all firings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ...audit.auditor.prompt import load_auditor_prompt
from ...audit.runner import AuditorSettings, ExtractorSettings
from ...replay.offline_driver import replay_pipeline_over_trajectory
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
    messages = spans_to_messages(instance.spans)
    n_spans = len(messages)

    ext_settings = ExtractorSettings.default()
    aud_default = AuditorSettings.default()
    aud_settings = AuditorSettings(
        base_prompt=load_auditor_prompt(auditor_prompt),
        observability_config=aud_default.observability_config,
        summary_threshold=aud_default.summary_threshold,
        tools=aud_default.tools,
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
        if step.surfaced_reminder is not None and step.auditor_record is not None:
            output = step.auditor_record.output
            if output and output.get("surface_reminder"):
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


__all__ = [
    "EvalResult",
    "evaluate_instance",
]
