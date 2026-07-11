"""TELBench evaluation driver (post-refactor).

Runs the llmharness cognitive-audit pipeline (auditor) over TELBench
instances and scores the resulting error-span predictions against gold
annotations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from . import _tel_agent_dir
from .adapter import TelBenchInstance, spans_to_messages
from .scoring import SpanScores, score_instance


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Outcome of evaluating one TELBench instance."""

    instance_id: str
    predicted_error_indices: set[int]
    gold_error_indices: set[int]
    scores: SpanScores
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    n_spans: int = 0
    session_id: str = ""
    reason_session_id: str = ""


async def evaluate_instance(
    instance: TelBenchInstance,
    *,
    mode: Literal["posthoc", "online"],
    provider: tuple[str, dict[str, Any]] | None = None,
    model: str | None = None,
    cwd: str,
    audit_interval: int = 5,
    auditor_prompt: str = "index",
    extractor_interval: int = 5,  # deprecated no-op
) -> EvalResult:
    """Run the llmharness auditor pipeline on one TELBench instance and score."""
    from agentm_eval.methods.auditor import index_to_context, run_auditor
    from agentm_eval.methods.index import build_index, extract_symbols

    _ = extractor_interval  # deprecated no-op

    messages = spans_to_messages(instance.spans)
    n_spans = len(messages)

    context_index: dict[str, Any] | None = None
    try:
        chunks = await extract_symbols(
            messages, model=model, run_id=instance.id,
            chunk_size=(4, 8), vocabulary="default",
        )
        if chunks:
            idx = await build_index(chunks, model=model, resolve=True)
            context_index = index_to_context(idx)
    except Exception:
        logger.debug("telbench: index build failed for {}", instance.id, exc_info=True)

    eff_interval: int | None = None if mode == "posthoc" else audit_interval

    audit_result = await run_auditor(
        messages, model=model, provider=provider, cwd=cwd,
        auditor_prompt=auditor_prompt, audit_interval=eff_interval,
        context_index=context_index,
    )

    all_verdicts = [v.to_dict() for v in audit_result.verdicts]
    matched_event_ids: set[int] = set()
    for v in all_verdicts:
        matched_event_ids.update(v.get("matched_event_ids", []))

    predicted_span_indices: set[int] = set()
    last_session_id = audit_result.session_ids[-1] if audit_result.session_ids else ""

    gold = instance.gold_error_indices
    scores = score_instance(predicted_span_indices, gold, n_spans)

    return EvalResult(
        instance_id=instance.id,
        predicted_error_indices=predicted_span_indices,
        gold_error_indices=gold,
        scores=scores,
        verdicts=all_verdicts,
        n_spans=n_spans,
        session_id=last_session_id,
        reason_session_id=last_session_id,
    )


# ---------------------------------------------------------------------------
# TEL agent mode
# ---------------------------------------------------------------------------


async def evaluate_instance_tel(
    instance: TelBenchInstance,
    *,
    provider: tuple[str, dict[str, Any]] | None = None,
    model: str | None = None,
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

    from llmharness.agents.tel.tools import SUBMIT_TOOL_NAME

    if prompt_name == "2pass":
        return await _run_tel_2pass(instance, provider=provider, model=model, cwd=cwd)

    _OBS = "agentm.extensions.builtin.observability"
    _OPS = "agentm.extensions.builtin.operations"
    _TEL_CTX = "llmharness.agents.tel.context"
    _TEL_TOOLS = "llmharness.agents.tel.tools"

    # Notepads from every instance in a run collect in one inspectable dir
    # (a sibling of the per-instance cwds), each file named by session id so it
    # lines up with the ``agentm trace`` command the session logs.
    notepad_dir = Path(cwd).parent / "notepads"

    ctx_config: dict[str, Any] = {
        "question": instance.question,
        "spans": instance.spans,
        # NOTE: span ``stage`` lives under dataset ``annotations`` (alongside the
        # gold ``error_type``), NOT in the raw span objects, and correlates ~0.7
        # with being a gold error span. Feeding it to the agent leaks the label.
        # The trajectory input is raw span text only.
        "stages": {},
        "prompt_name": prompt_name,
    }

    extensions: list[tuple[str, dict[str, Any]]] = [
        (_OBS, {}),
        (_OPS, {}),
        (_TEL_CTX, ctx_config),
        (_TEL_TOOLS, {"notepad_dir": str(notepad_dir)}),
    ]

    config = AgentSessionConfig(
        cwd=cwd,
        model=model,
        provider=provider,
        extensions=extensions,
        purpose=f"tel_eval_{instance.id}",
        tool_allowlist=[
            "list_spans",
            "get_span",
            "search_spans",
            "note",
            SUBMIT_TOOL_NAME,
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

    predicted_span_ids: list[str] = []
    reasoning = ""
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if not (isinstance(block, ToolCallBlock) and block.name == SUBMIT_TOOL_NAME):
                continue
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

    # Append the outcome to this session's notepad file so each file is a
    # self-contained record (notes the agent took + what it predicted + gold +
    # score), inspectable live as a batch run progresses.
    idx_to_id = {i: s["id"] for i, s in enumerate(instance.spans)}
    footer = (
        f"\n\n## result ({instance.id}, session {sid})\n"
        f"- predicted: {predicted_span_ids}\n"
        f"- gold:      {[idx_to_id[i] for i in sorted(gold) if i in idx_to_id]}\n"
        f"- P={scores.precision:.3f} R={scores.recall:.3f} F1={scores.f1:.3f} "
        f"FEA={'Y' if scores.first_error_accurate else 'N'}\n"
        f"- reasoning: {reasoning}\n"
    )
    try:
        notepad_dir.mkdir(parents=True, exist_ok=True)
        with open(notepad_dir / f"{sid}.md", "a", encoding="utf-8") as fh:
            fh.write(footer)
    except OSError as exc:
        logger.warning("telbench: notepad footer write failed: {}", exc)

    return EvalResult(
        instance_id=instance.id,
        predicted_error_indices=predicted_indices,
        gold_error_indices=gold,
        scores=scores,
        verdicts=[{"predicted_span_ids": predicted_span_ids, "reasoning": reasoning}]
        if predicted_span_ids
        else [],
        n_spans=len(instance.spans),
        session_id=sid,
        reason_session_id=sid,
    )


_TEL_WORKFLOW_SCRIPT = _tel_agent_dir() / "workflow.py"


async def _run_tel_2pass(
    instance: TelBenchInstance,
    *,
    provider: tuple[str, dict[str, Any]] | None = None,
    model: str | None = None,
    cwd: str,
) -> EvalResult:
    """Two-pass TEL via the workflow module (note → reason → submit)."""
    from agentm.core.abi import AgentSessionConfig
    from agentm.core.runtime import AgentSession

    notepad_dir = Path(cwd).parent / "notepads"
    gold = instance.gold_error_indices
    n_spans = len(instance.spans)

    wf_args: dict[str, Any] = {
        "question": instance.question,
        "spans": instance.spans,
        "notepad_dir": str(notepad_dir),
        "instance_id": instance.id,
    }

    config = AgentSessionConfig(
        cwd=cwd,
        model=model,
        provider=provider,
        extensions=[
            ("agentm.extensions.builtin.observability", {}),
            ("agentm.extensions.builtin.operations", {}),
            ("agentm.extensions.builtin.artifact_store", {}),
            ("agentm.extensions.builtin.workflow", {}),
        ],
        purpose=f"tel_2pass_{instance.id}",
        auto_commit=False,
        log_trace_command=False,
    )

    session = await AgentSession.create(config)

    try:
        runner = session.get_service("workflow_runner")
        if runner is None:
            raise RuntimeError("workflow_runner service not found")
        result = await runner.run_file(_TEL_WORKFLOW_SCRIPT, wf_args)
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            await session.shutdown()

    predicted_span_ids: list[str] = []
    reasoning = ""
    reason_session_id = ""
    if isinstance(result, dict):
        predicted_span_ids = list(result.get("predicted_span_ids", []))
        reasoning = str(result.get("reasoning", ""))
        reason_session_id = str(result.get("reason_session_id", ""))

    id_to_idx = {s["id"]: i for i, s in enumerate(instance.spans)}
    predicted_indices = {id_to_idx[sid] for sid in predicted_span_ids if sid in id_to_idx}
    scores = score_instance(predicted_indices, gold, n_spans)

    # Result footer
    idx_to_id = {i: s["id"] for i, s in enumerate(instance.spans)}
    footer = (
        f"\n\n## result ({instance.id}, 2pass)\n"
        f"- predicted: {predicted_span_ids}\n"
        f"- gold:      {[idx_to_id[i] for i in sorted(gold) if i in idx_to_id]}\n"
        f"- P={scores.precision:.3f} R={scores.recall:.3f} F1={scores.f1:.3f} "
        f"FEA={'Y' if scores.first_error_accurate else 'N'}\n"
        f"- reasoning: {reasoning}\n"
    )
    try:
        notepad_dir.mkdir(parents=True, exist_ok=True)
        with open(notepad_dir / f"2pass_{instance.id}.md", "w", encoding="utf-8") as fh:
            fh.write(footer)
    except OSError as exc:
        logger.warning("telbench: notepad footer write failed: {}", exc)

    return EvalResult(
        instance_id=instance.id,
        predicted_error_indices=predicted_indices,
        gold_error_indices=gold,
        scores=scores,
        verdicts=[{"predicted_span_ids": predicted_span_ids, "reasoning": reasoning}]
        if predicted_span_ids
        else [],
        n_spans=n_spans,
        session_id=session.session_id,
        reason_session_id=reason_session_id,
    )


async def reflect_on_result(
    result: EvalResult,
    instance: TelBenchInstance,
    *,
    provider: tuple[str, dict[str, Any]] | None = None,
    model: str | None = None,
    cwd: str = "/tmp/tel_reflect",
) -> str:
    """Resume the reason session with gold labels and a reflection prompt.

    Returns the reflection text (structured methodology lessons).
    """
    import contextlib

    from agentm.core.abi import AgentSessionConfig
    from agentm.core.runtime import AgentSession
    from agentm.core.runtime.session_bootstrap import make_default_session_store

    if not result.reason_session_id:
        return f"[{result.instance_id}] no reason_session_id — cannot reflect"

    idx_to_id = {i: s["id"] for i, s in enumerate(instance.spans)}
    predicted_ids = [idx_to_id[i] for i in sorted(result.predicted_error_indices) if i in idx_to_id]
    gold_ids = [idx_to_id[i] for i in sorted(result.gold_error_indices) if i in idx_to_id]
    missed = sorted(set(gold_ids) - set(predicted_ids))
    false_pos = sorted(set(predicted_ids) - set(gold_ids))

    correction = (
        f"## CORRECTION\n\n"
        f"- Your predicted error spans: {predicted_ids}\n"
        f"- Gold (correct) error spans: {gold_ids}\n"
        f"- Missed (false negatives): {missed}\n"
        f"- Wrongly flagged (false positives): {false_pos}\n"
        f"- Scores: P={result.scores.precision:.3f} R={result.scores.recall:.3f} "
        f"F1={result.scores.f1:.3f}\n"
    )

    store = make_default_session_store(cwd)
    try:
        session_manager: Any = store.open(result.reason_session_id)
    except FileNotFoundError:
        return (
            f"[{result.instance_id}] session {result.reason_session_id} "
            f"not found in store — cannot resume for reflection"
        )

    _TEL_CTX = "llmharness.agents.tel.context"
    _TEL_TOOLS = "llmharness.agents.tel.tools"

    config = AgentSessionConfig(
        cwd=cwd,
        model=model,
        provider=provider,
        session_manager=session_manager,
        session_id=result.reason_session_id,
        extensions=[
            ("agentm.extensions.builtin.observability", {}),
            ("agentm.extensions.builtin.operations", {}),
            (
                _TEL_CTX,
                {
                    "question": instance.question,
                    "spans": instance.spans,
                    "stages": {},
                    "prompt_name": "reflect",
                },
            ),
            (_TEL_TOOLS, {}),
        ],
        purpose=f"tel_reflect_{result.instance_id}",
        tool_allowlist=["list_spans", "get_span", "search_spans"],
    )

    session = await AgentSession.create(config)

    try:
        messages = await session.prompt(correction)
    finally:
        with contextlib.suppress(Exception):
            await session.shutdown()

    from agentm.core.abi import AssistantMessage, TextContent

    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            parts = [b.text for b in msg.content if isinstance(b, TextContent) and b.text.strip()]
            if parts:
                return "\n".join(parts)

    return f"[{result.instance_id}] reflection produced no text output"


__all__ = [
    "EvalResult",
    "evaluate_instance",
    "evaluate_instance_tel",
    "reflect_on_result",
]
