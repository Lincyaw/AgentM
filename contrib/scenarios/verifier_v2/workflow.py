"""Single convergence loop — the entire verification workflow.

This is the orchestration core. It wires:
  evaluate_gaps → plan_work → streaming pool (searcher → compiler → judge) → apply

No separate seed phase, propagation phase, audit phase, or final check loop.
One loop to convergence.
"""
from __future__ import annotations

import asyncio
import heapq
from typing import Any

from agentm.extensions.builtin.workflow import WorkflowContext

from .compiler import compile_evidence
from .gaps import evaluate_gaps
from .judge import build_judge_prompt, compute_verdict, majority_vote
from .planner import expand_frontier, plan_work
from .schema import (
    CompiledDossier,
    EvidenceDossier,
    GapReport,
    JudgeVerdict,
    PropagationResult,
    TaskAttempt,
    Verdict,
    VerificationTask,
)
from .searcher import build_searcher_prompt
from .state import Case, GraphState


async def run(ctx: WorkflowContext) -> dict[str, Any]:
    """Entry point: single convergence loop."""
    case = Case.from_args(ctx.args)
    state = GraphState(case=case, log=ctx.log)

    ctx.phase("verify")
    round_n = 0

    while True:
        round_n += 1
        state.rounds = round_n

        # ① Evaluate gaps (deterministic)
        gap_report = evaluate_gaps(case, state)
        if gap_report.satisfied:
            ctx.log(f"All structural invariants satisfied after {round_n - 1} rounds")
            break

        # ② Plan work (deterministic)
        tasks = plan_work(case, state, gap_report)
        if not tasks:
            ctx.log(f"No more candidates available; {len(gap_report.gaps)} gaps remain")
            break

        ctx.log(
            f"Round {round_n}: {len(gap_report.gaps)} gaps, "
            f"{len(tasks)} tasks planned"
        )

        # ③ Execute through streaming pool
        progress = await _execute_pool(ctx, case, state, tasks)

        if not progress:
            ctx.log("No progress this round; stopping")
            break

    # Optional: explore post-pass (if gaps remain)
    final_gaps = evaluate_gaps(case, state)
    if not final_gaps.satisfied:
        ctx.log(f"Convergence with {len(final_gaps.gaps)} remaining gaps")

    # Finalize
    ctx.phase("finalize")
    return _build_result(case, state, final_gaps)


async def _execute_pool(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    tasks: list[VerificationTask],
) -> bool:
    """Streaming worker pool: process tasks by priority, expand on confirm."""
    # Priority queue: (priority, seq, task)
    pq: list[tuple[Any, int, VerificationTask]] = []
    seq = 0
    scheduled: set[str] = set()

    def _enqueue(task: VerificationTask) -> None:
        nonlocal seq
        if task.edge_key in scheduled:
            return
        scheduled.add(task.edge_key)
        heapq.heappush(pq, (task.priority, seq, task))
        seq += 1

    for task in tasks:
        _enqueue(task)

    in_flight: dict[asyncio.Task[tuple[VerificationTask, Verdict]], VerificationTask] = {}
    progress = False

    def _fill() -> None:
        """Submit highest-priority tasks to idle workers."""
        while pq and len(in_flight) < case.max_parallel:
            _, _, task = heapq.heappop(pq)
            if task.edge_key in state.exhausted_edges:
                continue
            if task.edge_key in state.rejected_edges:
                continue
            coro = _verify_one(ctx, case, state, task)
            aio_task = asyncio.create_task(coro)
            in_flight[aio_task] = task

    _fill()

    while in_flight:
        done, _ = await asyncio.wait(
            set(in_flight), return_when=asyncio.FIRST_COMPLETED,
        )
        for aio_task in done:
            task = in_flight.pop(aio_task)
            _, verdict = aio_task.result()

            if verdict.kind == "confirmed":
                changed = state.accept(task, verdict)
                if changed:
                    progress = True
                    ctx.log(
                        f"  ✓ {task.source_seed}: "
                        f"{task.from_entity} → {task.to_entity}: confirmed "
                        f"({verdict.predicate})"
                    )
                    # Expand frontier from newly confirmed node
                    if task.kind == "hop":
                        new_tasks = expand_frontier(
                            case, state, task.to_entity, task.source_seed,
                        )
                        for t in new_tasks:
                            _enqueue(t)
                    elif task.kind == "seed":
                        new_tasks = expand_frontier(
                            case, state, task.from_entity, task.source_seed,
                        )
                        for t in new_tasks:
                            _enqueue(t)

                    # Check early termination
                    if evaluate_gaps(case, state).satisfied:
                        ctx.log("All gaps satisfied — canceling remaining work")
                        for remaining in in_flight:
                            remaining.cancel()
                        await asyncio.gather(*in_flight, return_exceptions=True)
                        in_flight.clear()
                        return True

            elif verdict.kind == "rejected":
                state.mark_rejected(task)
                ctx.log(
                    f"  ✗ {task.source_seed}: "
                    f"{task.from_entity} → {task.to_entity}: rejected"
                )

            else:  # inconclusive
                if task.kind == "seed" and not state.confirmed_seeds:
                    # First pass: no other seeds confirmed yet.
                    # Mark as pending context retry — don't exhaust yet.
                    state.inconclusive_seeds_pending_context.add(task.edge_key)
                    ctx.log(
                        f"  ? {task.source_seed}: "
                        f"{task.from_entity} → {task.to_entity}: "
                        f"inconclusive (pending multi-fault retry)"
                    )
                elif (
                    task.kind == "seed"
                    and task.edge_key in state.inconclusive_seeds_pending_context
                ):
                    # Was pending, now retried with context — truly exhausted
                    state.inconclusive_seeds_pending_context.discard(task.edge_key)
                    state.mark_exhausted(task)
                    ctx.log(
                        f"  ? {task.source_seed}: "
                        f"{task.from_entity} → {task.to_entity}: "
                        f"inconclusive (exhausted after context retry)"
                    )
                else:
                    state.mark_exhausted(task)
                    ctx.log(
                        f"  ? {task.source_seed}: "
                        f"{task.from_entity} → {task.to_entity}: inconclusive"
                    )

        _fill()

    return progress


async def _verify_one(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    task: VerificationTask,
) -> tuple[VerificationTask, Verdict]:
    """Single task: searcher → compiler → judge, with retry."""
    for attempt_n in range(task.max_retries):
        state.agent_calls += 1
        history = state.attempt_history(task)

        # Determine coverage feedback from previous attempt
        coverage_feedback = ""
        if history and history[-1].coverage_gaps:
            coverage_feedback = "; ".join(history[-1].coverage_gaps)

        # ① Searcher
        searcher_prompt = build_searcher_prompt(
            case, task, history, coverage_feedback, state=state,
        )
        dossier = await _call_searcher(ctx, case, task, searcher_prompt)
        if dossier is None:
            state.record_attempt(task, TaskAttempt(
                attempt_n=attempt_n,
                coverage_gaps=["searcher returned no structured output"],
                verdict_kind=None,
            ))
            continue

        # ② Compiler (deterministic)
        compiled = compile_evidence(dossier, task, case.data_dir, case.data_profile)

        # Coverage retry: if critical gaps and retries remain, go back to searcher
        if compiled.has_critical_gaps and attempt_n < task.max_retries - 1:
            state.record_attempt(task, TaskAttempt(
                attempt_n=attempt_n,
                coverage_gaps=[g.description for g in compiled.coverage_gaps],
                verdict_kind=None,
                sql_summary=[r.sql[:80] for r in
                             compiled.target_results + compiled.relationship_results
                             if r.success],
            ))
            continue

        # ③ Judge
        state.agent_calls += 1
        if task.is_critical:
            verdict = await _judge_with_voting(ctx, case, state, compiled)
        else:
            verdict = await _call_judge(ctx, case, state, compiled, with_global_context=True)

        # Record attempt
        state.record_attempt(task, TaskAttempt(
            attempt_n=attempt_n,
            coverage_gaps=[g.description for g in compiled.coverage_gaps],
            verdict_kind=verdict.kind,
            judge_rationale=verdict.rationale[:200],
            sql_summary=[r.sql[:80] for r in
                         compiled.target_results + compiled.relationship_results
                         if r.success],
        ))

        if verdict.kind != "inconclusive":
            return task, verdict

        # Strategy retry: continue loop with updated history

    return task, Verdict(kind="inconclusive", rationale="exhausted after retries")


async def _call_searcher(
    ctx: WorkflowContext,
    case: Case,
    task: VerificationTask,
    prompt: str,
) -> EvidenceDossier | None:
    """Dispatch searcher agent and parse structured output."""
    label = f"search-{task.edge_key[:60]}"
    result = await ctx.agent(
        prompt,
        scenario="verifier_v2/searcher",
        schema=EvidenceDossier,
        atom_config={"duckdb_sql": {"data_dir": case.data_dir}},
        retry=2,
        trace_label=label,
    )
    if isinstance(result, EvidenceDossier):
        ctx.log(
            f"  searcher returned dossier: "
            f"{len(result.target_observations)} target, "
            f"{len(result.control_observations)} control, "
            f"{len(result.counter_evidence)} counter queries"
        )
        return result
    ctx.log(f"  searcher returned non-dossier: {type(result).__name__}: {str(result)[:200]}")
    return None


async def _call_judge(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    compiled: CompiledDossier,
    *,
    with_global_context: bool = True,
) -> Verdict:
    """Dispatch judge agent (no tools) and compute verdict."""
    prompt = build_judge_prompt(compiled, case, state, with_global_context=with_global_context)
    label = f"judge-{compiled.task.edge_key[:60]}"
    result = await ctx.agent(
        prompt,
        scenario="verifier_v2/judge",
        schema=JudgeVerdict,
        retry=2,
        trace_label=label,
    )
    if isinstance(result, JudgeVerdict):
        return compute_verdict(result, compiled)
    return Verdict(kind="inconclusive", rationale="judge returned no structured output")


async def _judge_with_voting(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    compiled: CompiledDossier,
    n_votes: int = 3,
) -> Verdict:
    """Run multiple independent judges and take majority vote."""
    state.agent_calls += n_votes - 1  # already counted 1

    verdicts = await ctx.parallel([
        _call_judge(ctx, case, state, compiled, with_global_context=True),
        _call_judge(ctx, case, state, compiled, with_global_context=False),
        _call_judge(ctx, case, state, compiled, with_global_context=True),
    ][:n_votes])

    valid = [v for v in verdicts if isinstance(v, Verdict)]
    return majority_vote(valid) if valid else Verdict(
        kind="inconclusive", rationale="all judges failed",
    )


def _build_result(
    case: Case,
    state: GraphState,
    final_gaps: GapReport,
) -> dict[str, Any]:
    """Assemble the final PropagationResult."""
    return PropagationResult(
        nodes=list(state.nodes.values()),
        edges=state.edges,
        confirmed_seeds=sorted(state.confirmed_seeds),
        gaps_satisfied=final_gaps.satisfied,
        total_rounds=state.rounds,
        total_agent_calls=state.agent_calls,
        exhausted_edges=sorted(state.exhausted_edges),
        rejected_edges=sorted(state.rejected_edges),
        remaining_gaps=[
            {"kind": g.kind, "id": g.id, "target": g.target}
            for g in final_gaps.gaps
        ],
    ).model_dump(mode="json")
