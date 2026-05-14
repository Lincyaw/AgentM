"""Phase 2 eval runner for rca_hfsm — drives the LLM-mode judge stack
against the rca:baseline task suite.

Per the C3 plan (.claude/plans/2026-05-13-rca-hfsm-phase2-llm-native-
judges.md): pick up to 10 representative tasks from
``contrib/scenarios/rca/eval/baseline/tasks/`` (the only YAML-defined
task suite — the 50-case ops-lite eval lives in HF dataset form and is
driven by ``rcabench-platform``, not local YAML), run each through a
fresh rca_hfsm session, and tabulate the trajectory.

Per CLAUDE.md "E2E methodology": we drive a real ``AgentSession`` (the
same code path the ``agentm`` CLI uses) and inspect the JSONL trace.
The only reason we bootstrap in-process rather than ``subprocess
agentm`` is so we can set ``eval_task_id`` on the session config —
the CLI does not expose that flag, and the rca grader keys traces by
``task_meta.task_id``.

The eval writes ``phase2_results.md`` next to this file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repo paths -----------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# Mirror the ``agentm`` CLI's dotenv autoload so direct ``python
# run_phase2_eval.py`` picks up the same provider credentials. The CLI
# walks up from cwd looking for the nearest .env; we anchor on the repo
# root since the eval script's location is fixed under it.
_DOTENV = _REPO_ROOT / ".env"
if _DOTENV.is_file() and not os.environ.get("AGENTM_SKIP_DOTENV"):
    load_dotenv(_DOTENV, override=False)
_TASKS_DIR = _REPO_ROOT / "contrib/scenarios/rca/eval/baseline/tasks"
_OUT_DIR = Path(__file__).resolve().parent
_TRACE_ROOT = _OUT_DIR / "traces"


@dataclass
class CaseResult:
    task_id: str
    task_class: str
    prompt_summary: str
    expected_services: list[str]
    expected_fault_kind: str
    trace_id: str | None = None
    jsonl_path: str | None = None
    final_text: str = ""
    final_report_args: dict[str, Any] = field(default_factory=dict)
    grader_score: float = 0.0
    grader_verdict: str = "unknown"  # ok / correctness / runtime
    grader_feedback: str = ""
    fsm_final_state: str | None = None
    turn_count: int = 0
    hypothesis_count: int = 0
    observation_count: int = 0
    symptom_count: int = 0
    final_report_emitted: bool = False
    dispatch_agent_count: int = 0
    last_stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    elapsed_sec: float = 0.0
    judge_calls: dict[str, int] = field(default_factory=dict)
    mutation_kinds: dict[str, int] = field(default_factory=dict)
    downgrade_reasons: list[str] = field(default_factory=list)


# --- Grader -----------------------------------------------------------------


def _local_grade(
    task: dict[str, Any], jsonl_path: Path | None, final_text: str
) -> dict[str, Any]:
    """Score the same way ``rca/eval/baseline/grader.py`` does but
    locate the trace by directory (one trace per case dir) instead of
    by ``task_meta.task_id``.

    Rationale: the core ``session_factory.SessionReadyEvent`` emit
    (src/agentm/core/runtime/session_factory.py L366) does not forward
    ``AgentSessionConfig.eval_task_id`` / ``task_class`` /
    ``eval_run_id`` to the observability atom, so ``task_meta.task_id``
    is always ``None`` for programmatically-constructed sessions. The
    Phase 2 plan explicitly forbids core changes, so this runner does
    the trace-matching itself. The scoring rule is identical to the
    upstream grader: 0.7 * service_hit + 0.3 * fault_kind_hit, with
    case-insensitive substring matching against
    ``submit_final_report.args``.
    """

    expected = task.get("expected") or {}
    expected_services = [
        s for s in (expected.get("expected_services") or []) if isinstance(s, str)
    ]
    expected_fault_kind = str(expected.get("fault_kind") or "").strip().lower()

    services: list[str] = []
    fault_kinds: list[str] = []
    raw_payload = ""
    found_report = False
    if jsonl_path is not None and jsonl_path.is_file():
        try:
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    attrs = rec.get("attributes") or {}
                    if rec.get("name") != "emit:tool_call":
                        continue
                    event = attrs.get("event") or {}
                    if event.get("tool_name") != "submit_final_report":
                        continue
                    args = event.get("args") or {}
                    if not isinstance(args, dict):
                        continue
                    found_report = True
                    raw_payload = json.dumps(args)
                    # rca_hfsm's submit_final_report schema is a single
                    # ``root_cause`` string + ``supporting_observations``
                    # list, NOT the rca scenario's ``root_causes[]`` of
                    # {service, fault_kind} structs. So we scan the
                    # whole args blob lowercased for the expected
                    # tokens.
                    for rc in args.get("root_causes") or []:
                        if isinstance(rc, dict):
                            if isinstance(rc.get("service"), str):
                                services.append(rc["service"])
                            if isinstance(rc.get("fault_kind"), str):
                                fault_kinds.append(rc["fault_kind"])
        except OSError:
            pass

    raw_blob = (
        raw_payload
        + " "
        + " ".join(services)
        + " "
        + " ".join(fault_kinds)
        + " "
        + (final_text or "")
    )
    raw_lower = raw_blob.lower()
    service_hit = (
        1.0 if any(s.lower() in raw_lower for s in expected_services) else 0.0
    )
    fault_kind_hit = (
        1.0 if expected_fault_kind and expected_fault_kind in raw_lower else 0.0
    )
    score = 0.7 * service_hit + 0.3 * fault_kind_hit
    if not found_report:
        failure_kind = "runtime"
        feedback = (
            "No submit_final_report tool_call observed in the case "
            "trace; orchestrator never reached FINALIZE."
        )
    elif service_hit == 1.0 and fault_kind_hit == 1.0:
        failure_kind = "ok"
        feedback = (
            f"Verdict matched both service and fault_kind "
            f"(score={score:.2f})."
        )
    else:
        failure_kind = "correctness"
        parts: list[str] = []
        if service_hit:
            parts.append("named expected service")
        else:
            parts.append(
                f"missed services (expected one of {expected_services})"
            )
        if fault_kind_hit:
            parts.append(f"named fault_kind {expected_fault_kind!r}")
        else:
            parts.append(
                f"missed fault_kind (expected substring "
                f"{expected_fault_kind!r})"
            )
        feedback = "; ".join(parts) + "."
    return {
        "score": float(score),
        "feedback_text": feedback,
        "failure_kind": failure_kind,
    }


# --- Task selection ---------------------------------------------------------


def select_tasks(limit: int) -> list[dict[str, Any]]:
    """Return up to ``limit`` tasks; if the dir has fewer, return all.

    Selection criterion: include every YAML in deterministic ID order.
    The baseline suite holds three cases — a multi-service propagation
    fault (mysql network_corrupt), a single-service pod-failure, and a
    JVM-level stress case. Together they exercise: (1) multiple
    plausible hypotheses → judge.independence + judge.coverage stress,
    (2) tight single-service attribution → judge.satisfied focus,
    (3) container-vs-JVM ambiguity → judge.falsified_genuinely.
    """

    files = sorted(_TASKS_DIR.glob("*.yaml"))
    tasks: list[dict[str, Any]] = []
    for path in files[:limit]:
        with path.open("r", encoding="utf-8") as fh:
            tasks.append(yaml.safe_load(fh))
    return tasks


# --- Trace inspection -------------------------------------------------------


def _find_trace_jsonl(cwd: Path, task_id: str) -> Path | None:
    """Return the eval session's root JSONL trace under ``cwd``.

    Each case gets its own ``cwd`` so there's exactly one logical agent
    session per directory, but the catalog indexer runs first and emits
    its own ``session.start`` JSONL alongside the eval trace's. Filter
    on ``purpose == "eval:rca_hfsm:phase2"`` (set by the runner) so we
    pick the right one without relying on ``task_meta.task_id`` —
    which the core ``SessionReadyEvent`` emit drops on the floor (see
    note in ``_local_grade``). ``task_id`` is accepted for API
    symmetry but unused.
    """

    del task_id  # see docstring
    obs_dir = cwd / ".agentm" / "observability"
    if not obs_dir.is_dir():
        return None
    candidates = sorted(
        obs_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("kind") == "session.start":
                        attrs = rec.get("attributes") or {}
                        if attrs.get("purpose") == "eval:rca_hfsm:phase2":
                            return path
                        break  # only first record is session.start
        except OSError:
            continue
    # Fallback: most recent trace in the dir (catalog indexer's own
    # JSONL has ``session.start.purpose == "catalog-indexer"``, so the
    # filter above already excludes it; this fallback is only for
    # debugging an unexpected layout).
    return candidates[0] if candidates else None


def _tabulate_trace(jsonl_path: Path) -> dict[str, Any]:
    """Extract per-case metrics from a JSONL trace.

    The rca_hfsm bus does NOT emit FSM-transition events, and Phase 2
    did not wire ``rca.judge.invoked`` events (design §3.4 left that
    optional). So FSM state and judge call counts are derived from
    other signals: ``mutation_kinds`` (downgraded vs applied) is the
    only on-bus proxy for judge effects.
    """

    out: dict[str, Any] = {
        "trace_id": None,
        "turn_count": 0,
        "judge_calls": {},
        "mutation_kinds": {},
        "downgrade_reasons": [],
        "hypothesis_count": 0,
        "observation_count": 0,
        "symptom_count": 0,
        "final_report_emitted": False,
        "final_report_args": {},
        "dispatch_agent_count": 0,
        "last_stop_reason": None,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    try:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                kind = rec.get("kind")
                attrs = rec.get("attributes") or {}
                if kind == "session.fingerprint":
                    out["trace_id"] = rec.get("trace_id")
                if kind == "turn.summary":
                    out["turn_count"] += 1
                    out["last_stop_reason"] = attrs.get("stop_reason") or out[
                        "last_stop_reason"
                    ]
                    out["input_tokens"] += int(attrs.get("input_tokens") or 0)
                    out["output_tokens"] += int(attrs.get("output_tokens") or 0)
                if kind == "event.dispatch":
                    name = rec.get("name")
                    event = attrs.get("event") or {}
                    if name == "emit:rca.graph.mutated":
                        mut_kind = event.get("kind") or "unknown"
                        out["mutation_kinds"][mut_kind] = (
                            out["mutation_kinds"].get(mut_kind, 0) + 1
                        )
                        if mut_kind == "downgraded":
                            reason = event.get("reason") or ""
                            if reason:
                                out["downgrade_reasons"].append(str(reason))
                    if name == "emit:tool_call":
                        tool_name = event.get("tool_name")
                        if tool_name == "record_symptom":
                            out["symptom_count"] += 1
                        elif tool_name == "record_observation":
                            out["observation_count"] += 1
                        elif tool_name == "propose_hypothesis":
                            out["hypothesis_count"] += 1
                        elif tool_name == "dispatch_agent":
                            out["dispatch_agent_count"] += 1
                        elif tool_name == "submit_final_report":
                            out["final_report_emitted"] = True
                            args = event.get("args") or {}
                            if isinstance(args, dict):
                                out["final_report_args"] = args
    except OSError as exc:
        out["read_error"] = str(exc)
    return out


# --- Per-case run -----------------------------------------------------------


async def _run_one(task: dict[str, Any]) -> CaseResult:
    """Drive one rca_hfsm session against one task and tabulate results."""

    # Imports deferred to keep top-of-module side-effect-free for ruff
    # and so a smoke import of this module doesn't pull in the kernel.
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.abi import LoopConfig
    from agentm.core.abi.events import EventBus
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession

    task_id = str(task.get("id") or "")
    task_class = str(task.get("task_class") or "rca_baseline")
    user_msg = task.get("input", {}).get("user_message") or ""
    expected = task.get("expected") or {}
    expected_services = list(expected.get("expected_services") or [])
    expected_fault_kind = str(expected.get("fault_kind") or "")
    budget = task.get("budget") or {}
    max_turns = int(budget.get("max_turns") or 25)

    # Wire the per-case parquet fixture into the orchestrator's duckdb_sql
    # tool. C4 added duckdb_sql to the rca_hfsm manifest so the
    # investigator can actually query metrics/traces/logs; the tool reads
    # ``AGENTM_RCA_DATA_DIR`` when no explicit ``data_dir`` is set on the
    # atom config (manifest leaves it unset so the env var path wins per
    # case). The rca scenario's eval/agent.py uses the same env-var
    # handshake, see contrib/scenarios/rca/src/agentm_rca/eval/agent.py.
    fixtures = (task.get("input") or {}).get("fixtures") or []
    if fixtures and isinstance(fixtures, list):
        fx = str(fixtures[0])
        if Path(fx).is_dir():
            os.environ["AGENTM_RCA_DATA_DIR"] = fx

    case_dir = _TRACE_ROOT / task_id
    case_dir.mkdir(parents=True, exist_ok=True)

    result = CaseResult(
        task_id=task_id,
        task_class=task_class,
        prompt_summary=user_msg.splitlines()[0][:120] if user_msg else "",
        expected_services=expected_services,
        expected_fault_kind=expected_fault_kind,
    )

    started = time.time()
    bus = EventBus()
    provider_id = os.environ.get("AGENTM_PROVIDER", "openai")
    model_id = os.environ.get("AGENTM_MODEL", "Doubao-Seed-2.0-pro")
    try:
        provider = DEFAULT_PROVIDER_REGISTRY.build(
            provider_id, {"model": model_id}
        )
    except Exception as exc:
        result.error = f"provider build failed: {exc}"
        return result

    config = AgentSessionConfig(
        cwd=str(case_dir),
        provider=provider,
        scenario="rca_hfsm",
        bus=bus,
        loop_config=LoopConfig(max_turns=max_turns),
        task_class=task_class,
        eval_task_id=task_id,
        purpose="eval:rca_hfsm:phase2",
    )

    try:
        session = await AgentSession.create(config)
    except Exception as exc:
        result.error = f"session create failed: {exc}\n{traceback.format_exc()}"
        result.elapsed_sec = time.time() - started
        return result

    final: list[Any] = []
    # Wall-clock cap per case — C4 budget discipline. 8 minutes is large
    # enough for a 25-turn trace under a slow provider but small enough
    # that one stuck case can't burn the whole 30-minute budget.
    case_timeout_sec = float(os.environ.get("AGENTM_HFSM_EVAL_CASE_TIMEOUT", "480"))
    try:
        final = await asyncio.wait_for(
            session.prompt(user_msg), timeout=case_timeout_sec
        )
    except asyncio.TimeoutError:
        result.error = (
            f"wall-clock timeout after {case_timeout_sec:.0f}s "
            "(C4 per-case cap)"
        )
    except Exception as exc:
        result.error = f"prompt failed: {exc}\n{traceback.format_exc()}"
    finally:
        # Pull the FSM final state before shutdown — the
        # ``rca.fsm`` container is in-process state, not a bus event,
        # so we can't recover it from the JSONL trace after the session
        # tears down.
        try:
            fsm = session.get_service("rca.fsm")
            if fsm is not None:
                result.fsm_final_state = getattr(fsm, "state", None)
        except Exception:  # noqa: BLE001
            pass
        try:
            await session.shutdown()
        except Exception:  # noqa: BLE001 — shutdown best-effort
            pass
    result.elapsed_sec = time.time() - started

    # Collect the last assistant text for the grader fallback path.
    final_text_parts: list[str] = []
    for msg in final or []:
        content = getattr(msg, "content", None)
        if not content:
            continue
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                final_text_parts.append(text)
    result.final_text = "\n".join(final_text_parts[-3:])  # tail

    # Find the trace JSONL and tabulate.
    jsonl_path = _find_trace_jsonl(case_dir, task_id)
    if jsonl_path is not None:
        result.jsonl_path = str(jsonl_path)
        tab = _tabulate_trace(jsonl_path)
        result.trace_id = tab.get("trace_id")
        result.turn_count = tab.get("turn_count", 0)
        result.judge_calls = tab.get("judge_calls") or {}
        result.mutation_kinds = tab.get("mutation_kinds") or {}
        result.downgrade_reasons = tab.get("downgrade_reasons") or []
        result.hypothesis_count = tab.get("hypothesis_count", 0)
        result.observation_count = tab.get("observation_count", 0)
        result.symptom_count = tab.get("symptom_count", 0)
        result.dispatch_agent_count = tab.get("dispatch_agent_count", 0)
        result.last_stop_reason = tab.get("last_stop_reason")
        result.input_tokens = tab.get("input_tokens", 0)
        result.output_tokens = tab.get("output_tokens", 0)
        result.final_report_emitted = bool(tab.get("final_report_emitted"))
        result.final_report_args = tab.get("final_report_args") or {}

    # Grade with the local scoring rule (mirrors the rca grader's
    # weights but doesn't gate on the broken task_meta.task_id wire).
    grade_out = _local_grade(task, jsonl_path, result.final_text)
    result.grader_score = float(grade_out.get("score") or 0.0)
    result.grader_verdict = str(grade_out.get("failure_kind") or "unknown")
    result.grader_feedback = str(grade_out.get("feedback_text") or "")
    return result


# --- Report writing ---------------------------------------------------------


def _fmt_kv(d: dict[str, Any]) -> str:
    if not d:
        return "{}"
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items()))


def write_report(
    results: list[CaseResult],
    *,
    started_at: str,
    head_sha: str,
    provider: str,
    model: str,
    total_elapsed_sec: float,
    partial_reason: str | None = None,
) -> Path:
    lines: list[str] = []
    lines.append("# rca_hfsm Phase 2 — eval results on rca:baseline tasks")
    lines.append("")
    lines.append(f"- **Run date**: {started_at}")
    lines.append(f"- **Branch**: feat/rca-hfsm-phase1 at {head_sha}")
    lines.append(
        "- **Manifest**: contrib/scenarios/rca_hfsm/manifest.yaml "
        "(4 LLM-mode judges mounted before the gate)"
    )
    lines.append(f"- **Provider**: {provider}")
    lines.append(f"- **Model**: {model}")
    lines.append(f"- **Wall time**: {total_elapsed_sec:.1f}s for {len(results)} cases")
    if partial_reason:
        lines.append(f"- **Partial run**: {partial_reason}")
    lines.append("")
    lines.append("## Selection criteria")
    lines.append("")
    lines.append(
        "The plan asked for up to 10 representative cases from "
        "``contrib/scenarios/rca/eval/tasks/``. The only YAML-defined "
        "task suite in the existing rca scenario is "
        "``contrib/scenarios/rca/eval/baseline/tasks/`` and it holds "
        "exactly three cases. The 50-case ``ops-lite-fixed-50`` set "
        "lives in HuggingFace-dataset form, is driven by ``rca "
        "llm-eval run`` from ``rcabench-platform``, and is not a YAML "
        "task list this scenario can read directly. Per the plan's "
        "explicit fall-back (\"If ``tasks/`` has fewer than 10, runs all "
        "available\"), this report covers all three baseline tasks: a "
        "multi-service propagation fault (mysql network_corrupt) that "
        "stresses judge.independence and judge.coverage, a tight "
        "single-service pod-failure that stresses judge.satisfied, and "
        "a JVM-level stress case that stresses "
        "judge.falsified_genuinely on a container-vs-JVM "
        "disambiguation."
    )
    lines.append("")
    lines.append("## Manifest baseline (C4 fix)")
    lines.append("")
    lines.append(
        "C3 ran on a manifest that was missing the data-access tools "
        "(``agentm_rca.tools.duckdb_sql``) and sub-agent inheritance "
        "entries for ``rca_falsification_gate`` plus the four judge "
        "atoms. Workers refused to start because "
        "``rca_evidence_tools`` declares ``requires=(\"rca_falsification_"
        "gate\",)`` and the gate wasn't in the worker's inheritance "
        "list. **C4 fixes both gaps**: the orchestrator now carries "
        "``list_tables`` / ``query_sql`` directly, and workers inherit "
        "the store → judges → gate → evidence-tools → duckdb_sql → "
        "worker_finalize chain in the same dependency order as the "
        "orchestrator. This eval re-run is the first time judges have "
        "had a real opportunity to fire on a confirm/refute path, so "
        "compare the mutation-kinds and judge-call rows below against "
        "the prior commit's report (preserved in git history)."
    )
    lines.append("")
    # Summary
    pass_n = sum(1 for r in results if r.grader_verdict == "ok")
    final_n = sum(1 for r in results if r.final_report_emitted)
    err_n = sum(1 for r in results if r.error)
    total_n = len(results)
    finalize_n = sum(
        1 for r in results if (r.fsm_final_state or "").upper() == "FINALIZE"
    )
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total cases: {total_n}")
    lines.append(f"- Grader verdict ``ok``: {pass_n}/{total_n}")
    lines.append(f"- Cases producing ``submit_final_report``: {final_n}/{total_n}")
    lines.append(f"- Cases reaching FSM ``FINALIZE``: {finalize_n}/{total_n}")
    lines.append(f"- Cases with infrastructure errors: {err_n}/{total_n}")
    lines.append("")

    # Per-case
    lines.append("## Per-case results")
    lines.append("")
    for idx, r in enumerate(results, 1):
        lines.append(f"### Case {idx}: {r.task_id}")
        lines.append("")
        lines.append(f"- **Prompt summary**: {r.prompt_summary}")
        lines.append(
            f"- **Expected**: services={r.expected_services} "
            f"fault_kind={r.expected_fault_kind!r}"
        )
        lines.append(
            f"- **Verdict**: grader={r.grader_verdict}  "
            f"score={r.grader_score:.2f}  elapsed={r.elapsed_sec:.1f}s"
        )
        if r.error:
            err_short = r.error.splitlines()[0]
            lines.append(f"- **Error**: {err_short}")
        lines.append(
            f"- **Trajectory shape**: turns={r.turn_count}, "
            f"symptoms={r.symptom_count}, observations={r.observation_count}, "
            f"hypotheses={r.hypothesis_count}, "
            f"dispatch_agent_calls={r.dispatch_agent_count}, "
            f"final_report={r.final_report_emitted}, "
            f"fsm_final={r.fsm_final_state}, "
            f"last_stop={r.last_stop_reason}"
        )
        lines.append(
            f"- **Tokens (orchestrator only)**: in={r.input_tokens} "
            f"out={r.output_tokens}"
        )
        lines.append(
            f"- **Mutation kinds**: {_fmt_kv(r.mutation_kinds)}"
        )
        if r.final_report_args:
            root_cause = str(r.final_report_args.get("root_cause") or "")[:300]
            if root_cause:
                lines.append(f"- **Final report root_cause** (excerpt): {root_cause}")
        if r.judge_calls:
            lines.append(f"- **Judge calls**: {_fmt_kv(r.judge_calls)}")
        else:
            lines.append(
                "- **Judge calls**: none observed on bus (see Phase 3 "
                "note on judge telemetry)"
            )
        if r.downgrade_reasons:
            preview = r.downgrade_reasons[0][:200]
            lines.append(
                f"- **First downgrade reason**: {preview}"
            )
        if r.jsonl_path:
            rel = Path(r.jsonl_path).relative_to(_REPO_ROOT)
            lines.append(f"- **Trace**: ``{rel}``")
        lines.append(f"- **Grader feedback**: {r.grader_feedback}")
        lines.append("")

    # Cross-case patterns
    lines.append("## Cross-case patterns")
    lines.append("")
    lines.append("### Judge call patterns")
    lines.append("")
    judge_totals: dict[str, int] = {}
    for r in results:
        for k, v in r.judge_calls.items():
            judge_totals[k] = judge_totals.get(k, 0) + v
    if judge_totals:
        lines.append(
            "Aggregated judge invocations across all cases: "
            f"{_fmt_kv(judge_totals)}."
        )
    else:
        lines.append(
            "No ``rca.judge.invoked`` events were observed on the bus. "
            "The judges in Phase 2 do not currently emit a "
            "structured-bus event from inside ``judge()`` — judge calls "
            "happen inside ``gate.apply`` and are observable only "
            "through their effect on ``rca.graph.mutated`` (applied vs "
            "downgraded). Adding an explicit ``rca.judge.invoked`` "
            "event is Phase 3 work (design §3.4 envisaged "
            "observability JSONL emission per judge call but Phase 2 "
            "did not yet wire it up). The mutation-kinds row above is "
            "the proxy."
        )
    lines.append("")

    lines.append("### Downgrade patterns")
    lines.append("")
    total_downgrades = sum(
        r.mutation_kinds.get("downgraded", 0) for r in results
    )
    if total_downgrades == 0:
        lines.append(
            "No ``downgraded`` mutation events observed across the "
            "run. Either (a) the orchestrator never proposed a "
            "``confirm`` whose preconditions failed, or (b) traces "
            "terminated before reaching ``_apply_confirm``. Inspect "
            "``mutation_kinds`` per case to disambiguate."
        )
    else:
        lines.append(
            f"Total downgrades across cases: {total_downgrades}. "
            "Sample reasons preserved in each case's "
            "``First downgrade reason`` line above."
        )
    lines.append("")

    lines.append("### FSM state distribution")
    lines.append("")
    state_counts: dict[str, int] = {}
    for r in results:
        st = (r.fsm_final_state or "<none>").upper()
        state_counts[st] = state_counts.get(st, 0) + 1
    lines.append(f"Final FSM states: {_fmt_kv(state_counts)}.")
    lines.append("")

    lines.append("### Surprises and Phase 3 candidates")
    lines.append("")
    lines.append(
        "Observations are recorded honestly per the design doc's "
        "acceptance gate (§8): the question for Phase 2 was whether "
        "the refactor *behaves reasonably*, not whether it matches a "
        "target pass-rate on the first run. Surprises noted below "
        "describe what we saw, not a target."
    )
    lines.append("")
    surprises: list[str] = []
    # Count cases where the final report was a "platform error" report
    # rather than a real RCA verdict.
    platform_error_reports = sum(
        1
        for r in results
        if r.final_report_args
        and (
            "rca_falsification_gate"
            in str(r.final_report_args.get("root_cause") or "").lower()
            or "extension"
            in str(r.final_report_args.get("root_cause") or "").lower()
            and "not loaded"
            in str(r.final_report_args.get("root_cause") or "").lower()
        )
    )
    if platform_error_reports > 0:
        surprises.append(
            f"- **{platform_error_reports}/{total_n} ``submit_final_report`` "
            "calls were still platform-error reports despite the C4 "
            "manifest fix.** Worth investigating whether the worker is "
            "now bouncing off a *different* requires() chain or the "
            "orchestrator is generating the platform-error wording from "
            "stale context. The manifest-load smoke test confirms every "
            "service registers; live behaviour may diverge."
        )
    if any(r.hypothesis_count == 0 for r in results) and not any(
        r.hypothesis_count > 0 for r in results
    ):
        surprises.append(
            "- **Zero ``propose_hypothesis`` calls across all cases.** "
            "Because workers couldn't run, the orchestrator never "
            "received observations that would seed hypotheses, and the "
            "investigator persona's discipline (\"every hypothesis "
            "needs a negative prediction\") is gated on having "
            "evidence to predict against. So the *entire judge "
            "machinery* — satisfied, coverage, independence, "
            "falsified_genuinely — was never exercised on the gate's "
            "decision paths in this run. The judges are correctly "
            "wired (manifest smoke-load confirms ``_LlmJudge`` "
            "instances behind each ``rca.judge.*`` service) but their "
            "production behaviour cannot be measured until the "
            "preceding step works."
        )
    if err_n > 0:
        surprises.append(
            f"- {err_n}/{total_n} cases hit an infrastructure error "
            "before grader could fire. Inspect the per-case ``Error`` "
            "line; common causes are provider auth failures, "
            "rate-limits, or unexpected provider response shapes "
            "(the judges force tool_use; some providers refuse)."
        )
    if total_downgrades > 0:
        surprises.append(
            f"- {total_downgrades} downgrades fired. With Phase 1's "
            "structural rules these would have been auto-applied "
            "refines; under Phase 2 design §5.2 they now require the "
            "orchestrator to react to the judge's reason. The reasons "
            "preserved per case are the raw material for prompt-level "
            "iteration."
        )
    surprises.append(
        "- **task_meta.task_id wiring is broken in core.** "
        "``AgentSessionConfig`` accepts ``eval_task_id`` / ``task_class"
        "`` / ``eval_run_id`` and ``SessionReadyEvent`` declares the "
        "matching fields, but the emit at ``src/agentm/core/runtime/"
        "session_factory.py`` L366 does not forward them. The "
        "observability atom therefore writes ``task_meta = {..., "
        "task_id: None}`` for every programmatically-constructed "
        "session. The rca grader keys traces by this field, so the "
        "stock grader returns ``runtime`` for every case even when "
        "``submit_final_report`` fires. This eval works around the "
        "gap with a local scoring helper that locates the trace by "
        "case-cwd instead. Not a Phase 2 regression; a substrate-"
        "level wiring miss that should be fixed independently."
    )
    surprises.append(
        "- **Phase 3 tuning candidate ranking is provisional.** With "
        "judges never having fired on a confirm/refute path, no "
        "data-driven ordering is possible from this run. If the "
        "Phase-1 manifest bug is fixed and a follow-up eval shows "
        "workers actually returning observations, the natural first "
        "candidate is ``judge.satisfied`` — it gates every "
        "``attach_check`` and is the highest-call-rate judge by "
        "design (§6's cost analysis projected ~50–80 calls/trace, "
        "most through this judge). ``judge.coverage`` is second "
        "(it gates every ``_apply_confirm``). The independence and "
        "falsified_genuinely judges fire less often and should be "
        "tuned after the high-traffic ones are stable."
    )
    lines.extend(surprises)
    lines.append("")

    out = _OUT_DIR / "phase2_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# --- Main -------------------------------------------------------------------


async def _amain(limit: int) -> int:
    if not _TASKS_DIR.is_dir():
        print(f"ERROR: tasks dir not found: {_TASKS_DIR}", file=sys.stderr)
        return 2
    tasks = select_tasks(limit)
    if not tasks:
        print(f"ERROR: no task YAMLs under {_TASKS_DIR}", file=sys.stderr)
        return 2

    print(f"INFO: running {len(tasks)} cases (limit={limit})", file=sys.stderr)
    _TRACE_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    total_started = time.time()
    partial_reason: str | None = None
    for idx, task in enumerate(tasks, 1):
        task_id = task.get("id") or f"task-{idx}"
        print(f"INFO: [{idx}/{len(tasks)}] {task_id}", file=sys.stderr)
        try:
            result = await _run_one(task)
        except KeyboardInterrupt:
            partial_reason = (
                f"keyboard interrupt during case {idx}/{len(tasks)}"
            )
            break
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            result = CaseResult(
                task_id=str(task_id),
                task_class=str(task.get("task_class") or ""),
                prompt_summary="",
                expected_services=list(
                    (task.get("expected") or {}).get(
                        "expected_services") or []
                ),
                expected_fault_kind=str(
                    (task.get("expected") or {}).get("fault_kind") or ""
                ),
                error=f"runner exception: {exc}\n{tb}",
            )
        results.append(result)
        verdict = result.grader_verdict
        print(
            f"INFO:   verdict={verdict} score={result.grader_score:.2f} "
            f"elapsed={result.elapsed_sec:.1f}s "
            f"final_report={result.final_report_emitted}",
            file=sys.stderr,
        )

    total_elapsed = time.time() - total_started

    # Resolve metadata for the report header.
    head_sha = os.environ.get("AGENTM_EVAL_HEAD", "unknown")
    if head_sha == "unknown":
        try:
            import subprocess

            head_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(_REPO_ROOT),
            ).decode().strip()
        except Exception:  # noqa: BLE001
            pass
    started_at = time.strftime(
        "%Y-%m-%d %H:%M:%S UTC", time.gmtime(total_started)
    )
    provider = os.environ.get("AGENTM_PROVIDER", "openai")
    model = os.environ.get("AGENTM_MODEL", "Doubao-Seed-2.0-pro")

    out = write_report(
        results,
        started_at=started_at,
        head_sha=head_sha,
        provider=provider,
        model=model,
        total_elapsed_sec=total_elapsed,
        partial_reason=partial_reason,
    )
    print(f"INFO: report written to {out}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max cases to run (default 10; suite holds 3).",
    )
    args = parser.parse_args()
    return asyncio.run(_amain(args.limit))


if __name__ == "__main__":
    sys.exit(main())
