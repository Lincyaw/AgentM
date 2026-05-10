"""Programmatic grader for the rca_single eval suite.

Returns the μ_f feedback shape (per design §3.2). The agent terminates by
calling ``submit_final_report`` (see ``contrib/scenarios/rca/src/.../finalize.py``)
whose args carry the verdict — ``root_causes[].service`` and
``root_causes[].fault_kind``. The args land on the ``emit:tool_call`` event
under ``attributes.event.args`` (NOT ``attributes.args``).

Score components (per the task brief):
  service_hit    1.0 if any expected service appears in the verdict
  fault_kind_hit 1.0 if expected.fault_kind substring appears in verdict
  score          0.7 * service_hit + 0.3 * fault_kind_hit

Plus a ``module_feedback`` channel that fingers ``query_sql`` whenever the
trace shows a Binder Error / quoting failure, so the GEPA tuner can target
the investigator prompt's SQL guidance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# tool_eval_run cd's into the tuner cwd; observability lives under
# ``<cwd>/.agentm/observability/<trace>.jsonl``. The grader runs in the same
# process, so plain Path().cwd() resolves to the tuner cwd.
_OBS_DIR = Path(".agentm") / "observability"


def grade(task: dict[str, Any], output: str) -> dict[str, Any]:
    expected = task.get("expected") or {}
    expected_services = [
        s for s in (expected.get("expected_services") or []) if isinstance(s, str)
    ]
    expected_fault_kind = str(expected.get("fault_kind") or "").strip().lower()

    if not expected_services and not expected_fault_kind:
        # Eval-suite gap — not the agent's fault; flag as a correctness
        # contract violation so upstream observers don't treat it as
        # noise.
        return _result(
            0.0,
            feedback_text="task is missing expected.expected_services and expected.fault_kind",
            module_feedback={"eval_suite": "task YAML lacks expected fields"},
            failure_kind="correctness",
        )

    trace_missing = False
    verdict = _extract_verdict_from_trace(task)
    if verdict is None:
        # Fall back to scanning the assistant's final text — better than 0.0
        # if the agent happened to summarize without the tool call (rare).
        # Treat missing-trace as ``runtime``: the agent never produced a
        # reachable submit_final_report tool call, which is closer to a
        # crash/hang than a wrong answer.
        # TODO: detect timeouts — needs tool_eval_run to surface
        #   per-sample wall-clock; currently we cannot distinguish a
        #   crashed sub-agent from one that hit the turn cap.
        verdict = {"services": [], "fault_kinds": [], "raw": output or ""}
        trace_missing = True

    raw_blob = (verdict.get("raw") or "") + " " + " ".join(
        verdict.get("services") or []
    ) + " " + " ".join(verdict.get("fault_kinds") or [])
    raw_lower = raw_blob.lower()

    service_hit = 1.0 if any(
        svc.lower() in raw_lower for svc in expected_services
    ) else 0.0
    fault_kind_hit = (
        1.0 if expected_fault_kind and expected_fault_kind in raw_lower else 0.0
    )
    score = 0.7 * service_hit + 0.3 * fault_kind_hit

    feedback_text = _summarize(
        expected_services=expected_services,
        expected_fault_kind=expected_fault_kind,
        verdict=verdict,
        service_hit=service_hit,
        fault_kind_hit=fault_kind_hit,
    )

    module_feedback: dict[str, str] = {}
    sql_hint = _detect_sql_quoting_issue(task)
    if sql_hint:
        module_feedback["query_sql"] = sql_hint

    # failure_kind taxonomy:
    #   "ok"          full hit on services + fault_kind
    #   "runtime"     no submit_final_report trace — agent crashed/hung
    #   "correctness" verdict produced but missed expected
    # We don't emit "regression" here; that's a comparison the gate
    # would draw against a baseline run, not a single-sample property.
    if trace_missing:
        failure_kind = "runtime"
    elif service_hit == 1.0 and fault_kind_hit == 1.0:
        failure_kind = "ok"
    else:
        failure_kind = "correctness"

    return _result(
        score,
        feedback_text=feedback_text,
        module_feedback=module_feedback,
        failure_kind=failure_kind,
    )


# ---------------------------------------------------------------------------


def _result(
    score: float,
    *,
    feedback_text: str,
    module_feedback: dict[str, str],
    failure_kind: str,
) -> dict[str, Any]:
    return {
        "score": float(score),
        "dimensions": {},
        "feedback_text": feedback_text,
        "module_feedback": module_feedback,
        "failure_kind": failure_kind,
    }


def _summarize(
    *,
    expected_services: list[str],
    expected_fault_kind: str,
    verdict: dict[str, Any],
    service_hit: float,
    fault_kind_hit: float,
) -> str:
    got_services = verdict.get("services") or []
    got_fault_kinds = verdict.get("fault_kinds") or []
    if service_hit == 1.0 and fault_kind_hit == 1.0:
        return (
            f"Verdict matched: services={sorted(set(got_services))} "
            f"fault_kinds={sorted(set(got_fault_kinds))}"
        )
    if not got_services and not got_fault_kinds and not (verdict.get("raw") or ""):
        return (
            "Could not extract agent verdict from trace; check submit_final_report "
            "tool emit (no matching task_id in .agentm/observability)."
        )
    parts = []
    if service_hit == 1.0:
        parts.append(
            f"named expected service ({_first_overlap(expected_services, got_services)})"
        )
    else:
        parts.append(
            f"missed services (got: {sorted(set(got_services)) or 'none'}; "
            f"expected: {expected_services})"
        )
    if fault_kind_hit == 1.0:
        parts.append(f"fault_kind '{expected_fault_kind}' present")
    else:
        parts.append(
            f"missed fault_kind (got: {sorted(set(got_fault_kinds)) or 'none'}; "
            f"expected substring: '{expected_fault_kind}')"
        )
    return "; ".join(parts).capitalize() + "."


def _first_overlap(expected: list[str], got: list[str]) -> str:
    got_lower = {g.lower() for g in got}
    for e in expected:
        if e.lower() in got_lower:
            return e
    # Fallback for substring hits (e.g. expected "mysql" matched inside a
    # got string like "mysql-primary").
    return expected[0] if expected else ""


def _extract_verdict_from_trace(task: dict[str, Any]) -> dict[str, Any] | None:
    """Walk the most recent trace under .agentm/observability whose
    task_meta.task_id matches ``task['id']`` and an eval_run_id is set.
    Pull ``submit_final_report`` args (root_causes[]) into a flat verdict.
    """
    task_id = str(task.get("id") or "")
    if not task_id or not _OBS_DIR.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for p in _OBS_DIR.glob("*.jsonl"):
        if not p.is_file():
            continue
        candidates.append((p.stat().st_mtime, p))
    candidates.sort(key=lambda kv: kv[0], reverse=True)
    for _mtime, path in candidates[:50]:
        verdict = _parse_trace(path, task_id)
        if verdict is not None:
            return verdict
    return None


def _parse_trace(path: Path, expected_task_id: str) -> dict[str, Any] | None:
    services: list[str] = []
    fault_kinds: list[str] = []
    raw_payload: str = ""
    matched = False
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
                if not isinstance(rec, dict):
                    continue
                kind = rec.get("kind")
                attrs = rec.get("attributes") or {}
                if kind == "session.fingerprint":
                    task_meta = attrs.get("task_meta") or {}
                    if (
                        isinstance(task_meta, dict)
                        and str(task_meta.get("task_id") or "") == expected_task_id
                    ):
                        matched = True
                if kind == "event.dispatch" and rec.get("name") == "emit:tool_call":
                    event = attrs.get("event") or {}
                    if (
                        isinstance(event, dict)
                        and event.get("tool_name") == "submit_final_report"
                    ):
                        args = event.get("args") or {}
                        if isinstance(args, dict):
                            raw_payload = json.dumps(args)
                            for rc in args.get("root_causes") or []:
                                if not isinstance(rc, dict):
                                    continue
                                svc = rc.get("service")
                                if isinstance(svc, str):
                                    services.append(svc)
                                fk = rc.get("fault_kind")
                                if isinstance(fk, str):
                                    fault_kinds.append(fk)
    except OSError:
        return None
    if not matched:
        return None
    return {"services": services, "fault_kinds": fault_kinds, "raw": raw_payload}


def _detect_sql_quoting_issue(task: dict[str, Any]) -> str | None:
    """Best-effort: scan this task's trace for the canonical Binder Error
    that fires when the agent uses an unquoted ``attr.*`` column name.
    """
    task_id = str(task.get("id") or "")
    if not task_id or not _OBS_DIR.is_dir():
        return None
    for path in sorted(
        _OBS_DIR.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:20]:
        try:
            with path.open("r", encoding="utf-8") as fh:
                body = fh.read()
        except OSError:
            continue
        if task_id not in body:
            continue
        # The trace body is JSON-encoded twice (tool result text is itself
        # JSON, then wrapped in the event JSON), so escape forms vary; match
        # the two substrings independently rather than a fully escaped phrase.
        if "Binder Error" in body and "Referenced table" in body and "attr" in body:
            return (
                "First SQL hit a DuckDB Binder Error on an unquoted attr.* "
                "column; agent self-corrected on retry. Adding schema-quoting "
                "guidance to the investigator prompt would prevent this."
            )
        return None
    return None


__all__ = ["grade"]
