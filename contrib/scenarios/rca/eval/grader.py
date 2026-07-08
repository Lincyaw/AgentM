"""Programmatic grader for the rca:baseline eval suite.

Returns the μ_f feedback shape (per design §3.2). The agent terminates by
calling ``submit_final_report`` (see ``contrib/scenarios/rca/src/.../finalize.py``)
whose args carry the verdict. Current runs use fpg ``ModelRCAOutput``:
``root_causes[]`` contains node ids, and the corresponding ``nodes[]`` entries
carry ``subject`` / ``predicate``. Older traces with rcabench
``root_causes[].service`` / ``root_causes[].fault_kind`` are still accepted.

Score components (per the task brief):
  service_hit    1.0 if any expected service appears in the verdict
  fault_kind_hit 1.0 if expected.fault_kind substring appears in the verdict
  score          0.7 * service_hit + 0.3 * fault_kind_hit

When the task points at a case directory containing
``causal_graph_verified.json`` and the verdict is fpg ``ModelRCAOutput``, the
grader delegates graph scoring to ``fpg.compare_model_to_ground_truth``. The
root-only score above remains the fallback for legacy tasks and old fpg
packages that do not yet provide the comparison helper.

Plus a ``module_feedback`` channel that fingers ``query_sql`` whenever the
trace shows a Binder Error / quoting failure, so the GEPA tuner can target
the investigator prompt's SQL guidance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from agentm.core.abi import TraceReader
from agentm.core.lib import cap_duckdb_threads
from agentm.core.lib.observability_dir import resolve_observability_dir

# The grader runs in the same process as tool_eval_run; resolve from cwd so
# AGENTM_OBSERVABILITY_DIR and AGENTM_HOME follow the same rules as agentm.
_OBS_DIR = resolve_observability_dir(Path.cwd())
_SQL_EVIDENCE_EXCLUDE = frozenset({"conclusion.parquet"})
_SQL_ALLOWED_HEADS = {"SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"}
_SQL_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|COPY|"
    r"PRAGMA|EXPORT|IMPORT|INSTALL|LOAD|CALL|SET)\b",
    re.IGNORECASE,
)
_SQL_HELPER_MACROS = (
    "CREATE OR REPLACE MACRO p50(x) AS quantile_cont(x, 0.5)",
    "CREATE OR REPLACE MACRO p90(x) AS quantile_cont(x, 0.9)",
    "CREATE OR REPLACE MACRO p95(x) AS quantile_cont(x, 0.95)",
    "CREATE OR REPLACE MACRO p99(x) AS quantile_cont(x, 0.99)",
)


def grade(task: dict[str, Any], output: str) -> dict[str, Any]:
    expected = task.get("expected") or {}
    expected_services = [
        s for s in (expected.get("expected_services") or []) if isinstance(s, str)
    ]
    expected_fault_kind = str(expected.get("fault_kind") or "").strip().lower()

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
        verdict = _verdict_from_output_text(output or "")
        trace_missing = True

    module_feedback: dict[str, str] = {}
    sql_hint = _detect_sql_quoting_issue(task)
    if sql_hint:
        module_feedback["query_sql"] = sql_hint

    sql_evidence_eval = _evaluate_fpg_sql_evidence(task, verdict)
    sql_evidence_feedback = _sql_evidence_module_feedback(sql_evidence_eval)
    if sql_evidence_feedback:
        module_feedback["evidence_sql"] = sql_evidence_feedback

    graph_comparison = _compare_fpg_graph(task, verdict)
    if graph_comparison is not None:
        graph_score = float(graph_comparison.get("score") or 0.0)
        dimensions = _graph_dimensions(graph_comparison)
        dimensions.update(_sql_evidence_dimensions(sql_evidence_eval))
        sql_ok = sql_evidence_eval is None or bool(
            sql_evidence_eval.get("all_executable")
        )
        return _result(
            graph_score,
            dimensions=dimensions,
            feedback_text=_join_feedback(
                _summarize_graph_comparison(graph_comparison),
                _summarize_sql_evidence(sql_evidence_eval),
            ),
            module_feedback=module_feedback,
            failure_kind="ok" if graph_score >= 0.999 and sql_ok else "correctness",
        )

    if not expected_services and not expected_fault_kind:
        # Eval-suite gap — not the agent's fault; flag as a correctness
        # contract violation so upstream observers don't treat it as
        # noise.
        module_feedback["eval_suite"] = (
            "task YAML lacks expected fields and no usable "
            "causal_graph_verified.json graph comparison was available"
        )
        return _result(
            0.0,
            dimensions=_sql_evidence_dimensions(sql_evidence_eval),
            feedback_text=_join_feedback(
                "task is missing expected.expected_services and expected.fault_kind",
                _summarize_sql_evidence(sql_evidence_eval),
            ),
            module_feedback=module_feedback,
            failure_kind="runtime" if trace_missing else "correctness",
        )

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
        dimensions=_sql_evidence_dimensions(sql_evidence_eval),
        feedback_text=_join_feedback(
            feedback_text, _summarize_sql_evidence(sql_evidence_eval)
        ),
        module_feedback=module_feedback,
        failure_kind=failure_kind,
    )


# ---------------------------------------------------------------------------


def _result(
    score: float,
    *,
    dimensions: dict[str, float] | None = None,
    feedback_text: str,
    module_feedback: dict[str, str],
    failure_kind: str,
) -> dict[str, Any]:
    return {
        "score": float(score),
        "dimensions": dimensions or {},
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
            "tool emit (no matching task_id in the observability log)."
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
    """Walk the most recent trace whose task metadata matches this eval case."""
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
    """Parse an OTLP/JSON trace for the verdict matching ``expected_task_id``.

    Identity comes from the ``agentm.session.fingerprint`` log record's
    ``task_meta.task_id``. The verdict args come from the
    ``execute_tool submit_final_report`` span's
    ``gen_ai.tool.call.arguments`` attribute (JSON-encoded by the writer).
    """
    if not path.is_file():
        return None
    reader = TraceReader(path)

    # Identity match: look for an ``agentm.session.fingerprint`` whose
    # ``task_meta.task_id`` is the one we're scoring.
    matched = False
    for record in reader.iter_log_records(name="agentm.session.fingerprint"):
        body = record.body
        if isinstance(body, dict):
            task_meta = body.get("task_meta") or {}
            if (
                isinstance(task_meta, dict)
                and str(task_meta.get("task_id") or "") == expected_task_id
            ):
                matched = True
                break
    if not matched:
        return None

    services: list[str] = []
    fault_kinds: list[str] = []
    fpg_output: dict[str, Any] | None = None
    raw_payload: str = ""
    for span in reader.iter_spans(name="execute_tool submit_final_report"):
        raw = span.attributes.get("gen_ai.tool.call.arguments")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            args = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(args, dict):
            continue
        raw_payload = json.dumps(args)
        _collect_old_rcabench_verdict(args, services, fault_kinds)
        _collect_fpg_verdict(args, services, fault_kinds)
        if _is_fpg_output(args):
            fpg_output = args
    return {
        "services": services,
        "fault_kinds": fault_kinds,
        "raw": raw_payload,
        "fpg_output": fpg_output,
    }


def _verdict_from_output_text(output: str) -> dict[str, Any]:
    services: list[str] = []
    fault_kinds: list[str] = []
    fpg_output: dict[str, Any] | None = None
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        _collect_old_rcabench_verdict(parsed, services, fault_kinds)
        _collect_fpg_verdict(parsed, services, fault_kinds)
        if _is_fpg_output(parsed):
            fpg_output = parsed
    return {
        "services": services,
        "fault_kinds": fault_kinds,
        "raw": output,
        "fpg_output": fpg_output,
    }


def _is_fpg_output(args: dict[str, Any]) -> bool:
    return isinstance(args.get("nodes"), list) and isinstance(
        args.get("root_causes"), list
    )


def _collect_old_rcabench_verdict(
    args: dict[str, Any], services: list[str], fault_kinds: list[str]
) -> None:
    for rc in args.get("root_causes") or []:
        if not isinstance(rc, dict):
            continue
        svc = rc.get("service")
        if isinstance(svc, str):
            services.append(svc)
        fk = rc.get("fault_kind")
        if isinstance(fk, str):
            fault_kinds.append(fk)


def _collect_fpg_verdict(
    args: dict[str, Any], services: list[str], fault_kinds: list[str]
) -> None:
    nodes = args.get("nodes")
    roots = args.get("root_causes")
    if not isinstance(nodes, list) or not isinstance(roots, list):
        return
    by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if isinstance(node_id, str):
            by_id[node_id] = node
    for root_id in roots:
        if not isinstance(root_id, str):
            continue
        node = by_id.get(root_id)
        if node is None:
            continue
        subject = node.get("subject")
        if isinstance(subject, str):
            services.append(_subject_name(subject))
        predicate = node.get("predicate")
        if isinstance(predicate, str):
            fault_kinds.append(predicate)


def _subject_name(subject: str) -> str:
    if ":" not in subject:
        return subject
    return subject.split(":", 1)[1]


def _compare_fpg_graph(
    task: dict[str, Any], verdict: dict[str, Any]
) -> dict[str, Any] | None:
    payload = verdict.get("fpg_output")
    if not isinstance(payload, dict):
        return None
    graph_path = _verified_graph_path(task)
    if graph_path is None:
        return None
    try:
        from fpg import ModelRCAOutput, Scenario, compare_model_to_ground_truth
    except ImportError:
        return None
    try:
        model_output = ModelRCAOutput.model_validate(payload)
        scenario = Scenario.model_validate_json(graph_path.read_text(encoding="utf-8"))
        comparison = compare_model_to_ground_truth(model_output, scenario)
    except Exception as exc:
        logger.warning("fpg graph comparison failed: {}", exc)
        return None
    return comparison.model_dump(mode="json")


def _case_dir_candidates(task: dict[str, Any]) -> list[Path]:
    expected = task.get("expected") if isinstance(task.get("expected"), dict) else {}
    meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
    case_dirs: list[Path] = []
    for source in (task, expected, meta):
        if not isinstance(source, dict):
            continue
        for key in ("path", "data_dir", "case_dir"):
            raw = source.get(key)
            if isinstance(raw, str):
                case_dirs.append(Path(raw))

    source = task.get("source")
    source_path = task.get("source_path")
    if isinstance(source, str) and isinstance(source_path, str):
        case_dirs.append(Path(source_path) / source)
    return case_dirs


def _case_dir_path(task: dict[str, Any]) -> Path | None:
    for case_dir in _case_dir_candidates(task):
        expanded = case_dir.expanduser()
        if expanded.is_dir():
            return expanded
    return None


def _verified_graph_path(task: dict[str, Any]) -> Path | None:
    expected = task.get("expected") if isinstance(task.get("expected"), dict) else {}
    meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
    for source in (task, expected, meta):
        if not isinstance(source, dict):
            continue
        for key in (
            "causal_graph_verified_path",
            "verified_graph_path",
            "ground_truth_graph_path",
        ):
            raw = source.get(key)
            if isinstance(raw, str):
                path = Path(raw)
                if path.is_file():
                    return path

    for case_dir in _case_dir_candidates(task):
        path = case_dir / "causal_graph_verified.json"
        if path.is_file():
            return path
    return None


def _evaluate_fpg_sql_evidence(
    task: dict[str, Any], verdict: dict[str, Any]
) -> dict[str, Any] | None:
    payload = verdict.get("fpg_output")
    if not isinstance(payload, dict):
        return None
    statements = list(_iter_fpg_sql_evidence(payload))
    if not statements:
        return {
            "total": 0,
            "executable": 0,
            "failed": 0,
            "ratio": 0.0,
            "all_executable": True,
            "failures": [],
        }
    case_dir = _case_dir_path(task)
    if case_dir is None:
        return {
            "total": len(statements),
            "executable": 0,
            "failed": len(statements),
            "ratio": 0.0,
            "all_executable": False,
            "failures": [
                {
                    "error": "case directory not found; cannot replay evidence SQL",
                    "node_id": item["node_id"],
                    "evidence_index": item["evidence_index"],
                }
                for item in statements[:5]
            ],
        }
    try:
        import duckdb  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        return {
            "total": len(statements),
            "executable": 0,
            "failed": len(statements),
            "ratio": 0.0,
            "all_executable": False,
            "failures": [
                {
                    "error": "duckdb is not installed; cannot replay evidence SQL",
                    "node_id": item["node_id"],
                    "evidence_index": item["evidence_index"],
                }
                for item in statements[:5]
            ],
        }

    failures: list[dict[str, Any]] = []
    executable = 0
    try:
        conn = duckdb.connect(":memory:")
        cap_duckdb_threads(conn)
        _install_sql_eval_macros(conn)
        _register_case_parquets(conn, case_dir)
    except Exception as exc:  # noqa: BLE001 - setup failure is grader feedback
        logger.warning("SQL eval setup failed: {}", exc)
        return {
            "total": len(statements),
            "executable": 0,
            "failed": len(statements),
            "ratio": 0.0,
            "all_executable": False,
            "failures": [
                {
                    "error": f"failed to initialize evidence SQL replay: {exc}",
                    "node_id": item["node_id"],
                    "evidence_index": item["evidence_index"],
                }
                for item in statements[:5]
            ],
        }

    try:
        for item in statements:
            validated = _validate_evidence_sql(str(item["statement"]))
            if isinstance(validated, str):
                failures.append({**_sql_failure_context(item), "error": validated})
                continue
            sql, head = validated
            try:
                conn.execute(_wrap_evidence_sql(sql, head)).fetchone()
                executable += 1
            except duckdb.Error as exc:
                failures.append(
                    {**_sql_failure_context(item), "error": str(exc)[:500]}
                )
    finally:
        conn.close()

    total = len(statements)
    failed = total - executable
    return {
        "total": total,
        "executable": executable,
        "failed": failed,
        "ratio": executable / total if total else 0.0,
        "all_executable": failed == 0,
        "failures": failures[:5],
    }


def _iter_fpg_sql_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return items
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        evidence = node.get("evidence")
        if not isinstance(evidence, list):
            continue
        for index, entry in enumerate(evidence):
            if not isinstance(entry, dict):
                continue
            query = entry.get("query")
            if not isinstance(query, dict):
                continue
            language = query.get("language")
            statement = query.get("statement")
            if not isinstance(language, str) or language.lower() != "sql":
                continue
            if not isinstance(statement, str) or not statement.strip():
                continue
            items.append(
                {
                    "node_id": str(node_id) if isinstance(node_id, str) else "",
                    "subject": node.get("subject") if isinstance(node.get("subject"), str) else "",
                    "predicate": node.get("predicate")
                    if isinstance(node.get("predicate"), str)
                    else "",
                    "evidence_index": index,
                    "statement": statement,
                }
            )
    return items


def _validate_evidence_sql(sql_raw: str) -> tuple[str, str] | str:
    sql_raw = sql_raw.strip()
    if not sql_raw:
        return "sql is required"
    sql = sql_raw.rstrip(";").strip()
    if ";" in sql:
        return "only one statement is allowed"
    if _SQL_WRITE_KEYWORDS.search(sql):
        return "only read-only SQL evidence is allowed"
    head = sql.lstrip().split(None, 1)[0].upper() if sql.lstrip() else ""
    if head not in _SQL_ALLOWED_HEADS:
        return (
            f"unsupported leading keyword: {head!r}; "
            "use SELECT/WITH/EXPLAIN/DESCRIBE/SHOW/SUMMARIZE"
        )
    return sql, head


def _wrap_evidence_sql(sql: str, head: str) -> str:
    if head in {"EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"}:
        return sql
    return f"SELECT * FROM ({sql}) LIMIT 1"


def _install_sql_eval_macros(conn: Any) -> None:
    for stmt in _SQL_HELPER_MACROS:
        conn.execute(stmt)


def _register_case_parquets(conn: Any, case_dir: Path) -> None:
    for parquet in sorted(case_dir.iterdir()):
        if (
            not parquet.is_file()
            or parquet.suffix != ".parquet"
            or parquet.name in _SQL_EVIDENCE_EXCLUDE
        ):
            continue
        view = parquet.stem
        path = parquet.as_posix().replace("'", "''")
        conn.execute(
            f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{path}')"
        )


def _sql_failure_context(item: dict[str, Any]) -> dict[str, Any]:
    context = {
        "node_id": item.get("node_id") or "",
        "subject": item.get("subject") or "",
        "predicate": item.get("predicate") or "",
        "evidence_index": item.get("evidence_index"),
    }
    statement = str(item.get("statement") or "")
    if statement:
        context["statement"] = " ".join(statement.split())[:240]
    return context


def _sql_evidence_dimensions(sql_eval: dict[str, Any] | None) -> dict[str, float]:
    if sql_eval is None:
        return {}
    return {
        "fpg_sql_evidence_count": float(sql_eval.get("total") or 0),
        "fpg_sql_executable_count": float(sql_eval.get("executable") or 0),
        "fpg_sql_failed_count": float(sql_eval.get("failed") or 0),
        "fpg_sql_executable_ratio": float(sql_eval.get("ratio") or 0.0),
        "fpg_sql_all_executable": 1.0
        if bool(sql_eval.get("all_executable"))
        else 0.0,
    }


def _summarize_sql_evidence(sql_eval: dict[str, Any] | None) -> str | None:
    if sql_eval is None:
        return None
    total = int(sql_eval.get("total") or 0)
    executable = int(sql_eval.get("executable") or 0)
    failed = int(sql_eval.get("failed") or 0)
    if total == 0:
        return "FPG SQL evidence: no sql evidence statements to replay."
    return (
        "FPG SQL evidence executable="
        f"{executable}/{total} ({float(sql_eval.get('ratio') or 0.0):.3f}); "
        f"failed={failed}."
    )


def _sql_evidence_module_feedback(sql_eval: dict[str, Any] | None) -> str | None:
    if sql_eval is None or bool(sql_eval.get("all_executable")):
        return None
    failures = sql_eval.get("failures")
    if not isinstance(failures, list) or not failures:
        return "Final fpg evidence SQL could not be replayed."
    return "Final fpg evidence SQL replay failures: " + _brief_list(failures, limit=3)


def _join_feedback(*parts: str | None) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _graph_dimensions(comparison: dict[str, Any]) -> dict[str, float]:
    dimensions: dict[str, float] = {"fpg_score": float(comparison.get("score") or 0.0)}
    for name in (
        "root_nodes",
        "root_subjects",
        "nodes",
        "subjects",
        "edges",
        "soft_edges",
        "subject_edges",
        "soft_subject_edges",
    ):
        metric = comparison.get(name)
        if not isinstance(metric, dict):
            continue
        for field in ("precision", "recall", "f1"):
            raw = metric.get(field)
            if isinstance(raw, (int, float)):
                dimensions[f"fpg_{name}_{field}"] = float(raw)
    for key in (
        "exact_path_match_hit",
        "subject_path_match_hit",
        "path_reachability_hit",
        "subject_path_reachability_hit",
    ):
        dimensions[f"fpg_{key}"] = 1.0 if comparison.get(key) is True else 0.0
    return dimensions


def _summarize_graph_comparison(comparison: dict[str, Any]) -> str:
    score = float(comparison.get("score") or 0.0)
    root_subjects = _metric_f1(comparison, "root_subjects")
    subjects = _metric_f1(comparison, "subjects")
    soft_subject_edges = _metric_f1(comparison, "soft_subject_edges")
    exact_nodes = _metric_f1(comparison, "nodes")
    subject_edges = _metric_f1(comparison, "subject_edges")
    return (
        f"FPG graph score={score:.3f}; "
        f"root_subject_f1={root_subjects:.3f}, subject_f1={subjects:.3f}, "
        f"soft_subject_edge_f1={soft_subject_edges:.3f}, "
        f"subject_edge_f1={subject_edges:.3f}, exact_node_f1={exact_nodes:.3f}, "
        f"subject_path_hit={bool(comparison.get('subject_path_match_hit'))}, "
        "subject_path_reachability_hit="
        f"{bool(comparison.get('subject_path_reachability_hit'))}. "
        f"Missing nodes: {_brief_list(comparison.get('missing_nodes'))}; "
        f"extra nodes: {_brief_list(comparison.get('extra_nodes'))}; "
        f"missing edges: {_brief_list(comparison.get('missing_edges'))}; "
        f"extra edges: {_brief_list(comparison.get('extra_edges'))}; "
        f"missing paths: {_brief_list(comparison.get('missing_paths'))}; "
        f"extra paths: {_brief_list(comparison.get('extra_paths'))}."
    )


def _metric_f1(comparison: dict[str, Any], key: str) -> float:
    metric = comparison.get(key)
    if not isinstance(metric, dict):
        return 0.0
    raw = metric.get("f1")
    return float(raw) if isinstance(raw, (int, float)) else 0.0


def _brief_list(value: Any, *, limit: int = 5) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    shown = [str(item) for item in value[:limit]]
    suffix = f" (+{len(value) - limit} more)" if len(value) > limit else ""
    return ", ".join(shown) + suffix


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
