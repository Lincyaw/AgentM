"""Standalone re-judge for eval runs whose fpg ModelRCAOutput was scored 0
by rcabench-platform's evaluate_v2 (schema mismatch: fpg root_causes are
node-id strings, platform expects RootCauseClaim objects).

Reads rollout responses from eval.db, scores each via the fpg graph
comparison path (causal_graph_verified.json), and writes results back to
eval_metrics / correct / confidence.

Usage:
    uv run python contrib/scenarios/rca/eval/rejudge_fpg.py \
        --exp-id agentm-rca-opslite-clean-baseline-litellm \
        --case-root datasets/ops-lite-clean/cases \
        [--db eval.db] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


def _fpg_graph_score(
    response_json: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any] | None:
    graph_path = case_dir / "causal_graph_verified.json"
    if not graph_path.is_file():
        return None
    try:
        from fpg import ModelRCAOutput, Scenario, compare_model_to_ground_truth
    except ImportError:
        print("ERROR: fpg package not installed", file=sys.stderr)
        return None

    nodes = response_json.get("nodes", [])
    root_causes = response_json.get("root_causes", [])
    if not isinstance(nodes, list) or not isinstance(root_causes, list):
        return None
    if not nodes and not root_causes:
        return {"score": 0.0, "_empty": True}

    try:
        model_output = ModelRCAOutput.model_validate(response_json)
        scenario = Scenario.model_validate_json(
            graph_path.read_text(encoding="utf-8")
        )
        comparison = compare_model_to_ground_truth(model_output, scenario)
    except Exception as exc:
        return {"score": 0.0, "_error": str(exc)}
    return comparison.model_dump(mode="json")


def _extract_services_and_fault_kinds(
    response_json: dict[str, Any],
) -> tuple[list[str], list[str]]:
    services: list[str] = []
    fault_kinds: list[str] = []
    nodes = response_json.get("nodes")
    roots = response_json.get("root_causes")
    if not isinstance(nodes, list) or not isinstance(roots, list):
        return services, fault_kinds
    by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if isinstance(node, dict) and isinstance(node.get("id"), str):
            by_id[node["id"]] = node
    for root_id in roots:
        if not isinstance(root_id, str):
            continue
        node = by_id.get(root_id)
        if node is None:
            continue
        subject = node.get("subject")
        if isinstance(subject, str):
            if ":" in subject:
                services.append(subject.split(":", 1)[1])
            else:
                services.append(subject)
        predicate = node.get("predicate")
        if isinstance(predicate, str):
            fault_kinds.append(predicate)
    return services, fault_kinds


def _gt_services_from_injection(case_dir: Path) -> list[str]:
    inj_path = case_dir / "injection.json"
    if not inj_path.is_file():
        return []
    try:
        inj = json.loads(inj_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    services: list[str] = []
    for ec in inj.get("engine_config_summary", []):
        if isinstance(ec, dict):
            app = ec.get("app")
            if isinstance(app, str):
                services.append(app)
    return services


def _sql_evidence_replay(
    response_json: dict[str, Any], case_dir: Path
) -> dict[str, Any] | None:
    try:
        import duckdb
    except ImportError:
        return None

    nodes = response_json.get("nodes")
    if not isinstance(nodes, list):
        return None

    statements: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        evidence = node.get("evidence")
        if not isinstance(evidence, list):
            continue
        for idx, entry in enumerate(evidence):
            if not isinstance(entry, dict):
                continue
            query = entry.get("query")
            if not isinstance(query, dict):
                continue
            lang = query.get("language")
            stmt = query.get("statement")
            if (
                isinstance(lang, str)
                and lang.lower() == "sql"
                and isinstance(stmt, str)
                and stmt.strip()
            ):
                statements.append(
                    {
                        "node_id": node.get("id", ""),
                        "evidence_index": idx,
                        "statement": stmt,
                    }
                )

    if not statements:
        return {"total": 0, "executable": 0, "failed": 0, "ratio": 0.0}

    try:
        conn = duckdb.connect(":memory:")
        for pq in case_dir.glob("*.parquet"):
            if pq.name == "conclusion.parquet":
                continue
            view = pq.stem
            conn.execute(
                f"CREATE VIEW IF NOT EXISTS \"{view}\" AS "
                f"SELECT * FROM read_parquet('{pq}')"
            )
    except Exception as exc:
        return {
            "total": len(statements),
            "executable": 0,
            "failed": len(statements),
            "ratio": 0.0,
            "setup_error": str(exc),
        }

    executable = 0
    for item in statements:
        sql = str(item["statement"]).strip().rstrip(";").strip()
        if not sql or ";" in sql:
            continue
        try:
            conn.execute(f"SELECT * FROM ({sql}) LIMIT 1").fetchone()
            executable += 1
        except Exception:
            pass
    conn.close()

    total = len(statements)
    return {
        "total": total,
        "executable": executable,
        "failed": total - executable,
        "ratio": executable / total if total else 0.0,
    }


def rejudge(
    db_path: Path,
    exp_id: str,
    case_root: Path,
    dry_run: bool = False,
) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, source, response, eval_metrics FROM evaluation_data "
        "WHERE exp_id = ? AND stage IN ('judged', 'rollout') "
        "AND response IS NOT NULL AND response != ''",
        (exp_id,),
    ).fetchall()

    print(f"Found {len(rows)} samples for exp_id={exp_id}")

    scored = 0
    graph_scores: list[float] = []
    service_hits = 0
    sql_ratios: list[float] = []
    errors: list[str] = []

    for row in rows:
        row_id = row["id"]
        source = row["source"]
        response_raw = row["response"]

        try:
            response_json = json.loads(response_raw)
        except (json.JSONDecodeError, TypeError):
            errors.append(f"{source}: response not JSON")
            continue

        if not isinstance(response_json, dict):
            errors.append(f"{source}: response not a dict")
            continue

        is_fpg = isinstance(response_json.get("nodes"), list) and isinstance(
            response_json.get("root_causes"), list
        )
        if not is_fpg:
            errors.append(f"{source}: not fpg format")
            continue

        case_dir = case_root / source
        if not case_dir.is_dir():
            errors.append(f"{source}: case dir not found at {case_dir}")
            continue

        comparison = _fpg_graph_score(response_json, case_dir)
        if comparison is None:
            errors.append(f"{source}: graph comparison failed")
            continue

        graph_score = float(comparison.get("score", 0.0))
        graph_scores.append(graph_score)

        services, fault_kinds = _extract_services_and_fault_kinds(response_json)
        gt_services = _gt_services_from_injection(case_dir)
        got_lower = {s.lower() for s in services}
        svc_hit = any(s.lower() in got_lower for s in gt_services)
        if svc_hit:
            service_hits += 1

        sql_eval = _sql_evidence_replay(response_json, case_dir)

        eval_metrics = {
            "fpg_graph_score": graph_score,
            "fpg_comparison": comparison,
            "extracted_services": services,
            "extracted_fault_kinds": fault_kinds,
            "gt_services": gt_services,
            "service_hit": svc_hit,
            "sql_evidence": sql_eval,
        }

        correct = graph_score >= 0.5
        confidence = graph_score

        if not dry_run:
            conn.execute(
                "UPDATE evaluation_data SET "
                "eval_metrics = ?, correct = ?, confidence = ?, stage = 'judged' "
                "WHERE id = ?",
                (json.dumps(eval_metrics, ensure_ascii=False), correct, confidence, row_id),
            )

        scored += 1

    if not dry_run:
        conn.commit()
    conn.close()

    print(f"\nRejudged: {scored}/{len(rows)}")
    if graph_scores:
        avg_score = sum(graph_scores) / len(graph_scores)
        nonzero = sum(1 for s in graph_scores if s > 0)
        perfect = sum(1 for s in graph_scores if s >= 0.999)
        print(f"Avg fpg graph score: {avg_score:.4f}")
        print(f"Nonzero: {nonzero}/{len(graph_scores)}")
        print(f"Perfect (>=0.999): {perfect}/{len(graph_scores)}")
        print(f"Service hit: {service_hits}/{len(graph_scores)}")
    if sql_ratios:
        print(f"Avg SQL executable rate: {sum(sql_ratios)/len(sql_ratios):.4f}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    if dry_run:
        print("\n(dry-run, no DB changes written)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-judge fpg eval results")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--db", default="eval.db", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rejudge(args.db, args.exp_id, args.case_root, args.dry_run)


if __name__ == "__main__":
    main()
