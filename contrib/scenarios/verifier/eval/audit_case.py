#!/usr/bin/env python3
"""Driver: verify/clean one case's service-level causal-graph labels with the
``verifier`` scenario.

Projects the case's ``causal_graph.json`` to service-level edges (the
candidate graph), hands that candidate to the verifier (which fetches the
injection spec and per-fault mechanism docs through its own tools), and
diffs the verifier's corrected ``propagation_edges`` against the candidate:

  - candidate edges the verifier DROPPED  → suspected label false-positives;
  - edges the verifier ADDED              → suspected label false-negatives.

Run under uv (needs agentm + agentm_rca importable):

    uv run --no-sync python contrib/scenarios/verifier/eval/audit_case.py \
        <case_dir> [--out <dir>] [--budget N]

``<case_dir>`` is an opslite case directory (injection.json / env.json /
causal_graph.json / *_traces.parquet). Output lands under ``--out``
(default ``<case_dir>/.verify``).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SYNTHETIC = {"loadgenerator", "locust", "wrk2", "dsb-wrk2", "k6", "load-generator", "load_generator"}


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    elif isinstance(obj, str):
        yield obj


def extract_report(obs_dir: Path):
    best = None
    for trace in sorted(obs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        for line in trace.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            for s in _walk(row):
                t = s.strip()
                if not t.startswith("{") or '"injections"' not in t or '"propagation_edges"' not in t:
                    continue
                if "You verify" in t:  # system-prompt echo
                    continue
                try:
                    obj = json.loads(t)
                except Exception:
                    continue
                if isinstance(obj, dict) and "propagation_edges" in obj and "injections" in obj:
                    best = obj
    return best


def _duck_conn(data_dir: Path):
    """Replicate the duckdb_sql tool's view setup: one view per parquet
    (filename stem), excluding conclusion.parquet (ground truth)."""
    import duckdb

    conn = duckdb.connect(":memory:")
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix == ".parquet" and f.name != "conclusion.parquet":
            path = f.as_posix().replace("'", "''")
            conn.execute(f"CREATE OR REPLACE VIEW {f.stem} AS SELECT * FROM read_parquet('{path}')")
    return conn


def _run_sql(conn, sql: str) -> dict:
    """Re-execute one evidence SQL. Status: ok (>=1 row) / empty (0 rows) / error."""
    if not isinstance(sql, str) or not sql.strip():
        return {"status": "error", "detail": "empty sql", "rows": 0}
    try:
        rows = conn.execute(sql).fetchall()
    except Exception as exc:  # noqa: BLE001 — surface the DB error verbatim
        return {"status": "error", "detail": str(exc).splitlines()[0][:200], "rows": 0}
    return {"status": "ok" if rows else "empty", "rows": len(rows)}


def verify_report_sql(data_dir: Path, report: dict) -> dict:
    """Re-run every node symptom_sql and edge relationship_sql against the case
    parquets. The checker only confirms each SQL is queryable (runs + returns
    rows); semantic judgement (does the evidence prove propagation) is left to
    the agent / an optional critic. An edge is 'verified' iff its
    relationship_sql is ok AND both endpoints' node symptom_sql are ok."""
    conn = _duck_conn(data_dir)
    nodes = {}
    for n in report.get("propagation_nodes", []) or []:
        evs = [_run_sql(conn, ev.get("sql", "")) for ev in (n.get("symptom_evidence") or [])]
        ok = any(e["status"] == "ok" for e in evs)
        status = "ok" if ok else ("error" if not evs else "empty")
        nodes[n.get("service")] = {"status": status, "evidence": evs}
    edges = []
    for e in report.get("propagation_edges", []) or []:
        f, t = e.get("from_service"), e.get("to_service")
        rel = _run_sql(conn, e.get("relationship_sql", ""))
        verified = (
            rel["status"] == "ok"
            and nodes.get(f, {}).get("status") == "ok"
            and nodes.get(t, {}).get("status") == "ok"
        )
        edges.append({"from": f, "to": t, "relationship": rel, "verified": verified})
    conn.close()
    return {
        "nodes": nodes,
        "edges": edges,
        "node_sql_ok": sum(v["status"] == "ok" for v in nodes.values()),
        "node_sql_total": len(nodes),
        "edge_verified": sum(e["verified"] for e in edges),
        "edge_total": len(edges),
    }


def candidate_edges(causal_graph: dict) -> list[tuple[str, str]]:
    comp2svc = causal_graph.get("component_to_service") or {}

    def to_svc(comp: str):
        if comp in comp2svc:
            return comp2svc[comp]
        if "|" in comp:
            rest = comp.split("|", 1)[1]
            return rest.split("::", 1)[0] if "::" in rest else rest
        return comp

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for e in causal_graph.get("edges", []):
        s, t = to_svc(e["source"]), to_svc(e["target"])
        if not s or not t or s == t or s in SYNTHETIC or t in SYNTHETIC:
            continue
        if (s, t) not in seen:
            seen.add((s, t))
            out.append((s, t))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    data_dir = args.case_dir.resolve()
    out = (args.out or data_dir / ".verify").resolve()
    out.mkdir(parents=True, exist_ok=True)

    cand = candidate_edges(json.loads((data_dir / "causal_graph.json").read_text()))
    cand_json = [{"from_service": a, "to_service": b} for a, b in cand]

    # Hand the agent only the task and its input (the candidate graph). It
    # discovers what was injected (get_injection_spec) and the fault mechanism
    # (get_fault_kind_doc) itself, and checks the data with query_sql — no
    # pre-injection of spec/docs (that hand-holding biased the reasoning).
    prompt = (
        "Verify this case's service-level fault-propagation labels against the "
        "data. A candidate graph derived from the existing labels is below.\n\n"
        "## Candidate graph (from existing labels)\n```json\n"
        + json.dumps(cand_json, ensure_ascii=False, indent=2)
        + "\n```\n\nVerify EVERY candidate edge against the parquets one by one "
        "(查准: keep supported, drop unsupported) and add any missing "
        "data-supported edges (查漏), then submit the corrected graph via "
        "submit_propagation_report."
    )

    env = dict(os.environ)
    env["AGENTM_PROJECT_ROOT"] = str(REPO)
    env["AGENTM_RCA_DATA_DIR"] = str(data_dir)
    # Prefer the console script directly (we run inside the venv) over a
    # nested ``uv run`` — concurrent nested ``uv run`` invocations can race on
    # the environment and spuriously fail to import workspace packages.
    base = ["agentm"] if shutil.which("agentm") else ["uv", "run", "--no-sync", "agentm"]
    cmd = [
        *base, "--scenario", "verifier",
        "--provider", env.get("AGENTM_PROVIDER", "openai"),
        "--model", env.get("AGENTM_MODEL", "Doubao-Seed-2.0-pro"),
        "--cwd", str(out), "--quiet", "--max-tool-calls", str(args.budget), prompt,
    ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
    (out / "stderr.log").write_text(r.stderr)
    (out / "candidate.json").write_text(json.dumps(cand_json, ensure_ascii=False, indent=2))

    report = extract_report(out / ".agentm/observability")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) if report else "null")
    if not report:
        print("FAIL: verifier produced no report (see stderr.log)")
        return 1

    # Verification layer: re-run every node/edge SQL, keep only SQL-verified
    # edges as the verifier's graph. A submitted edge whose evidence does not
    # re-execute is not counted.
    verif = verify_report_sql(data_dir, report)
    (out / "verification.json").write_text(json.dumps(verif, ensure_ascii=False, indent=2))
    verified = {(e["from"], e["to"]) for e in verif["edges"] if e["verified"]}
    unverified = sorted((e["from"], e["to"]) for e in verif["edges"] if not e["verified"])

    cand_set = set(cand)
    dropped = sorted(cand_set - verified)   # suspected label false-positives
    added = sorted(verified - cand_set)     # suspected label false-negatives
    kept = sorted(cand_set & verified)
    findings = {
        "injections": [
            {"target_service": i.get("target_service"), "fault_kind": i.get("fault_kind"),
             "verdict": i.get("verdict")}
            for i in report.get("injections", []) or []
        ],
        "candidate_edges": len(cand_set),
        "kept": kept,
        "dropped_suspected_label_FP": dropped,
        "added_suspected_label_FN": added,
        "sql_unverified_edges": unverified,
        "sql_check": {
            "node_sql_ok": verif["node_sql_ok"], "node_sql_total": verif["node_sql_total"],
            "edge_verified": verif["edge_verified"], "edge_total": verif["edge_total"],
        },
    }
    (out / "label_findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2))
    print(f"candidate edges: {len(cand_set)}  kept: {len(kept)}")
    print(f"  SQL check: nodes {verif['node_sql_ok']}/{verif['node_sql_total']} ok, "
          f"edges {verif['edge_verified']}/{verif['edge_total']} verified")
    print(f"  suspected label FP (dropped): {dropped}")
    print(f"  suspected label FN (added)  : {added}")
    if unverified:
        print(f"  edges dropped (SQL failed to verify): {unverified}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
