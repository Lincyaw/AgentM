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
                if not t.startswith("{") or '"injection_effective"' not in t or '"propagation_edges"' not in t:
                    continue
                if "You verify" in t:  # system-prompt echo
                    continue
                try:
                    obj = json.loads(t)
                except Exception:
                    continue
                if isinstance(obj, dict) and "propagation_edges" in obj:
                    best = obj
    return best


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

    prompt = (
        "A candidate service-level fault-propagation graph derived from this "
        "case's existing labels is below. Call get_injection_spec and "
        "get_fault_kind_doc to learn the mechanism, then verify each "
        "candidate edge against the parquets (drop the unsupported, 查准) and "
        "add any missing data-supported edges (查漏). Submit the corrected "
        "graph via submit_propagation_report.\n\n"
        "## Candidate graph\n```json\n"
        + json.dumps(cand_json, ensure_ascii=False, indent=2)
        + "\n```"
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

    verified = {(e["from_service"], e["to_service"]) for e in report["propagation_edges"]}
    cand_set = set(cand)
    dropped = sorted(cand_set - verified)   # suspected label false-positives
    added = sorted(verified - cand_set)     # suspected label false-negatives
    kept = sorted(cand_set & verified)
    findings = {
        "injection_effective": report.get("injection_effective"),
        "candidate_edges": len(cand_set),
        "kept": kept,
        "dropped_suspected_label_FP": dropped,
        "added_suspected_label_FN": added,
    }
    (out / "label_findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2))
    print(f"candidate edges: {len(cand_set)}  kept: {len(kept)}")
    print(f"  suspected label FP (dropped): {dropped}")
    print(f"  suspected label FN (added)  : {added}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
