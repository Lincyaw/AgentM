#!/usr/bin/env python3
"""Driver: audit one case's service-level causal-graph labels with the
``auditor`` scenario.

Projects ``causal_graph.json`` to service-level edges (the candidate graph),
assembles the injection spec + per-fault mechanism docs + candidate graph
into the auditor prompt, runs the auditor, and writes the audit report.

The auditor's verdict is the dataset-label finding for this case:
  - candidate edges marked ``unsupported`` are suspected label false-positives;
  - ``added_edges`` are suspected label false-negatives.

Run under uv (needs agentm + agentm_rca importable):

    uv run --no-sync python contrib/scenarios/auditor/eval/audit_case.py \
        <case_dir> [--out <dir>] [--budget N]

``<case_dir>`` is a directory holding ``injection.json`` / ``env.json`` /
``causal_graph.json`` / ``*_traces.parquet`` (an opslite case). Output lands
under ``--out`` (default ``<case_dir>/.audit``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VERIFIER_DIR = REPO / "contrib/scenarios/verifier"
FAULT_DOCS = VERIFIER_DIR / "fault_kinds"
SYNTHETIC = {"loadgenerator", "locust", "wrk2", "dsb-wrk2", "k6", "load-generator", "load_generator"}


def _load(modpath: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    elif isinstance(obj, str):
        yield obj


def extract_audit(obs_dir: Path):
    best = None
    for trace in sorted(obs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        for line in trace.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            for s in _walk(row):
                t = s.strip()
                if not t.startswith("{") or '"edge_audits"' not in t or '"added_edges"' not in t:
                    continue
                if "You audit" in t:  # system-prompt echo
                    continue
                try:
                    obj = json.loads(t)
                except Exception:
                    continue
                if isinstance(obj, dict) and "edge_audits" in obj:
                    best = obj
    return best


def candidate_edges(causal_graph: dict) -> list[dict]:
    comp2svc = causal_graph.get("component_to_service") or {}

    def to_svc(comp: str):
        if comp in comp2svc:
            return comp2svc[comp]
        if "|" in comp:
            rest = comp.split("|", 1)[1]
            return rest.split("::", 1)[0] if "::" in rest else rest
        return comp

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for e in causal_graph.get("edges", []):
        s, t = to_svc(e["source"]), to_svc(e["target"])
        if not s or not t or s == t or s in SYNTHETIC or t in SYNTHETIC:
            continue
        if (s, t) not in seen:
            seen.add((s, t))
            out.append({"from_service": s, "to_service": t})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    data_dir = args.case_dir.resolve()
    out = (args.out or data_dir / ".audit").resolve()
    out.mkdir(parents=True, exist_ok=True)

    spec_mod = _load(VERIFIER_DIR / "verifier_fault_context.py", "vfc_audit")
    inj_spec = spec_mod._build_spec(data_dir)
    cand = candidate_edges(json.loads((data_dir / "causal_graph.json").read_text()))

    docs = []
    for kind in inj_spec.get("fault_kinds", []):
        doc = FAULT_DOCS / f"{kind}.md"
        if doc.exists():
            docs.append(f"### {kind}\n{doc.read_text()}")

    prompt = (
        "Audit the service-level fault-propagation labels of this case.\n\n"
        "## Known injection spec\n```json\n"
        + json.dumps(inj_spec, ensure_ascii=False, indent=2)
        + "\n```\n\n## Per-fault mechanism reference\n"
        + ("\n\n".join(docs) if docs else "(none)")
        + "\n\n## Candidate service-level propagation graph (from existing labels)\n```json\n"
        + json.dumps(cand, ensure_ascii=False, indent=2)
        + "\n```\n\nVerify every candidate edge (查准) and add any missing "
        "data-supported edges (查漏), then call submit_audit."
    )

    env = dict(os.environ)
    env["AGENTM_PROJECT_ROOT"] = str(REPO)
    env["AGENTM_RCA_DATA_DIR"] = str(data_dir)
    cmd = [
        "uv", "run", "--no-sync", "agentm", "--scenario", "auditor",
        "--provider", env.get("AGENTM_PROVIDER", "openai"),
        "--model", env.get("AGENTM_MODEL", "Doubao-Seed-2.0-pro"),
        "--cwd", str(out), "--quiet", "--max-tool-calls", str(args.budget), prompt,
    ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
    (out / "stderr.log").write_text(r.stderr)
    (out / "candidate.json").write_text(json.dumps(cand, ensure_ascii=False, indent=2))

    audit = extract_audit(out / ".agentm/observability")
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) if audit else "null")
    if not audit:
        print("FAIL: no audit produced (see stderr.log)")
        return 1

    unsupported = [(e["from_service"], e["to_service"]) for e in audit["edge_audits"] if e["status"] == "unsupported"]
    added = [(e["from_service"], e["to_service"]) for e in audit["added_edges"]]
    supported = sum(1 for e in audit["edge_audits"] if e["status"] == "supported")
    print(f"candidate edges: {len(cand)}  supported: {supported}")
    print(f"  suspected label FP (unsupported): {unsupported}")
    print(f"  suspected label FN (added)      : {added}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
