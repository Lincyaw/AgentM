"""GT comparison: compare verifier graph against ground truth labels."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _gt_services(case_dir: Path) -> tuple[set[str], set[str]]:
    """Extract GT injection seeds and propagated services.

    Seeds come from injection.json ``engine_config``.  Propagated
    services come from causal_graph.json: fold the span-level graph to
    service granularity via ``component_to_service``, then subtract
    the seeds.
    """
    inj_path = case_dir / "injection.json"
    cg_path = case_dir / "causal_graph.json"
    if not inj_path.exists():
        return set(), set()

    inj = json.loads(inj_path.read_text())
    seeds: set[str] = set()
    raw_ec = inj.get("engine_config", [])
    if isinstance(raw_ec, list):
        for item in raw_ec:
            if isinstance(item, dict) and item.get("app"):
                seeds.add(item["app"])
    if not seeds:
        for item in inj.get("engine_config_summary", []):
            if isinstance(item, dict) and item.get("app"):
                seeds.add(item["app"])
    if not seeds:
        gt = inj.get("ground_truth")
        if isinstance(gt, dict):
            for s in gt.get("service", []):
                seeds.add(s)
        elif isinstance(gt, list):
            for g in gt:
                if isinstance(g, dict):
                    for s in g.get("service", []):
                        seeds.add(s)

    if not cg_path.exists():
        return seeds, set()

    cg = json.loads(cg_path.read_text())
    c2s = cg.get("component_to_service", {})
    gt_all = {svc for svc in c2s.values() if svc}
    return seeds, gt_all - seeds


def diff_cases(
    dataset_dir: Path,
    run_dir: Path,
) -> tuple[list[dict], dict[str, int], dict[str, dict[str, int]]]:
    """Compare verifier results against GT for all completed cases.

    Returns (rows, agg, svc_agg) where:
    - rows: per-case dicts with agree/v_only/rejected/unreachable
    - agg: aggregate counts
    - svc_agg: per-service category counts
    """
    cases = sorted(
        p.name for p in run_dir.iterdir()
        if p.is_dir() and (p / "final_propagation.json").exists()
    )

    agg: dict[str, int] = defaultdict(int)
    svc_agg: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows: list[dict] = []

    for name in cases:
        cdir = dataset_dir / name
        if not cdir.exists():
            continue
        fp = json.loads((run_dir / name / "final_propagation.json").read_text())
        seeds = set(fp.get("seeds", []))
        v_prop = set(fp.get("propagated", []))

        gt_seeds, gt_prop = _gt_services(cdir)
        seeds |= gt_seeds

        evaluated: set[str] = set()
        avp = run_dir / name / "all_verdicts.json"
        if avp.exists():
            evaluated = {v["to"] for v in json.loads(avp.read_text())}

        agree = v_prop & gt_prop
        v_only = v_prop - gt_prop
        gt_only = gt_prop - v_prop
        rejected = gt_only & evaluated
        unreachable = gt_only - evaluated

        row: dict = {
            "case": name,
            "seeds": sorted(seeds),
            "agree": sorted(agree),
            "v_only": sorted(v_only),
            "rejected": sorted(rejected),
            "unreachable": sorted(unreachable),
        }
        rows.append(row)

        agg["cases"] += 1
        agg["agree"] += len(agree)
        agg["v_only"] += len(v_only)
        agg["rejected"] += len(rejected)
        agg["unreachable"] += len(unreachable)
        if not v_only and not rejected and not unreachable:
            agg["exact_match"] += 1

        for s in rejected:
            svc_agg[s]["rejected"] += 1
        for s in unreachable:
            svc_agg[s]["unreachable"] += 1
        for s in v_only:
            svc_agg[s]["v_only"] += 1

    return rows, dict(agg), dict(svc_agg)
