#!/usr/bin/env python3
"""Producer-consumer fault propagation verifier (fpg-native).

Phase 0: build a relationship graph from traces (call graph,
bidirectional) and optionally deployment co-location.
Propagation: every confirmed node is a *producer* — it enqueues ALL
its neighbours. Each neighbour is a *consumer*: a hop-agent checks
whether it is genuinely degraded. If confirmed, it becomes a
producer itself. EVERY edge goes through a hop agent — including
edges between already-confirmed services. Same-round hops run in
parallel.

The primary output per case is ``fpg_scenario.json`` — a ground-truth
scenario in the fault-propagation-graph schema
(https://github.com/Lincyaw/fpg-convention), validated against the
profile-bound schema (fpg_profile.toml) before writing. Pipeline
telemetry (hop log, all verdicts, judge review) goes to
``run_meta.json``.

Orchestration is a pre-written workflow script
(``propagation_workflow.py``) executed by the ``workflow`` atom engine.
The harness prepares the args (DuckDB graph, injections, prebuilt fpg
seed nodes) and invokes the workflow tool directly on an AgentSession.

Usage (single case):
    uv run python cli.py run <case_dir> [--model X]

Usage (batch — ablation runs):
    uv run python cli.py batch <dataset_dir> \\
        --run-dir /tmp/verifier-seed2pro --model litellm --limit 10
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any, TypedDict

import typer

# eval/ modules are not a package; add to sys.path so bare imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))

from diff import diff_cases  # noqa: E402
from fpg_mapping import (  # noqa: E402
    REL_MECHANISM,
    assemble_scenario,
    build_seed_node,
    load_injection_meta,
)
from graph import (  # noqa: E402
    SYNTHETIC,
    _build_neighbor_graph,
    get_infra_edges,
    get_infra_nodes,
    get_node_map,
    get_relationships,
    profile_dataset,
    vanished_endpoints,
)
from injection import (  # noqa: E402
    TargetEvidence,
    _load_fault_doc,
    get_injections,
    get_target_evidence,
)

REPO = Path(__file__).resolve().parents[3]
WORKFLOW_SCRIPT = Path(__file__).resolve().parent / "eval" / "propagation_workflow.py"

# ------------------------------------------------------------------
# Provider helpers
# ------------------------------------------------------------------

ProviderSpec = tuple[str, dict[str, Any]]
ExtensionSpec = tuple[str, dict[str, Any]]

class WorkflowArgs(TypedDict, total=False):
    data_dir: str
    graph: dict[str, list[list[str]]]
    injections: list[dict[str, str]]
    infra_nodes: list[str]
    node_map: dict[str, str]
    fault_docs: dict[str, str]
    budget: int
    out_dir: str
    skip_propagate: bool
    skip_judge: bool
    window: dict[str, str]
    dataset_profile: dict[str, Any]
    vanished: dict[str, Any]
    entry_services: list[str]
    seed_nodes: dict[str, dict[str, Any]]
    rel_mechanism: dict[str, str]
    existing_state: dict[str, Any]

def _resolve_provider() -> ProviderSpec:
    """Build a provider spec from the environment (config.toml profile)."""
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib import resolve_model_profile

    model_name = os.environ.get("AGENTM_MODEL")
    profile = resolve_model_profile(model_name)
    if profile is not None:
        build_config = profile.to_build_config()
        provider_id = os.environ.get("AGENTM_PROVIDER") or profile.provider
    else:
        registry = DEFAULT_PROVIDER_REGISTRY
        provider_id = os.environ.get("AGENTM_PROVIDER") or registry.default_provider().id
        build_config = {"model": model_name or registry.default_model(provider_id)}

    return DEFAULT_PROVIDER_REGISTRY.build(provider_id, build_config)

# ------------------------------------------------------------------
# Workflow invocation
# ------------------------------------------------------------------

_WORKFLOW_EXTENSIONS: list[ExtensionSpec] = [
    ("agentm.extensions.builtin.operations", {"backend": "local"}),
    ("agentm.extensions.builtin.observability", {}),
    ("agentm.extensions.builtin.artifact_store", {}),
    ("agentm.extensions.builtin.workflow", {}),
]

async def _run_workflow_async(
    workflow_args: WorkflowArgs,
    out_dir: Path,
) -> dict[str, Any]:
    """Run a workflow script via the WorkflowRunner service."""
    from agentm.core.abi import AgentSessionConfig
    from agentm.core.runtime import AgentSession

    os.environ["AGENTM_PROJECT_ROOT"] = str(REPO)

    provider_spec = _resolve_provider()
    config = AgentSessionConfig(
        cwd=str(out_dir),
        provider=provider_spec,
        extensions=[(m, dict(c)) for m, c in _WORKFLOW_EXTENSIONS],
        auto_commit=False,
    )
    session = await AgentSession.create(config)
    try:
        runner = session.get_service("workflow_runner")
        if runner is None:
            raise RuntimeError("workflow_runner service not found")
        result = await runner.run_file(WORKFLOW_SCRIPT, workflow_args)
        return result if isinstance(result, dict) else {}
    finally:
        await session.shutdown()

# ------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------

def run_judge(
    case_dir: Path,
    run_dir: Path,
    *,
    budget: int = 20,
) -> dict:
    """Run judge via the workflow (skip_propagate=True, skip_judge=False).

    Reads the case's fpg_scenario.json + run_meta.json, applies the
    judge's cascade promotions (each attached through a confirmed
    upstream), and rewrites both files.
    """
    data_dir = case_dir.resolve()
    out = run_dir.resolve()
    scenario_path = out / "fpg_scenario.json"
    meta_path = out / "run_meta.json"
    if not scenario_path.exists():
        return {}

    scenario = json.loads(scenario_path.read_text())
    run_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    injections = get_injections(data_dir)
    if not injections:
        return {}

    fault_docs: dict[str, str] = {}
    for inj in injections:
        fk = inj["chaos_type"]
        if fk not in fault_docs:
            doc = _load_fault_doc(fk)
            if doc:
                fault_docs[fk] = doc

    meta = load_injection_meta(data_dir)
    rels = get_relationships(data_dir)
    infra_nodes = get_infra_nodes(data_dir)
    rels.extend(get_infra_edges(data_dir, infra_nodes))
    neighbor_graph = _build_neighbor_graph(rels)
    graph_serializable: dict[str, list[list[str]]] = {
        svc: [[n, r] for n, r in neighbors]
        for svc, neighbors in neighbor_graph.items()
        if svc not in SYNTHETIC
    }

    workflow_args: WorkflowArgs = {
        "data_dir": str(data_dir),
        "graph": graph_serializable,
        "injections": [
            i for i in injections
            if i.get("target") and i["target"] not in SYNTHETIC
        ],
        "infra_nodes": sorted(infra_nodes),
        "node_map": {},
        "fault_docs": fault_docs,
        "budget": budget,
        "out_dir": str(out),
        "skip_propagate": True,
        "skip_judge": False,
        "window": meta["window"],
        "rel_mechanism": dict(REL_MECHANISM),
        "dataset_profile": dict(profile_dataset(data_dir)),
        "entry_services": sorted(
            {a for a, b, r in rels if r == "caller_to_callee"}
            - {b for a, b, r in rels if r == "caller_to_callee"}
        ),
        "vanished": vanished_endpoints(
            data_dir,
            sorted({
                k.split("__", 1)[1]
                for k, v in run_meta.get("verdicts", {}).items()
                if v.get("verdict") == "rejected"
            }),
        ),
        "existing_state": {
            "nodes": scenario.get("graph", {}).get("nodes", []),
            "edges": scenario.get("graph", {}).get("edges", []),
            "verdicts": run_meta.get("verdicts", {}),
            "hop_log": run_meta.get("hop_log", []),
            "rounds": run_meta.get("rounds", 0),
        },
    }

    result = asyncio.run(_run_workflow_async(workflow_args, out))
    if not result:
        return {}

    scenario = assemble_scenario(meta, result["nodes"], result["edges"])
    scenario_path.write_text(
        json.dumps(scenario, indent=2, ensure_ascii=False) + "\n"
    )
    run_meta.update({
        "hop_log": result.get("hop_log", []),
        "rounds": result.get("rounds", 0),
        "verdicts": result.get("verdicts", {}),
        "judge": result.get("judge") or {},
        "judge_rounds": result.get("judge_rounds", []),
    })
    meta_path.write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n"
    )
    return result.get("judge") or {}

def run_one_case(
    case_dir: Path,
    out_dir: Path,
    *,
    budget: int = 15,
) -> dict:
    """Run propagation verification on a single case.

    Writes the fpg ground-truth scenario (fpg_scenario.json, validated
    against the profile-bound schema) plus pipeline telemetry
    (run_meta.json) to *out_dir*. Returns a summary dict with keys:
    case, seeds, confirmed, edges, rounds, error.
    """
    data_dir = case_dir.resolve()
    out = out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    case_name = data_dir.name

    injections = get_injections(data_dir)
    if not injections:
        return {"case": case_name, "error": "no injections"}
    meta = load_injection_meta(data_dir)

    rels = get_relationships(data_dir)
    infra_nodes = get_infra_nodes(data_dir)
    rels.extend(get_infra_edges(data_dir, infra_nodes))
    neighbor_graph = _build_neighbor_graph(rels)
    node_map = get_node_map(data_dir)
    (out / "relationships.json").write_text(json.dumps(
        [{"a": a, "b": b, "rel": r} for a, b, r in rels],
        indent=2, ensure_ascii=False,
    ))

    total_edges = sum(len(v) for v in neighbor_graph.values())
    print(f"Injections: {[(i['target'], i['chaos_type']) for i in injections]}")
    for inj in injections:
        fk = inj["chaos_type"]
        loaded = "loaded" if _load_fault_doc(fk) else "not found"
        print(f"Fault doc: {loaded} ({fk})")
    print(f"Relationship graph: {total_edges} directed edges "
          f"({len(neighbor_graph)} services)")
    print(f"Infra nodes (metrics-only): {sorted(infra_nodes)}")

    # Serialize the neighbor graph for the workflow (tuples -> lists)
    graph_serializable: dict[str, list[list[str]]] = {
        svc: [[n, r] for n, r in neighbors]
        for svc, neighbors in neighbor_graph.items()
        if svc not in SYNTHETIC
    }

    # Prebuild fpg seed nodes: predicate mapped from the chaos type,
    # evidence = the normal/abnormal latency comparison
    seed_nodes: dict[str, dict[str, Any]] = {}
    for inj in injections:
        target = inj["target"]
        if target and target not in SYNTHETIC and target not in seed_nodes:
            evidence: TargetEvidence = get_target_evidence(data_dir, target)
            seed_nodes[target] = build_seed_node(
                target, inj["chaos_type"], meta["window"], dict(evidence),
            )

    # Pre-compute fault docs so the workflow can pass them to context atoms
    fault_docs: dict[str, str] = {}
    for inj in injections:
        fk = inj["chaos_type"]
        if fk not in fault_docs:
            doc = _load_fault_doc(fk)
            if doc:
                fault_docs[fk] = doc

    workflow_args: WorkflowArgs = {
        "data_dir": str(data_dir),
        "graph": graph_serializable,
        "injections": [
            i for i in injections
            if i.get("target") and i["target"] not in SYNTHETIC
        ],
        "infra_nodes": sorted(infra_nodes),
        "node_map": node_map,
        "fault_docs": fault_docs,
        "budget": budget,
        "out_dir": str(out),
        "skip_judge": False,
        "window": meta["window"],
        "seed_nodes": seed_nodes,
        "rel_mechanism": dict(REL_MECHANISM),
        "dataset_profile": dict(profile_dataset(data_dir)),
    }

    result = asyncio.run(_run_workflow_async(workflow_args, out))

    scenario = assemble_scenario(meta, result["nodes"], result["edges"])
    (out / "fpg_scenario.json").write_text(
        json.dumps(scenario, indent=2, ensure_ascii=False) + "\n"
    )
    run_meta: dict[str, Any] = {
        "hop_log": result.get("hop_log", []),
        "rounds": result.get("rounds", 0),
        "verdicts": result.get("verdicts", {}),
    }
    if result.get("judge_rounds"):
        run_meta["judge_rounds"] = result["judge_rounds"]
    (out / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False, default=str) + "\n"
    )

    seeds = [i["target"] for i in injections]
    confirmed = [n["id"] for n in result["nodes"]]
    propagated = [s for s in confirmed if s not in seeds]
    edge_pairs = [[e["src"], e["dst"]] for e in result["edges"]]

    print(f"\nRounds: {result['rounds']}")
    print(f"Confirmed: {confirmed}")
    print(f"Edges ({len(edge_pairs)}): {edge_pairs}")

    return {
        "case": case_name,
        "fault_kind": "+".join(
            dict.fromkeys(i["chaos_type"] for i in injections)
        ),
        "seeds": seeds,
        "confirmed": confirmed,
        "propagated": propagated,
        "edges": len(edge_pairs),
        "rounds": result["rounds"],
    }

# ------------------------------------------------------------------
# Batch runner
# ------------------------------------------------------------------

def _read_cached_summary(out_dir: Path, case_name: str) -> dict | None:
    scenario_path = out_dir / "fpg_scenario.json"
    meta_path = out_dir / "run_meta.json"
    if not scenario_path.exists() or not meta_path.exists():
        return None
    try:
        scenario = json.loads(scenario_path.read_text())
        run_meta = json.loads(meta_path.read_text())
    except Exception:  # noqa: BLE001
        return None
    seeds = [i["node_id"] for i in scenario.get("injections", [])]
    confirmed = [n["id"] for n in scenario.get("graph", {}).get("nodes", [])]
    return {
        "case": case_name,
        "fault_kind": "+".join(
            dict.fromkeys(i["fault_type"] for i in scenario.get("injections", []))
        ) or "unknown",
        "seeds": seeds,
        "confirmed": confirmed,
        "propagated": [s for s in confirmed if s not in seeds],
        "edges": len(scenario.get("graph", {}).get("edges", [])),
        "rounds": run_meta.get("rounds", 0),
        "cached": True,
    }

def _run_or_cache(
    dataset_dir: Path,
    run_dir: Path,
    name: str,
    idx: int,
    total: int,
    budget: int,
) -> dict:
    """Run one case or return cached result.  Thread-safe."""
    case_out = run_dir / name
    existing = _read_cached_summary(case_out, name)
    if existing:
        prop_str = (f"propagated={existing['propagated']}"
                    if existing["propagated"] else "no propagation")
        print(f"[{idx}/{total}] {name} CACHED: {prop_str}", flush=True)
        return existing

    print(f"[{idx}/{total}] {name} ...", flush=True)
    try:
        summary = run_one_case(
            dataset_dir / name, case_out,
            budget=budget,
        )
        if "error" in summary:
            print(f"  [{name}] ERROR: {summary['error']}", flush=True)
        else:
            prop_str = (f"propagated={summary['propagated']}"
                        if summary.get("propagated") else "no propagation")
            print(f"  [{name}] OK: {prop_str}", flush=True)
        return summary
    except Exception as exc:  # noqa: BLE001
        print(f"  [{name}] EXCEPTION: {exc}", flush=True)
        return {"case": name, "error": str(exc)}

def run_batch(
    dataset_dir: Path,
    run_dir: Path,
    *,
    budget: int = 15,
    case_parallel: int = 1,
    limit: int | None = None,
    offset: int = 0,
    case_filter: set[str] | None = None,
) -> list[dict]:
    """Run propagation on multiple cases under *dataset_dir*.

    *case_parallel* controls how many cases run concurrently.
    ``run_summary.jsonl`` in *run_dir* holds one JSON line per case,
    suitable for diff/analysis across ablation runs.
    """
    cases = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    if case_filter is not None:
        cases = [c for c in cases if c in case_filter]
    cases = cases[offset:]
    if limit is not None:
        cases = cases[:limit]

    total = len(cases)
    run_dir.mkdir(parents=True, exist_ok=True)

    if case_parallel <= 1:
        summaries = [
            _run_or_cache(dataset_dir, run_dir, name, i, total, budget)
            for i, name in enumerate(cases, 1)
        ]
    else:
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=case_parallel) as pool:
            futures = {
                pool.submit(
                    _run_or_cache,
                    dataset_dir, run_dir, name, i, total, budget,
                ): name
                for i, name in enumerate(cases, 1)
            }
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
        summaries = [results[name] for name in cases]

    ok = sum(1 for s in summaries if "error" not in s)
    fail = sum(1 for s in summaries if "error" in s)
    cached = sum(1 for s in summaries if s.get("cached"))

    summary_path = run_dir / "run_summary.jsonl"
    with open(summary_path, "w") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Total: {total}  OK: {ok}  Failed: {fail}  Cached: {cached}")
    prop_count = sum(1 for s in summaries if s.get("propagated"))
    print(f"Cases with propagation: {prop_count}/{ok}")
    print(f"Summary: {summary_path}")

    return summaries

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

app = typer.Typer(
    name="audit-propagate",
    help=__doc__,
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

def _set_model(model: str | None) -> None:
    if model:
        os.environ["AGENTM_MODEL"] = model

@app.command()
def run(
    case_dir: Annotated[Path, typer.Argument(help="single case directory")],
    out: Annotated[Path | None, typer.Option(help="output directory")] = None,
    budget: Annotated[int, typer.Option(help="tool-call budget per hop agent")] = 15,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
) -> None:
    """Run propagation check on a single case."""
    _set_model(model)
    out_dir = out or case_dir.resolve() / ".verify_propagate"
    summary = run_one_case(case_dir, out_dir, budget=budget)
    if "error" in summary:
        raise typer.Exit(1)

@app.command()
def batch(
    dataset_dir: Annotated[Path, typer.Argument(help="directory containing case subdirectories")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="output directory for this run")],
    budget: Annotated[int, typer.Option(help="tool-call budget per hop agent")] = 15,
    case_parallel: Annotated[int, typer.Option("--case-parallel", help="concurrent cases")] = 1,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
    limit: Annotated[int | None, typer.Option(help="max cases to run")] = None,
    offset: Annotated[int, typer.Option(help="skip first N cases")] = 0,
    cases_file: Annotated[Path | None, typer.Option("--cases-file", help="text file with one case name per line")] = None,
) -> None:
    """Run propagation on all cases in a dataset."""
    _set_model(model)
    case_filter = None
    if cases_file is not None:
        case_filter = set(cases_file.read_text().strip().splitlines())
    run_batch(
        dataset_dir.resolve(), run_dir.resolve(),
        budget=budget,
        case_parallel=case_parallel,
        limit=limit, offset=offset,
        case_filter=case_filter,
    )

@app.command()
def judge(
    case_dir: Annotated[Path, typer.Argument(help="single case directory")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="verifier run output for this case")],
    budget: Annotated[int, typer.Option(help="tool-call budget for judge agent")] = 20,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
) -> None:
    """Run judge agent on a completed case to review hop verdicts."""
    _set_model(model)
    result = run_judge(case_dir, run_dir, budget=budget)
    if result:
        print(f"\nJudge rationale: {result.get('rationale', '?')}")
        added = result.get("add", [])
        if added:
            for promo in added:
                print(f"  promoted: {promo.get('via_service')} -> "
                      f"{promo.get('service')} ({promo.get('predicate')})")
    else:
        print("Judge produced no review.")

    scenario_path = run_dir / "fpg_scenario.json"
    if scenario_path.exists():
        scenario = json.loads(scenario_path.read_text())
        n_nodes = len(scenario.get("graph", {}).get("nodes", []))
        n_edges = len(scenario.get("graph", {}).get("edges", []))
        print(f"\nScenario after judge: {n_nodes} nodes, {n_edges} edges")
        print(f"Output: {scenario_path}")

def _run_judge_or_skip(
    dataset_dir: Path,
    run_dir: Path,
    name: str,
    idx: int,
    total: int,
    budget: int,
) -> dict:
    case_out = run_dir / name
    meta_path = case_out / "run_meta.json"
    if meta_path.exists():
        try:
            if "judge" in json.loads(meta_path.read_text()):
                print(f"[{idx}/{total}] {name} CACHED", flush=True)
                return {"case": name, "cached": True}
        except Exception:  # noqa: BLE001
            pass
    if not (case_out / "fpg_scenario.json").exists():
        print(f"[{idx}/{total}] {name} SKIP: no hop results", flush=True)
        return {"case": name, "error": "no hop results"}
    print(f"[{idx}/{total}] {name} judging...", flush=True)
    try:
        run_judge(dataset_dir / name, case_out, budget=budget)
        print(f"  [{name}] OK", flush=True)
        return {"case": name}
    except Exception as exc:  # noqa: BLE001
        print(f"  [{name}] EXCEPTION: {exc}", flush=True)
        return {"case": name, "error": str(exc)}

@app.command(name="judge-batch")
def judge_batch(
    dataset_dir: Annotated[Path, typer.Argument(help="dataset directory")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="verifier run output")],
    budget: Annotated[int, typer.Option(help="tool-call budget per judge")] = 20,
    parallel: Annotated[int, typer.Option(help="concurrent judge agents")] = 10,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
    limit: Annotated[int | None, typer.Option(help="max cases")] = None,
) -> None:
    """Run judge on all completed cases in a batch run."""
    _set_model(model)
    cases = sorted(
        p.name for p in run_dir.iterdir()
        if p.is_dir() and (p / "fpg_scenario.json").exists()
    )
    if limit:
        cases = cases[:limit]
    total = len(cases)
    print(f"Judging {total} cases (parallel={parallel})")

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                _run_judge_or_skip,
                dataset_dir.resolve(), run_dir.resolve(),
                name, i, total, budget,
            ): name
            for i, name in enumerate(cases, 1)
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()

    summaries = [results[name] for name in cases if name in results]
    ok = sum(1 for s in summaries if "error" not in s)
    cached = sum(1 for s in summaries if s.get("cached"))

    print(f"\n{'='*50}")
    print(f"Total: {total}  OK: {ok}  Cached: {cached}")

@app.command()
def diff(
    dataset_dir: Annotated[Path, typer.Argument(help="dataset with GT labels")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="verifier run output")],
    format: Annotated[str, typer.Option(help="table or ndjson")] = "table",
) -> None:
    """Compare verifier graph against GT per case.

    For each case, classifies every service into one of:
      agree      — both verifier and GT have it
      v_only     — verifier confirmed, GT lacks
      rejected   — GT has it, hop agent evaluated and rejected
      unreachable — GT has it, verifier never evaluated (not in BFS neighborhood)

    Outputs per-case JSONL to <run-dir>/gt_diff.jsonl and prints a summary.
    """
    rows, agg, svc_agg = diff_cases(dataset_dir, run_dir)

    diff_path = run_dir / "gt_diff.jsonl"
    with open(diff_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if format == "ndjson":
        for row in rows:
            typer.echo(json.dumps(row, ensure_ascii=False))
        return

    has_diff = [r for r in rows
                if r["v_only"] or r["rejected"] or r["unreachable"]]
    for r in has_diff:
        parts = []
        if r["rejected"]:
            parts.append(f"rejected={r['rejected']}")
        if r["unreachable"]:
            parts.append(f"unreachable={r['unreachable']}")
        if r["v_only"]:
            parts.append(f"v_only={r['v_only']}")
        typer.echo(f"{r['case']}  {', '.join(parts)}")

    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Cases: {agg.get('cases', 0)}  Exact match: {agg.get('exact_match', 0)}")
    typer.echo(f"Services — agree: {agg.get('agree', 0)}  v_only: {agg.get('v_only', 0)}  "
               f"rejected: {agg.get('rejected', 0)}  unreachable: {agg.get('unreachable', 0)}")

    top_rejected = sorted(
        ((s, c["rejected"]) for s, c in svc_agg.items() if c.get("rejected")),
        key=lambda x: -x[1],
    )
    top_unreachable = sorted(
        ((s, c["unreachable"]) for s, c in svc_agg.items() if c.get("unreachable")),
        key=lambda x: -x[1],
    )
    if top_rejected:
        typer.echo("\nTop rejected (GT has, hop agent rejected):")
        for s, n in top_rejected[:15]:
            typer.echo(f"  {s}: {n}")
    if top_unreachable:
        typer.echo("\nTop unreachable (GT has, BFS never reached):")
        for s, n in top_unreachable[:15]:
            typer.echo(f"  {s}: {n}")

    typer.echo(f"\nPer-case diff: {diff_path}")

if __name__ == "__main__":
    app()
