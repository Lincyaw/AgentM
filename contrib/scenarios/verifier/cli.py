#!/usr/bin/env python3
"""Fault propagation verifier CLI.

Usage (single case):
    uv run python -m verifier.cli run <case_dir> [--model X] [--judge-model Y]

Usage (batch):
    uv run python -m verifier.cli batch <dataset_dir> \\
        --run-dir /tmp/verifier-run --model doubao --judge-model azure-gpt --limit 10
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

from .lib.fpg import assemble_scenario
from .lib.quality import build_quality_report
from .prepare import prepare_case

REPO = Path(__file__).parents[3]
WORKFLOW_SCRIPT = Path(__file__).parent / "workflow.py"

_WORKFLOW_EXTENSIONS = [
    ("agentm.extensions.builtin.operations", {"backend": "local"}),
    ("agentm.extensions.builtin.retry_policy", {}),
    ("agentm.extensions.builtin.observability", {}),
    ("agentm.extensions.builtin.artifact_store", {}),
    ("agentm.extensions.builtin.workflow", {}),
]


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------

async def _run_workflow(
    workflow_args: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    from agentm.core.abi import AgentSessionConfig
    from agentm.core.runtime import AgentSession

    os.environ["AGENTM_PROJECT_ROOT"] = str(REPO)

    config = AgentSessionConfig(
        cwd=str(out_dir),
        model=os.environ.get("AGENTM_MODEL"),
        extensions=[(m, dict(c)) for m, c in _WORKFLOW_EXTENSIONS],
        auto_commit=False,
    )
    session = await AgentSession.create(config)
    sid = session.session_id
    tid = session.root_session_id
    logger.info("session:  {}", sid)
    logger.info("trace_id: {}", tid)
    logger.info("children: agentm trace index --format ndjson | grep {}", tid)

    def _on_phase(event: Any) -> None:
        kind = getattr(event, "kind", "log")
        text = getattr(event, "text", "")
        if kind == "phase":
            logger.info("▸ {}", text)
        else:
            logger.info("  {}", text)

    session.bus.on("workflow_phase", _on_phase)

    try:
        runner = session.get_service("workflow_runner")
        if runner is None:
            raise RuntimeError("workflow_runner service not found")
        result = await runner.run_file(WORKFLOW_SCRIPT, workflow_args)
    finally:
        try:
            await session.shutdown()
        except Exception:  # noqa: BLE001
            logger.warning("session shutdown failed", exc_info=True)
    return result if isinstance(result, dict) else {}


def _write_outputs(
    out: Path,
    meta: dict[str, Any],
    result: dict[str, Any],
    data_profile: dict[str, Any] | None = None,
) -> None:
    confirmed_seed_ids = set(result.get("confirmed_seeds", []))
    scenario = assemble_scenario(
        meta,
        result["nodes"],
        result["edges"],
        confirmed_seed_ids=confirmed_seed_ids,
    )
    (out / "fpg_scenario.json").write_text(
        json.dumps(scenario, indent=2, ensure_ascii=False) + "\n"
    )
    run_meta: dict[str, Any] = {
        "scenario_id": meta.get("scenario_id"),
        "testbed": meta.get("testbed"),
        "hop_log": result.get("hop_log", []),
        "rounds": result.get("rounds", 0),
        "verdicts": result.get("verdicts", {}),
    }
    if result.get("gate_log"):
        run_meta["gate_log"] = result["gate_log"]
    if result.get("audit_rounds"):
        run_meta["audit_rounds"] = result["audit_rounds"]
    if result.get("audit"):
        run_meta["audit"] = result["audit"]
    if result.get("execution_errors"):
        run_meta["execution_errors"] = result["execution_errors"]
    if result.get("unreachable_seeds"):
        run_meta["unreachable_seeds"] = result["unreachable_seeds"]
    if result.get("reachability_warnings"):
        run_meta["reachability_warnings"] = result["reachability_warnings"]
    if result.get("seed_verdicts"):
        run_meta["seed_verdicts"] = result["seed_verdicts"]
    if result.get("confirmed_seeds") is not None:
        run_meta["confirmed_seeds"] = result.get("confirmed_seeds", [])
    for key in (
        "anomaly_inventory",
        "candidate_edges",
        "node_attribution",
        "review_notes",
    ):
        if result.get(key):
            run_meta[key] = result[key]
    (out / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    _write_review_packet(out, meta, result, scenario, data_profile or {})


def _write_review_packet(
    out: Path,
    meta: dict[str, Any],
    result: dict[str, Any],
    scenario: dict[str, Any] | None,
    data_profile: dict[str, Any],
    error: str | None = None,
) -> None:
    """Write the compact human-adjudication packet for offline labeling."""
    packet = {
        "scenario_id": meta.get("scenario_id"),
        "testbed": meta.get("testbed"),
        "error": error,
        "candidate_fpg": scenario or {},
        "data_profile": data_profile,
        "anomaly_inventory": result.get("anomaly_inventory", []),
        "evidence_ledger": {
            "seed_verdicts": result.get("seed_verdicts", {}),
            "hop_verdicts": result.get("verdicts", {}),
            "gate_log": result.get("gate_log", []),
            "hop_log": result.get("hop_log", []),
            "audit": result.get("audit", {}),
            "audit_rounds": result.get("audit_rounds", []),
        },
        "candidate_edges": result.get("candidate_edges", []),
        "node_attribution": result.get("node_attribution", {}),
        "review_notes": result.get("review_notes", []),
        "human_adjudication": {
            "status": "pending",
            "instructions": (
                "Review the candidate FPG against the evidence ledger. "
                "Mark changed telemetry as covered, unrelated/pre-existing, "
                "or missed; decide any multi-fault gate semantics manually."
            ),
        },
    }
    (out / "review_packet.json").write_text(
        json.dumps(packet, indent=2, ensure_ascii=False, default=str) + "\n"
    )


def _write_error_meta(
    out: Path,
    error: str,
    result: dict[str, Any],
    meta: dict[str, Any],
    data_profile: dict[str, Any] | None = None,
) -> None:
    run_meta: dict[str, Any] = {"error": error}
    if meta.get("scenario_id"):
        run_meta["scenario_id"] = meta["scenario_id"]
    if meta.get("testbed"):
        run_meta["testbed"] = meta["testbed"]
    if result.get("hop_log"):
        run_meta["hop_log"] = result["hop_log"]
    if result.get("rounds") is not None:
        run_meta["rounds"] = result.get("rounds", 0)
    if result.get("verdicts"):
        run_meta["verdicts"] = result["verdicts"]
    if result.get("gate_log"):
        run_meta["gate_log"] = result["gate_log"]
    if result.get("audit_rounds"):
        run_meta["audit_rounds"] = result["audit_rounds"]
    if result.get("audit"):
        run_meta["audit"] = result["audit"]
    if result.get("execution_errors"):
        run_meta["execution_errors"] = result["execution_errors"]
    if result.get("unreachable_seeds"):
        run_meta["unreachable_seeds"] = result["unreachable_seeds"]
    if result.get("reachability_warnings"):
        run_meta["reachability_warnings"] = result["reachability_warnings"]
    if result.get("seed_verdicts"):
        run_meta["seed_verdicts"] = result["seed_verdicts"]
    if result.get("confirmed_seeds") is not None:
        run_meta["confirmed_seeds"] = result.get("confirmed_seeds", [])
    for key in (
        "anomaly_inventory",
        "candidate_edges",
        "node_attribution",
        "review_notes",
    ):
        if result.get(key):
            run_meta[key] = result[key]
    (out / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    _write_review_packet(out, meta, result, None, data_profile or {}, error=error)


def _run_one(
    case_dir: Path,
    out_dir: Path,
    budget: int = 15,
    judge_model: str | None = None,
    gate_retries: int = 3,
    skip_judge: bool = False,
) -> dict:
    data_dir = case_dir.resolve()
    out = out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ctx = prepare_case(data_dir)

    logger.info(
        "Injections: {}",
        [(i.get("node_id", i["target"]), i["chaos_type"]) for i in ctx.injections],
    )
    logger.info("Graph: {} edges, {} services",
                sum(len(v) for v in ctx.graph.values()), len(ctx.graph))

    workflow_args = ctx.to_workflow_args(
        out_dir=str(out),
        budget=budget,
        skip_judge=skip_judge,
        judge_model=judge_model,
        gate_retries=gate_retries,
    )
    result = asyncio.run(_run_workflow(workflow_args, out))

    if result.get("execution_errors"):
        logger.warning("Verifier execution errors: {}", result["execution_errors"])
        _write_error_meta(out, "execution errors", result, ctx.meta, ctx.data_profile)
        return {"case": data_dir.name, "error": "execution errors"}

    if not result.get("confirmed_seeds"):
        logger.warning("No seeds confirmed.")
        _write_error_meta(out, "no seeds confirmed", result, ctx.meta, ctx.data_profile)
        return {"case": data_dir.name, "error": "no seeds confirmed"}

    _write_outputs(out, ctx.meta, result, ctx.data_profile)

    seeds = {i.get("node_id", i["target"]) for i in ctx.injections}
    confirmed = [n["id"] for n in result["nodes"]]
    propagated = [s for s in confirmed if s not in seeds]

    logger.info("Rounds: {}", result['rounds'])
    logger.info("Confirmed: {}", confirmed)
    logger.info("Propagated: {}", propagated)

    return {
        "case": data_dir.name,
        "seeds": sorted(seeds),
        "confirmed": confirmed,
        "propagated": propagated,
        "rounds": result["rounds"],
    }


def _read_cached(out_dir: Path, case_name: str) -> dict | None:
    scenario_path = out_dir / "fpg_scenario.json"
    meta_path = out_dir / "run_meta.json"
    if scenario_path.exists():
        try:
            scenario = json.loads(scenario_path.read_text())
        except Exception:  # noqa: BLE001
            logger.debug("Failed to parse {}", scenario_path)
            return None
        seeds = [i["node_id"] for i in scenario.get("injections", [])]
        confirmed = [n["id"] for n in scenario.get("graph", {}).get("nodes", [])]
        return {
            "case": case_name,
            "seeds": seeds,
            "confirmed": confirmed,
            "propagated": [s for s in confirmed if s not in seeds],
            "cached": True,
        }
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:  # noqa: BLE001
            logger.debug("Failed to parse {}", meta_path)
            return None
        if meta.get("error"):
            return {"case": case_name, "error": meta["error"], "cached": True}
    return None


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

app = typer.Typer(
    name="verifier",
    help=__doc__,
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    from agentm.env import autoload_dotenv

    autoload_dotenv(REPO)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def run(
    case_dir: Annotated[Path, typer.Argument(help="single case directory")],
    out: Annotated[Path | None, typer.Option(help="output directory")] = None,
    budget: Annotated[int, typer.Option(help="tool-call budget per agent")] = 15,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
    judge_model: Annotated[
        str | None,
        typer.Option(
            "--judge-model",
            help="config.toml profile name used only by the judge agent",
        ),
    ] = None,
    gate_retries: Annotated[
        int,
        typer.Option(
            "--gate-retries",
            help="retry a seed/hop this many times after its gate rejects it",
        ),
    ] = 3,
    skip_judge: Annotated[
        bool,
        typer.Option(
            "--skip-judge",
            help="skip the audit loop after seed/hop propagation",
        ),
    ] = False,
) -> None:
    """Run propagation check on a single case."""
    if model:
        os.environ["AGENTM_MODEL"] = model
    out_dir = out or case_dir.resolve() / ".verify_propagate"
    summary = _run_one(
        case_dir,
        out_dir,
        budget=budget,
        judge_model=judge_model,
        gate_retries=gate_retries,
        skip_judge=skip_judge,
    )
    if "error" in summary:
        raise typer.Exit(1)


@app.command()
def quality_report(
    run_dir: Annotated[Path, typer.Argument(help="verifier batch run directory")],
    out: Annotated[Path | None, typer.Option(help="output JSON path")] = None,
    dataset_dir: Annotated[
        Path | None,
        typer.Option(help="dataset cases directory, used to include fault grouping"),
    ] = None,
) -> None:
    """Write a JSON quality report for a completed verifier batch run."""
    run = run_dir.resolve()
    if not run.is_dir():
        raise typer.BadParameter(f"run_dir does not exist or is not a directory: {run}")
    dataset = dataset_dir.resolve() if dataset_dir else None
    if dataset is not None and not dataset.is_dir():
        raise typer.BadParameter(
            f"dataset_dir does not exist or is not a directory: {dataset}"
        )
    report = build_quality_report(run, dataset)
    out_path = out.resolve() if out else run / "quality_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = report["summary"]
    logger.info(
        "Quality: total={} passed={} failed={}",
        summary["total_cases"],
        summary["passed"],
        summary["failed"],
    )
    logger.info("Report: {}", out_path)


@app.command()
def batch(
    dataset_dir: Annotated[Path, typer.Argument(help="dataset directory")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="output directory")],
    budget: Annotated[int, typer.Option(help="tool-call budget per agent")] = 15,
    parallel: Annotated[int, typer.Option(help="concurrent cases")] = 1,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
    judge_model: Annotated[
        str | None,
        typer.Option(
            "--judge-model",
            help="config.toml profile name used only by the judge agent",
        ),
    ] = None,
    gate_retries: Annotated[
        int,
        typer.Option(
            "--gate-retries",
            help="retry each seed/hop this many times after its gate rejects it",
        ),
    ] = 3,
    limit: Annotated[int | None, typer.Option(help="max cases")] = None,
    offset: Annotated[int, typer.Option(help="skip first N cases")] = 0,
) -> None:
    """Run propagation on all cases in a dataset."""
    if model:
        os.environ["AGENTM_MODEL"] = model
    dataset = dataset_dir.resolve()
    out = run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name for p in dataset.iterdir() if p.is_dir())
    cases = cases[offset:]
    if limit is not None:
        cases = cases[:limit]
    total = len(cases)

    def _do(name: str, idx: int) -> dict:
        case_out = out / name
        cached = _read_cached(case_out, name)
        if cached:
            logger.info("[{}/{}] {} CACHED", idx, total, name)
            return cached
        logger.info("[{}/{}] {} ...", idx, total, name)
        try:
            return _run_one(
                dataset / name,
                case_out,
                budget=budget,
                judge_model=judge_model,
                gate_retries=gate_retries,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[{}] {}", name, exc)
            return {"case": name, "error": str(exc)}

    shutdown_event = threading.Event()

    def _sigint_handler(sig: int, frame: object) -> None:
        logger.warning("Ctrl+C received — finishing in-progress cases, no new ones will start")
        shutdown_event.set()

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    def _do_guarded(name: str, idx: int) -> dict:
        if shutdown_event.is_set():
            return {"case": name, "error": "interrupted"}
        return _do(name, idx)

    try:
        if parallel <= 1:
            summaries = []
            for i, name in enumerate(cases, 1):
                if shutdown_event.is_set():
                    logger.info("Skipping remaining cases")
                    break
                summaries.append(_do(name, i))
        else:
            results: dict[str, dict] = {}
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {
                    pool.submit(_do_guarded, name, i): name
                    for i, name in enumerate(cases, 1)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    if shutdown_event.is_set():
                        for f in futures:
                            f.cancel()
                        break
            summaries = [
                results.get(name, {"case": name, "error": "interrupted"})
                for name in cases
            ]
    finally:
        signal.signal(signal.SIGINT, original_handler)

    summary_path = out / "run_summary.jsonl"
    with open(summary_path, "w") as fout:
        for s in summaries:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

    ok = sum(1 for s in summaries if "error" not in s)
    fail = total - ok
    logger.info("Total: {}  OK: {}  Failed: {}", total, ok, fail)
    logger.info("Summary: {}", summary_path)


if __name__ == "__main__":
    app()
