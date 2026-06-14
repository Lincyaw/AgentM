#!/usr/bin/env python3
"""Fault propagation verifier CLI.

Usage (single case):
    uv run python -m verifier.cli run <case_dir> [--model X]

Usage (batch):
    uv run python -m verifier.cli batch <dataset_dir> \\
        --run-dir /tmp/verifier-run --model doubao --limit 10
"""
from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

from .lib.fpg import assemble_scenario
from .prepare import prepare_case

REPO = Path(__file__).resolve().parents[3]
WORKFLOW_SCRIPT = Path(__file__).resolve().parent / "propagation_workflow.py"

_WORKFLOW_EXTENSIONS = [
    ("agentm.extensions.builtin.operations", {"backend": "local"}),
    ("agentm.extensions.builtin.observability", {}),
    ("agentm.extensions.builtin.artifact_store", {}),
    ("agentm.extensions.builtin.workflow", {}),
]


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------

def _resolve_provider() -> tuple[str, dict[str, Any]]:
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


async def _run_workflow(
    workflow_args: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    from agentm.core.abi import AgentSessionConfig
    from agentm.core.runtime import AgentSession

    os.environ["AGENTM_PROJECT_ROOT"] = str(REPO)

    config = AgentSessionConfig(
        cwd=str(out_dir),
        provider=_resolve_provider(),
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
        return result if isinstance(result, dict) else {}
    finally:
        await session.shutdown()


def _write_outputs(
    out: Path,
    meta: dict[str, Any],
    result: dict[str, Any],
) -> None:
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
    if result.get("judge"):
        run_meta["judge"] = result["judge"]
    (out / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False, default=str) + "\n"
    )


def _run_one(case_dir: Path, out_dir: Path, budget: int = 15) -> dict:
    data_dir = case_dir.resolve()
    out = out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ctx = prepare_case(data_dir)

    logger.info("Injections: {}", [(i['target'], i['chaos_type']) for i in ctx.injections])
    logger.info("Graph: {} edges, {} services",
                sum(len(v) for v in ctx.graph.values()), len(ctx.graph))

    workflow_args = ctx.to_workflow_args(out_dir=str(out), budget=budget)
    result = asyncio.run(_run_workflow(workflow_args, out))

    if not result.get("nodes"):
        logger.warning("No seeds confirmed.")
        return {"case": data_dir.name, "error": "no seeds confirmed"}

    _write_outputs(out, ctx.meta, result)

    seeds = {i["target"] for i in ctx.injections}
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
    if not scenario_path.exists():
        return None
    try:
        scenario = json.loads(scenario_path.read_text())
    except Exception:  # noqa: BLE001
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
    from agentm.cli import autoload_dotenv
    autoload_dotenv(REPO)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def run(
    case_dir: Annotated[Path, typer.Argument(help="single case directory")],
    out: Annotated[Path | None, typer.Option(help="output directory")] = None,
    budget: Annotated[int, typer.Option(help="tool-call budget per agent")] = 15,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
) -> None:
    """Run propagation check on a single case."""
    if model:
        os.environ["AGENTM_MODEL"] = model
    out_dir = out or case_dir.resolve() / ".verify_propagate"
    summary = _run_one(case_dir, out_dir, budget=budget)
    if "error" in summary:
        raise typer.Exit(1)


@app.command()
def batch(
    dataset_dir: Annotated[Path, typer.Argument(help="dataset directory")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="output directory")],
    budget: Annotated[int, typer.Option(help="tool-call budget per agent")] = 15,
    parallel: Annotated[int, typer.Option(help="concurrent cases")] = 1,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
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
            return _run_one(dataset / name, case_out, budget=budget)
        except Exception as exc:  # noqa: BLE001
            logger.error("[{}] {}", name, exc)
            return {"case": name, "error": str(exc)}

    if parallel <= 1:
        summaries = [_do(name, i) for i, name in enumerate(cases, 1)]
    else:
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_do, name, i): name
                for i, name in enumerate(cases, 1)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        summaries = [results[name] for name in cases]

    summary_path = out / "run_summary.jsonl"
    with open(summary_path, "w") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    ok = sum(1 for s in summaries if "error" not in s)
    fail = total - ok
    logger.info("Total: {}  OK: {}  Failed: {}", total, ok, fail)
    logger.info("Summary: {}", summary_path)


if __name__ == "__main__":
    app()
