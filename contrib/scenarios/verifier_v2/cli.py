#!/usr/bin/env python3
"""Verifier v2 CLI — convergence loop with Searcher/Compiler/Judge.

Usage:
    uv run python -m verifier_v2.cli run <case_dir> [--out <dir>]
    uv run python -m verifier_v2.cli batch <dataset_dir> --run-dir <dir> [--limit N]
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

# Reuse case preparation from v1 (domain knowledge, not architecture)
from contrib.scenarios.verifier.prepare import prepare_case

REPO = Path(__file__).parents[3]
WORKFLOW_SCRIPT = Path(__file__).parent / "workflow.py"

_ENTRY_KEYWORDS = {"frontend", "ui", "dashboard", "gateway", "proxy", "loadgenerator"}


def _detect_entry_services(data_dir: Path, graph: dict[str, list[list[str]]]) -> list[str]:
    """Detect entry services from trace data + keyword fallback.

    Primary signal: services with root spans (parent_span_id = '').
    Secondary: keyword matching on service names in the call graph.
    """
    root_services: set[str] = set()
    normal_parquet = data_dir / "normal_traces.parquet"
    if normal_parquet.exists():
        try:
            import duckdb

            conn = duckdb.connect(":memory:")
            path_str = normal_parquet.as_posix().replace("'", "''")
            rows = conn.execute(
                "SELECT service_name, "
                "       count(*) AS root_spans, "
                "       (SELECT count(*) FROM read_parquet('" + path_str + "') t2 "
                "        WHERE t2.service_name = t1.service_name) AS total_spans "
                "FROM read_parquet('" + path_str + "') t1 "
                "WHERE parent_span_id = '' "
                "GROUP BY service_name"
            ).fetchall()
            for svc, root_cnt, total_cnt in rows:
                ratio = root_cnt / total_cnt if total_cnt else 0
                if total_cnt < 50:
                    continue
                if root_cnt >= 100 or ratio > 0.2:
                    root_services.add(svc)
            conn.close()
        except Exception:  # noqa: BLE001, S110
            pass

    all_services = set(graph.keys())
    for neighbors in graph.values():
        for info in neighbors:
            if info:
                all_services.add(str(info[0]))
    keyword_services = {
        s for s in all_services if any(kw in s.lower() for kw in _ENTRY_KEYWORDS)
    }

    combined = root_services | keyword_services
    return sorted(combined & all_services) if combined else []

_WORKFLOW_EXTENSIONS = [
    ("agentm.extensions.builtin.operations", {"backend": "local"}),
    ("agentm.extensions.builtin.retry_policy", {}),
    ("agentm.extensions.builtin.observability", {}),
    ("agentm.extensions.builtin.artifact_store", {}),
    ("agentm.extensions.builtin.workflow", {}),
]

app = typer.Typer(pretty_exceptions_enable=False)


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
    logger.info("session:  {}", session.session_id)
    logger.info("trace_id: {}", session.root_session_id)

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
        except Exception:  # noqa: BLE001, S110
            pass
    out = result if isinstance(result, dict) else {}
    out["session_id"] = session.session_id
    out["trace_id"] = session.root_session_id
    return out


@app.command()
def run(
    case_dir: Annotated[Path, typer.Argument(help="Case directory with parquet + injection.json")],
    out: Annotated[Path | None, typer.Option("--out", help="Output directory")] = None,
    max_parallel: Annotated[int, typer.Option(help="Max concurrent agents")] = 8,
    max_retries: Annotated[int, typer.Option(help="Max retries per task")] = 3,
) -> None:
    """Run verifier v2 on a single case."""
    case_dir = case_dir.resolve()
    if not case_dir.is_dir():
        logger.error("Not a directory: {}", case_dir)
        raise typer.Exit(1)

    case_name = case_dir.name
    out_dir = out or Path(f"runs/v2-{case_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Case: {}", case_name)
    logger.info("Output: {}", out_dir)

    # Prepare using v1's case preparation (reusable domain logic)
    ctx = prepare_case(case_dir)
    workflow_args = ctx.to_workflow_args(
        out_dir=str(out_dir),
        skip_judge=False,
        max_parallel_tasks=max_parallel,
    )
    # Adapt args for v2 schema
    workflow_args["max_parallel"] = max_parallel
    workflow_args["max_retries"] = max_retries
    workflow_args["entry_services"] = _detect_entry_services(
        case_dir, workflow_args.get("graph", {}),
    )

    result = asyncio.run(_run_workflow(workflow_args, out_dir))

    # Write outputs
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n"
    )

    gaps_satisfied = result.get("gaps_satisfied", False)
    n_nodes = len(result.get("nodes", []))
    n_edges = len(result.get("edges", []))
    n_calls = result.get("total_agent_calls", 0)
    n_rounds = result.get("total_rounds", 0)

    status = "✓ PASS" if gaps_satisfied else "✗ FAIL"
    logger.info(
        "{} | nodes={} edges={} | rounds={} agent_calls={} | gaps_remaining={}",
        status, n_nodes, n_edges, n_rounds, n_calls,
        len(result.get("remaining_gaps", [])),
    )


@app.command()
def batch(
    dataset_dir: Annotated[Path, typer.Argument(help="Dataset directory containing case dirs")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="Batch output directory")],
    limit: Annotated[int, typer.Option(help="Max cases to run")] = 0,
    max_parallel: Annotated[int, typer.Option(help="Max concurrent agents per case")] = 8,
    cases: Annotated[str | None, typer.Option(help="Comma-separated case name filters")] = None,
) -> None:
    """Run verifier v2 on a batch of cases."""
    dataset_dir = dataset_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        d for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "injection.json").exists()
    )

    if cases:
        filters = [f.strip() for f in cases.split(",")]
        case_dirs = [d for d in case_dirs if any(f in d.name for f in filters)]

    if limit > 0:
        case_dirs = case_dirs[:limit]

    logger.info("Running {} cases", len(case_dirs))

    results: list[dict[str, Any]] = []
    for i, case_dir in enumerate(case_dirs):
        case_name = case_dir.name
        out_dir = run_dir / case_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[{}/{}] {}", i + 1, len(case_dirs), case_name)
        try:
            ctx = prepare_case(case_dir)
            workflow_args = ctx.to_workflow_args(
                out_dir=str(out_dir),
                skip_judge=False,
                max_parallel_tasks=max_parallel,
            )
            workflow_args["max_parallel"] = max_parallel
            workflow_args["entry_services"] = _detect_entry_services(
                case_dir, workflow_args.get("graph", {}),
            )

            result = asyncio.run(_run_workflow(workflow_args, out_dir))
            (out_dir / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n"
            )
            status = "PASS" if result.get("gaps_satisfied") else "FAIL"
            results.append({"case": case_name, "status": status, **result})
            logger.info("  → {}", status)
        except Exception as exc:  # noqa: BLE001
            logger.error("  → ERROR: {}", exc)
            results.append({"case": case_name, "status": "ERROR", "error": str(exc)})

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    logger.info("Done: {}/{} PASS, {} FAIL, {} ERROR", passed, len(results), failed, errors)

    (run_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n"
    )


if __name__ == "__main__":
    app()
