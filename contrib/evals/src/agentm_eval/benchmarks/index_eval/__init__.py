"""Standalone trajectory-index evaluation benchmark.

Loads recorded sessions from ClickHouse, runs the extraction pipeline,
and saves per-session per-chunk results for review.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from agentm_eval.experiment import Experiment
from agentm_eval.registry import register


class IndexEvalAdapter:
    name = "index"
    description = "Trajectory-index extraction evaluation"

    def create_cli(self) -> typer.Typer:
        cli = typer.Typer(name="index", help="Evaluate trajectory-index extraction.", add_completion=False)

        @cli.command()
        def run(
            session_id: Annotated[list[str] | None, typer.Option("--session", help="Session IDs")] = None,
            session_file: Annotated[Path | None, typer.Option("--session-file")] = None,
            model: Annotated[str, typer.Option(help="Extraction model profile")] = "azure-gpt",
            chunk_size: Annotated[str | None, typer.Option("--chunk-size", help="'3-6' or omit for full")] = "3-6",
            vocabulary: Annotated[str, typer.Option()] = "default",
            exp_id: Annotated[str | None, typer.Option("--exp-id")] = None,
            session_concurrency: Annotated[int, typer.Option("--session-concurrency", help="Parallel sessions")] = 4,
        ) -> None:
            """Run trajectory-index extraction on recorded sessions."""
            from agentm.env import autoload_dotenv
            autoload_dotenv()

            sids = list(session_id or [])
            if session_file:
                sids.extend(
                    line.strip() for line in session_file.read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                )
            if not sids:
                typer.echo("No session IDs. Use --session or --session-file.", err=True)
                raise typer.Exit(1)

            parsed_chunk = _parse_chunk_size(chunk_size)
            exp = Experiment.create("index-eval", model=model, exp_id=exp_id, vocabulary=vocabulary)
            typer.echo(f"Experiment: {exp.exp_id} → {exp.output_dir}")
            typer.echo(f"Sessions: {len(sids)}, model={model}, chunk_size={chunk_size}")

            asyncio.run(_run_sessions(
                sids, model=model, chunk_size=parsed_chunk, vocabulary=vocabulary, exp=exp,
                session_concurrency=session_concurrency,
            ))

        return cli


def _parse_chunk_size(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = value.split("-")
    return (int(parts[0]), int(parts[-1]))


async def _load_messages(session_id: str) -> list[Any]:
    from agentm.core.runtime.session_bootstrap import make_default_session_store
    store = make_default_session_store(".")
    return store.open(session_id).get_raw_messages()


async def _run_sessions(
    sids: list[str],
    *,
    model: str,
    chunk_size: tuple[int, int] | None,
    vocabulary: str,
    exp: Experiment,
    session_concurrency: int = 4,
) -> None:
    from agentm_eval.methods.index import build_index, extract_symbols

    total = len(sids)
    sem = asyncio.Semaphore(max(1, session_concurrency))

    async def _one_session(i: int, sid: str) -> dict[str, Any] | None:
        async with sem:
            typer.echo(f"[{i+1}/{total}] {sid} — start")
            try:
                messages = await _load_messages(sid)
            except FileNotFoundError:
                typer.echo(f"  {sid}: NOT FOUND", err=True)
                return None
            exp.register_session(sid, metadata={"n_messages": len(messages)})

            # Pass 1 (extraction): chunks extracted sequentially with inline
            # registry accumulation; Pass 2 (build_index) resolves aliases once.
            chunks = await extract_symbols(
                messages, model=model, vocabulary=vocabulary,
                chunk_size=chunk_size, run_id=sid,
            )
            idx = await build_index(chunks, model=model, resolve=True)

            stats = idx.stats(sid)
            warns = idx.warnings()
            warn_summary = idx.warning_summary()
            result_dict = {
                "session_id": sid,
                "n_messages": len(messages),
                "n_chunks": len(chunks),
                "n_symbols": stats.symbol_count,
                "n_references": stats.reference_count,
                "n_dependencies": stats.dependency_count,
                "warnings": warn_summary,
            }
            exp.record_result(result_dict)
            warnings_data = [
                {"kind": w.kind, "symbol": w.symbol_name, "detail": w.detail, "steps": list(w.step_ids)}
                for w in warns
            ]
            exp.write_session_artifact(sid, "", "index_stats.json", result_dict)
            exp.write_session_artifact(sid, "", "warnings.json", warnings_data)
            idx.dump(str(exp.session_dir(sid) / "index.json"))
            typer.echo(
                f"[{i+1}/{total}] {sid}: {stats.symbol_count} syms, "
                f"{stats.reference_count} refs, {stats.dependency_count} deps "
                f"({len(chunks)} chunks)"
            )
            return result_dict

    gathered = await asyncio.gather(*(_one_session(i, sid) for i, sid in enumerate(sids)))
    all_results = [r for r in gathered if r is not None]

    if all_results:
        avg_sym = sum(r["n_symbols"] for r in all_results) / len(all_results)
        avg_ref = sum(r["n_references"] for r in all_results) / len(all_results)
        summary = {"n_sessions": len(all_results), "avg_symbols": round(avg_sym, 1), "avg_references": round(avg_ref, 1)}
        exp.finish(summary=summary)
        typer.echo(f"\nSummary: {json.dumps(summary)}")
    else:
        exp.finish(status="failed", summary={"error": "no sessions processed"})

    typer.echo(f"Experiment: {exp.exp_id} → {exp.output_dir}")




register("index", IndexEvalAdapter.description, IndexEvalAdapter)
