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

            asyncio.run(_run_sessions(sids, model=model, chunk_size=parsed_chunk, vocabulary=vocabulary, exp=exp))

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
) -> None:
    from agentm_eval.methods.index import (
        _chunk_messages,
        _prescan_structural,
        _run_one,
        resolve_index,
    )
    from trajectory_index.index import TrajectoryIndex, normalize_name

    total = len(sids)
    all_results: list[dict[str, Any]] = []

    for i, sid in enumerate(sids):
        typer.echo(f"\n[{i+1}/{total}] {sid}")

        try:
            messages = await _load_messages(sid)
        except FileNotFoundError:
            typer.echo("  NOT FOUND", err=True)
            continue

        typer.echo(f"  loaded {len(messages)} messages")
        exp.register_session(sid, metadata={"n_messages": len(messages)})

        idx = TrajectoryIndex()
        registry: list[dict[str, Any]] = []
        seen: set[str] = set()
        n_chunks = 0

        if chunk_size is None:
            raw_chunks = [type("C", (), {"start": 0, "messages": messages})]
        else:
            raw_chunks = _chunk_messages(messages, chunk_size)

        for ci, chunk in enumerate(raw_chunks):
            chunk_registry, _structural = _prescan_structural(
                chunk.messages, registry if registry else None, seen,
            )
            try:
                result = await _run_one(
                    chunk.messages, model=model, vocabulary=vocabulary,
                    registry=chunk_registry if chunk_registry else None,
                    message_id_start=chunk.start, cwd=None,
                )
            except Exception:
                typer.echo(f"  chunk {ci+1} extraction failed", err=True)
                continue
            if result is None:
                typer.echo(f"  chunk {ci+1} no parseable result", err=True)
                continue

            n_chunks += 1
            n_new = len(result.symbols)

            idx.populate_from_extraction(result, chunk.messages, run_id=sid)

            # Pass 2+3 after each chunk (matches online behavior)
            await resolve_index(idx, model=model)

            # Accumulate registry for next chunk
            for sym in result.symbols:
                norm = normalize_name(sym.name)
                if norm not in seen:
                    seen.add(norm)
                    entry: dict[str, Any] = {"name": sym.name, "kind": sym.kind}
                    if sym.summary:
                        entry["summary"] = sym.summary
                    if sym.aliases:
                        entry["aliases"] = sym.aliases
                    registry.append(entry)

            stats_snap = idx.stats(sid)
            symbols_data = [
                {"name": s.name, "kind": s.kind, "entity_class": s.entity_class, "summary": s.summary}
                for s in result.symbols
            ]
            exp.write_session_artifact(sid, "index", f"chunk_{n_chunks:03d}.json", {
                "chunk": n_chunks,
                "n_messages": len(chunk.messages),
                "n_new_symbols": n_new,
                "symbols": symbols_data,
                "index_cumulative": {
                    "symbols": stats_snap.symbol_count,
                    "references": stats_snap.reference_count,
                    "dependencies": stats_snap.dependency_count,
                },
            })

            sym_preview = ", ".join(s.name for s in result.symbols[:3])
            if n_new > 3:
                sym_preview += f", ... (+{n_new - 3})"
            typer.echo(
                f"  chunk {n_chunks}: {len(chunk.messages)} msgs → {n_new} new symbols "
                f"[{sym_preview}] (total: {stats_snap.symbol_count} syms, "
                f"{stats_snap.reference_count} refs, {stats_snap.dependency_count} deps)"
            )

        stats = idx.stats(sid)
        result_dict = {
            "session_id": sid,
            "n_messages": len(messages),
            "n_chunks": n_chunks,
            "n_symbols": stats.symbol_count,
            "n_references": stats.reference_count,
            "n_dependencies": stats.dependency_count,
        }
        all_results.append(result_dict)
        exp.record_result(result_dict)
        exp.write_session_artifact(sid, "", "index_stats.json", result_dict)
        idx.dump(str(exp.session_dir(sid) / "index.json"))

        typer.echo(f"  total: {stats.symbol_count} symbols, {stats.reference_count} refs, {stats.dependency_count} deps")

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
