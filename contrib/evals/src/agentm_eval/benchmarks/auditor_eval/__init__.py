"""Auditor evaluation benchmark.

Loads recorded sessions, builds trajectory index (with caching),
runs the auditor, and saves all intermediate results for review.

Output structure per session:
  sessions/<sid>/index.json          — cached index (reused across runs)
  sessions/<sid>/index_stats.json    — index stats
  sessions/<sid>/warnings.json       — grounding warnings
  sessions/<sid>/auditor/result.json — auditor verdict + surfaces
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from agentm_eval.experiment import Experiment
from agentm_eval.registry import register


class AuditorEvalAdapter:
    name = "auditor"
    description = "Auditor evaluation — index + audit on recorded sessions"

    def create_cli(self) -> typer.Typer:
        cli = typer.Typer(name="auditor", help="Evaluate auditor on recorded sessions.", add_completion=False)

        @cli.command()
        def run(
            session_id: Annotated[list[str] | None, typer.Option("--session", help="Session IDs")] = None,
            session_file: Annotated[Path | None, typer.Option("--session-file")] = None,
            model: Annotated[str, typer.Option(help="Auditor model profile")] = "azure-gpt",
            index_model: Annotated[str, typer.Option("--index-model", help="Index extraction model")] = "azure-gpt",
            chunk_size: Annotated[str | None, typer.Option("--chunk-size")] = "8-12",
            auditor_prompt: Annotated[str, typer.Option("--prompt")] = "index",
            audit_interval: Annotated[int | None, typer.Option("--interval", help="Fire every N turns (None=post-hoc)")] = None,
            exp_id: Annotated[str | None, typer.Option("--exp-id")] = None,
            rebuild_index: Annotated[bool, typer.Option("--rebuild-index", help="Force rebuild index even if cached")] = False,
        ) -> None:
            """Run auditor on recorded sessions (builds index first)."""
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
            exp = Experiment.create(
                "auditor-eval", model=model, exp_id=exp_id,
                index_model=index_model, auditor_prompt=auditor_prompt,
            )
            typer.echo(f"Experiment: {exp.exp_id} → {exp.output_dir}")
            typer.echo(f"Sessions: {len(sids)}, model={model}, index_model={index_model}")

            asyncio.run(_run_sessions(
                sids, model=model, index_model=index_model,
                chunk_size=parsed_chunk, auditor_prompt=auditor_prompt,
                audit_interval=audit_interval, exp=exp,
                rebuild_index=rebuild_index,
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


async def _build_or_load_index(
    sid: str,
    messages: list[Any],
    *,
    model: str,
    chunk_size: tuple[int, int] | None,
    exp: Experiment,
    rebuild: bool = False,
) -> Any:
    """Build index for a session, or load from cache if available."""
    from trajectory_index.index import TrajectoryIndex

    index_path = exp.session_dir(sid) / "index.json"
    if index_path.is_file() and not rebuild:
        typer.echo("  index: loading from cache")
        idx = TrajectoryIndex.load(index_path)
        idx.build_dependencies()
        return idx

    typer.echo("  index: building...")
    from agentm_eval.methods.index import (
        _chunk_messages,
        _prescan_structural,
        _run_one,
        resolve_index,
    )
    from trajectory_index.index import normalize_name

    idx = TrajectoryIndex()
    registry: list[dict[str, Any]] = []
    seen: set[str] = set()

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
                chunk.messages, model=model, vocabulary="default",
                registry=chunk_registry if chunk_registry else None,
                message_id_start=chunk.start, cwd=None,
            )
        except Exception:
            typer.echo(f"    chunk {ci+1} extraction failed", err=True)
            continue
        if result is None:
            typer.echo(f"    chunk {ci+1} no parseable result", err=True)
            continue

        idx.populate_from_extraction(
            result, chunk.messages, run_id=sid,
            message_id_start=chunk.start,
        )
        await resolve_index(idx, model=model)

        for sym in result.symbols:
            norm = normalize_name(sym.name)
            if norm not in seen:
                seen.add(norm)
                entry: dict[str, Any] = {"name": sym.name, "kind": sym.kind}
                if sym.aliases:
                    entry["aliases"] = sym.aliases
                registry.append(entry)

        stats = idx.stats(sid)
        typer.echo(f"    chunk {ci+1}: {stats.symbol_count} syms, {stats.reference_count} refs")

    # Save index
    idx.dump(str(index_path))

    stats = idx.stats(sid)
    warns = idx.warning_summary()
    exp.write_session_artifact(sid, "", "index_stats.json", {
        "n_symbols": stats.symbol_count,
        "n_references": stats.reference_count,
        "n_dependencies": stats.dependency_count,
        "warnings": warns,
    })

    warnings_data = [
        {"kind": w.kind, "symbol": w.symbol_name, "detail": w.detail, "steps": list(w.step_ids)}
        for w in idx.warnings()
    ]
    exp.write_session_artifact(sid, "", "warnings.json", warnings_data)

    typer.echo(f"  index: {stats.symbol_count} symbols, {stats.reference_count} refs, {stats.dependency_count} deps")
    if warns:
        parts = ", ".join(f"{v} {k}" for k, v in sorted(warns.items()))
        typer.echo(f"  warnings: {parts}")

    return idx


def _index_to_context(idx: Any) -> dict[str, Any]:
    """Convert a TrajectoryIndex to the context dict the auditor expects."""
    grounding: dict[str, Any] = {"symbols": [], "warnings": []}
    for sym in idx.symbols.values():
        refs = idx.get_references(sym.id)
        grounding["symbols"].append({
            "name": sym.canonical_name,
            "kind": sym.kind,
            "entity_class": sym.entity_class,
            "n_references": len(refs),
            "grounded": any(r.grounded for r in refs),
        })
    for w in idx.warnings():
        grounding["warnings"].append({
            "kind": w.kind,
            "symbol": w.symbol_name,
            "detail": w.detail,
        })
    return grounding


async def _run_sessions(
    sids: list[str],
    *,
    model: str,
    index_model: str,
    chunk_size: tuple[int, int] | None,
    auditor_prompt: str,
    audit_interval: int | None,
    exp: Experiment,
    rebuild_index: bool,
) -> None:
    from agentm_eval.methods.auditor import run_auditor

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

        # Step 1: Build or load index
        idx = await _build_or_load_index(
            sid, messages, model=index_model,
            chunk_size=chunk_size, exp=exp, rebuild=rebuild_index,
        )

        # Step 2: Run auditor with index context
        typer.echo("  auditor: running...")
        context = _index_to_context(idx)

        try:
            audit_result = await run_auditor(
                messages, model=model,
                auditor_prompt=auditor_prompt,
                audit_interval=audit_interval,
                context_index=context,
            )
        except Exception as exc:
            typer.echo(f"  auditor FAILED: {exc}", err=True)
            continue

        # Save auditor results
        auditor_data: dict[str, Any] = {
            "session_id": sid,
            "n_verdicts": len(audit_result.verdicts),
            "n_surfaces": len(audit_result.surfaces),
            "session_ids": audit_result.session_ids,
            "verdicts": [v.to_dict() for v in audit_result.verdicts],
            "surfaces": [v.to_dict() for v in audit_result.surfaces],
        }
        exp.write_session_artifact(sid, "auditor", "result.json", auditor_data)

        typer.echo(f"  auditor: {len(audit_result.verdicts)} verdicts, {len(audit_result.surfaces)} surfaces")
        for v in audit_result.surfaces[:5]:
            typer.echo(f"    surface: {v.reminder_text[:120] if v.reminder_text else '(empty)'}")

        result_dict = {
            "session_id": sid,
            "n_messages": len(messages),
            "n_symbols": len(idx.symbols),
            "n_verdicts": len(audit_result.verdicts),
            "n_surfaces": len(audit_result.surfaces),
        }
        all_results.append(result_dict)
        exp.record_result(result_dict)

    if all_results:
        summary = {
            "n_sessions": len(all_results),
            "avg_verdicts": round(sum(r["n_verdicts"] for r in all_results) / len(all_results), 1),
            "avg_surfaces": round(sum(r["n_surfaces"] for r in all_results) / len(all_results), 1),
        }
        exp.finish(summary=summary)
        typer.echo(f"\nSummary: {json.dumps(summary)}")
    else:
        exp.finish(status="failed", summary={"error": "no sessions processed"})

    typer.echo(f"Experiment: {exp.exp_id} → {exp.output_dir}")


register("auditor", AuditorEvalAdapter.description, AuditorEvalAdapter)
