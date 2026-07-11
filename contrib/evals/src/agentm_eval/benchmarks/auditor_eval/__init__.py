"""Auditor evaluation benchmark.

Loads recorded sessions, interleaves index building with auditor
firing — matching online behavior where the auditor fires after
each index update, not after the full trajectory.

Output per session:
  sessions/<sid>/index.json              — final index (cached for reuse)
  sessions/<sid>/index_stats.json        — index stats
  sessions/<sid>/warnings.json           — grounding warnings
  sessions/<sid>/auditor/step_NNN.json   — auditor verdict at each checkpoint
  sessions/<sid>/auditor/final.json      — last auditor state
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
    description = "Auditor evaluation — interleaved index + audit"

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
            exp_id: Annotated[str | None, typer.Option("--exp-id")] = None,
            rebuild_index: Annotated[bool, typer.Option("--rebuild-index")] = False,
            max_messages: Annotated[int | None, typer.Option("--max-messages", help="Truncate trajectory to first N messages")] = None,
        ) -> None:
            """Run interleaved index + auditor on recorded sessions."""
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
                exp=exp, rebuild_index=rebuild_index,
                max_messages=max_messages,
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


def _index_to_context(idx: Any) -> dict[str, Any]:
    """Convert a TrajectoryIndex to the context dict the auditor tools expect.

    Produces ``symbols`` and ``references`` lists for the primary lookup
    path in ``auditor_index_tools._IndexState``, plus ``attention_hints``
    mapped from grounding warnings.
    """
    symbols: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []

    for sym in idx.symbols.values():
        symbols.append({
            "id": sym.id,
            "name": sym.canonical_name,
            "kind": sym.kind,
            "entity_class": sym.entity_class,
            "aliases": sorted(sym.aliases),
        })
        for ref in idx.get_references(sym.id):
            references.append({
                "symbol_id": sym.id,
                "step_id": ref.step_id,
                "kind": ref.kind,
                "text": (ref.text or "")[:200],
                "grounded": ref.grounded,
            })

    attention_hints: list[dict[str, Any]] = []
    for w in idx.warnings():
        attention_hints.append({
            "kind": w.kind,
            "summary": f"{w.symbol_name}: {w.detail}",
            "symbol": w.symbol_name,
        })

    return {
        "symbols": symbols,
        "references": references,
        "attention_hints": attention_hints,
    }


async def _run_sessions(
    sids: list[str],
    *,
    model: str,
    index_model: str,
    chunk_size: tuple[int, int] | None,
    auditor_prompt: str,
    exp: Experiment,
    rebuild_index: bool,
    max_messages: int | None = None,
) -> None:
    from agentm_eval.methods.auditor import run_auditor
    from agentm_eval.methods.index import (
        _chunk_messages,
        _match_known_symbol,
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

        if max_messages is not None and len(messages) > max_messages:
            messages = messages[:max_messages]
        typer.echo(f"  loaded {len(messages)} messages")
        exp.register_session(sid, metadata={"n_messages": len(messages)})

        # Check for cached index
        index_path = exp.session_dir(sid) / "index.json"
        if index_path.is_file() and not rebuild_index:
            typer.echo("  index: loading from cache")
            idx = TrajectoryIndex.load(index_path)
            idx.build_dependencies()

            # Run auditor once on full trajectory with cached index
            typer.echo("  auditor: running (post-hoc with cached index)...")
            context = _index_to_context(idx)
            try:
                audit_result = await run_auditor(
                    messages, model=model,
                    auditor_prompt=auditor_prompt,
                    context_index=context,
                )
                _save_auditor_step(exp, sid, len(messages), audit_result, idx)
            except Exception as exc:
                typer.echo(f"  auditor FAILED: {exc}", err=True)

            stats = idx.stats(sid)
            result_dict = {
                "session_id": sid,
                "n_messages": len(messages),
                "n_symbols": stats.symbol_count,
                "n_surfaces": len(audit_result.surfaces) if audit_result else 0,
                "cached_index": True,
            }
            all_results.append(result_dict)
            exp.record_result(result_dict)
            continue

        # Interleaved: build index chunk by chunk, fire auditor after each
        idx = TrajectoryIndex()
        registry: list[dict[str, Any]] = []
        seen: set[str] = set()

        if chunk_size is None:
            raw_chunks = [type("C", (), {"start": 0, "messages": messages})]
        else:
            raw_chunks = _chunk_messages(messages, chunk_size)

        n_auditor_fires = 0
        last_audit_result = None

        for ci, chunk in enumerate(raw_chunks):
            chunk_end = chunk.start + len(chunk.messages)

            # --- Index: extract + resolve ---
            chunk_registry, _structural = _prescan_structural(
                chunk.messages, registry if registry else None, seen,
            )
            try:
                result = await _run_one(
                    chunk.messages, model=index_model, vocabulary="default",
                    registry=chunk_registry if chunk_registry else None,
                    message_id_start=chunk.start, cwd=None,
                )
            except Exception:
                typer.echo(f"  chunk {ci+1} index extraction failed", err=True)
                continue
            if result is None:
                typer.echo(f"  chunk {ci+1} no parseable result", err=True)
                continue

            idx.populate_from_extraction(
                result, chunk.messages, run_id=sid,
                message_id_start=chunk.start,
            )
            await resolve_index(idx, model=index_model)

            for sym in result.symbols:
                norm = normalize_name(sym.name)
                if norm not in seen:
                    match = _match_known_symbol(sym.name, registry)
                    if match:
                        aliases = match.get("aliases", [])
                        if not isinstance(aliases, list):
                            aliases = []
                        if sym.name not in aliases:
                            aliases.append(sym.name)
                            match["aliases"] = aliases
                        seen.add(norm)
                    else:
                        seen.add(norm)
                        entry: dict[str, Any] = {"name": sym.name, "kind": sym.kind}
                        if sym.aliases:
                            entry["aliases"] = sym.aliases
                        registry.append(entry)

            stats = idx.stats(sid)

            # --- Auditor: fire on trajectory up to this chunk ---
            trajectory_prefix = messages[:chunk_end]
            context = _index_to_context(idx)

            try:
                audit_result = await run_auditor(
                    trajectory_prefix, model=model,
                    auditor_prompt=auditor_prompt,
                    context_index=context,
                )
                n_auditor_fires += 1
                last_audit_result = audit_result
                _save_auditor_step(exp, sid, chunk_end, audit_result, idx)

                n_surfaces = len(audit_result.surfaces)
                surface_preview = ""
                if audit_result.surfaces:
                    text = audit_result.surfaces[0].reminder_text or ""
                    surface_preview = f"  [{text[:80]}]"

                typer.echo(
                    f"  chunk {ci+1}: msgs 0-{chunk_end}  "
                    f"{stats.symbol_count} syms  "
                    f"{n_surfaces} surfaces{surface_preview}"
                )
            except Exception as exc:
                typer.echo(f"  chunk {ci+1}: index ok, auditor failed: {exc}", err=True)

        # Save final index
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

        typer.echo(f"  total: {stats.symbol_count} syms, {stats.reference_count} refs, {n_auditor_fires} auditor fires")
        if warns:
            parts = ", ".join(f"{v} {k}" for k, v in sorted(warns.items()))
            typer.echo(f"  warnings: {parts}")

        result_dict = {
            "session_id": sid,
            "n_messages": len(messages),
            "n_symbols": stats.symbol_count,
            "n_auditor_fires": n_auditor_fires,
            "n_surfaces": len(last_audit_result.surfaces) if last_audit_result else 0,
        }
        all_results.append(result_dict)
        exp.record_result(result_dict)

    if all_results:
        summary = {
            "n_sessions": len(all_results),
            "avg_surfaces": round(sum(r.get("n_surfaces", 0) for r in all_results) / len(all_results), 1),
        }
        exp.finish(summary=summary)
        typer.echo(f"\nSummary: {json.dumps(summary)}")
    else:
        exp.finish(status="failed", summary={"error": "no sessions processed"})

    typer.echo(f"Experiment: {exp.exp_id} → {exp.output_dir}")


def _save_auditor_step(
    exp: Experiment, sid: str, msg_count: int,
    audit_result: Any, idx: Any,
) -> None:
    """Save one auditor checkpoint."""
    data: dict[str, Any] = {
        "msg_count": msg_count,
        "n_verdicts": len(audit_result.verdicts),
        "n_surfaces": len(audit_result.surfaces),
        "session_ids": audit_result.session_ids,
        "verdicts": [v.to_dict() for v in audit_result.verdicts],
        "surfaces": [v.to_dict() for v in audit_result.surfaces],
    }
    exp.write_session_artifact(sid, "auditor", f"step_{msg_count:04d}.json", data)


register("auditor", AuditorEvalAdapter.description, AuditorEvalAdapter)
