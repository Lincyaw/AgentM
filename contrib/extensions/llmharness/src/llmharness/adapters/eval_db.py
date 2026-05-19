"""Adapter: rcabench-style eval.db trajectories → llmharness event graph.

Each row in ``evaluation_data.trajectories`` is a langgraph-style event
stream ``{trajectory_file, events: [...]}``. We fold those events into
the extractor-friendly ``new_turns`` shape (matches
``_serialize_message_for_extractor`` in ``adapters/agentm.py``) and run
the v3.1 extractor as a top-level session via
:func:`llmharness.replay.engine.run_phase_standalone`.

Output is a JSONL of :class:`llmharness.replay.record.ReplayRecord` —
one record per firing per row — so the result can later be re-run via
``llmharness-replay extractor`` for A/B prompt/model bisection.

Boundary: this is a host-side driver (not a §11 atom). It is allowed to
``import agentm.core.runtime`` indirectly via ``replay.engine``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from ..audit._session_helpers import bind_extractor_state
from ..audit.extractor import (
    SUBMIT_EVENTS_TOOL_NAME,
    ExtractionState,
    RawExtractorOutput,
    compose_extractor_extensions,
)
from ..audit.extractor.prompt import load_extractor_prompt
from ..replay.engine import run_phase_standalone
from ..replay.record import ReplayRecord, now_ns, write_record
from ..schema import Event as SchemaEvent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# .env autoload (mirror agentm/cli.py:30) so AGENTM_PROVIDER / AGENTM_MODEL /
# OPENAI_* are picked up when this CLI is invoked from any subdir of the
# AgentM workspace.
# --------------------------------------------------------------------------


def _autoload_dotenv() -> None:
    if os.environ.get("AGENTM_SKIP_DOTENV"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    cur = Path.cwd().resolve()
    for candidate in (cur, *cur.parents):
        env_path = candidate / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
            return


# --------------------------------------------------------------------------
# Fold: langgraph events -> extractor turns
# --------------------------------------------------------------------------


@dataclass
class FoldedTrajectory:
    """Result of folding one row's event stream into extractor inputs."""

    turns: list[dict[str, Any]]
    turn_texts: dict[int, str]


def fold_events(events: list[dict[str, Any]]) -> FoldedTrajectory:
    """Collapse a langgraph event stream into AgentM-shaped turns.

    Output shape matches :func:`_serialize_message_for_extractor`:
    ``{"index": int, "role": "user"|"assistant", "content": [...]}``
    where each content block is one of ``{type:"text"|"thinking"}``,
    ``{type:"tool_call", id, name, arguments}``, or
    ``{type:"tool_result", tool_call_id, content:[{type:"text",text:...}], is_error}``.
    """
    turns: list[dict[str, Any]] = []
    turn_texts: dict[int, str] = {}

    def _emit(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        idx = len(turns)
        turn = {"index": idx, "role": role, "content": blocks}
        turns.append(turn)
        turn_texts[idx] = _render_blocks(blocks)

    # Initial user turn: pull the last `user`/`human` message from the
    # first llm_start. System prompts are dropped — the extractor has
    # its own system prompt and doesn't need ours.
    initial_user: str | None = None
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("event_type") != "llm_start":
            continue
        msgs = ((ev.get("data") or {}).get("messages")) or []
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            mtype = (m.get("type") or m.get("role") or "").lower()
            if mtype in {"user", "human"}:
                content = m.get("content")
                if isinstance(content, str) and content.strip():
                    initial_user = content
                    break
        if initial_user is not None:
            break
    if initial_user is not None:
        _emit("user", [{"type": "text", "text": initial_user}])

    # Walk the stream; group consecutive tool_results between llm_end
    # boundaries into a single user turn.
    pending_results: list[dict[str, Any]] = []

    def _flush_results() -> None:
        if not pending_results:
            return
        _emit("user", pending_results.copy())
        pending_results.clear()

    for ev in events:
        if not isinstance(ev, dict):
            continue
        et = ev.get("event_type")
        data = ev.get("data") or {}

        if et == "llm_end":
            _flush_results()
            blocks: list[dict[str, Any]] = []
            text = data.get("content")
            if isinstance(text, str) and text.strip():
                blocks.append({"type": "text", "text": text})
            for tc in data.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name")
                args = tc.get("args") or tc.get("arguments") or {}
                if not isinstance(name, str):
                    continue
                blocks.append(
                    {
                        "type": "tool_call",
                        "id": tc.get("id"),
                        "name": name,
                        "arguments": dict(args) if isinstance(args, dict) else {},
                    }
                )
            _emit("assistant", blocks)

        elif et == "tool_result":
            result = data.get("result")
            if not isinstance(result, str):
                with contextlib.suppress(TypeError, ValueError):
                    result = json.dumps(result, ensure_ascii=False, default=str)
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_call_id": data.get("tool_call_id"),
                    "content": [{"type": "text", "text": result or ""}],
                    "is_error": bool(data.get("is_error", False)),
                }
            )

        elif et == "result":
            _flush_results()
            final = data.get("final_output")
            if isinstance(final, str) and final.strip():
                # langgraph trajectories often carry the final answer
                # both on the trailing ``llm_end.content`` and on
                # ``result.final_output``. Dedupe when the previous
                # assistant turn already holds the same text — otherwise
                # the extractor emits two near-identical ``concl`` events.
                prev_text: str | None = None
                if turns and turns[-1]["role"] == "assistant":
                    last_blocks = turns[-1]["content"]
                    if len(last_blocks) == 1 and last_blocks[0].get("type") == "text":
                        prev_text = last_blocks[0].get("text")
                if prev_text != final:
                    _emit("assistant", [{"type": "text", "text": final}])

        # tool_call / llm_start / run_start / run_complete / _meta are
        # either redundant (tool_call duplicates llm_end.tool_calls) or
        # control-only.

    _flush_results()
    return FoldedTrajectory(turns=turns, turn_texts=turn_texts)


def _render_blocks(blocks: list[dict[str, Any]]) -> str:
    """Flatten one turn's blocks into a witness-friendly text string.

    Mirrors ``_render_message_text`` in the live adapter: text blocks
    plus the JSON dump of tool_call arguments plus inner text of
    tool_result content. Whitespace boundaries don't matter — the
    witness layer normalises before substring matching.
    """
    parts: list[str] = []
    for b in blocks:
        t = b.get("type")
        if t in {"text", "thinking"}:
            text = b.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif t == "tool_call":
            # Include both name and arguments so witness checks against
            # ``cited_entities`` find the tool name in the rendered text.
            # Live ``_render_message_text`` dumps only arguments, but the
            # extractor frequently cites the tool name as the entity, so
            # we include it explicitly here.
            name = b.get("name")
            args = b.get("arguments")
            if isinstance(name, str):
                parts.append(name)
            if isinstance(args, dict):
                with contextlib.suppress(TypeError, ValueError):
                    parts.append(json.dumps(args, ensure_ascii=False, default=str))
        elif t == "tool_result":
            for sub in b.get("content") or []:
                if isinstance(sub, dict):
                    text = sub.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    return " ".join(parts)


# --------------------------------------------------------------------------
# Extract
# --------------------------------------------------------------------------


async def extract_trajectory(
    folded: FoldedTrajectory,
    *,
    cwd: str,
    provider: tuple[str, dict[str, Any]] | None,
    base_prompt: str,
    window: int,
    root_session_id: str,
    sink_path: Path,
    compose_kwargs_for_record: dict[str, Any],
    witness_retry_budget: int = 1,
    graph_accumulator: list[dict[str, Any]] | None = None,
) -> list[ReplayRecord]:
    """Run extractor over a folded trajectory; emit one ReplayRecord per firing.

    ``window <= 0`` means single-shot (whole trajectory in one firing).
    """
    turns = folded.turns
    turn_texts = folded.turn_texts
    if not turns:
        return []

    base_extensions = compose_extractor_extensions(base_prompt=base_prompt)
    records: list[ReplayRecord] = []
    recent_graph_events: list[SchemaEvent] = []

    step = window if window and window > 0 else len(turns)
    lo = 0
    while lo < len(turns):
        hi = min(lo + step, len(turns))
        window_turns = turns[lo:hi]

        state = ExtractionState()
        for t in window_turns:
            i = int(t["index"])
            state.turn_texts[i] = turn_texts.get(i, "")
        for ev in recent_graph_events:
            for src in ev.source_turns:
                state.turn_texts.setdefault(src, turn_texts.get(src, ""))
        state.recent_graph = tuple(recent_graph_events)

        recent_payload: list[dict[str, Any]] = []
        for ev in recent_graph_events:
            entry = ev.to_dict()
            entry["source_turn_texts"] = [
                turn_texts.get(src, "") for src in ev.source_turns
            ]
            recent_payload.append(entry)

        payload = {"new_turns": window_turns, "recent_graph": recent_payload}
        turn_window_json = json.dumps(
            window_turns, ensure_ascii=False, default=str
        )
        extensions = bind_extractor_state(
            base_extensions,
            state=state,
            turn_window_json=turn_window_json,
            witness_retry_budget=witness_retry_budget,
        )

        ts_ns = now_ns()
        t0 = time.monotonic()
        try:
            phase_result = await run_phase_standalone(
                cwd=cwd,
                extensions=extensions,
                provider=provider,
                payload=payload,
                terminal_tool=SUBMIT_EVENTS_TOOL_NAME,
                purpose="eval_db_extractor",
            )
            status = phase_result.status
            error = phase_result.error
            latency_ms = phase_result.latency_ms
        except Exception as exc:
            status = "spawn_error"
            error = str(exc)
            latency_ms = int((time.monotonic() - t0) * 1000)

        output_payload: dict[str, Any] | None
        if status == "ok":
            out = RawExtractorOutput.from_state(state)
            payload_dict: dict[str, Any] = {
                "events": [e.to_dict() for e in out.events],
                "edges": [ed.to_dict() for ed in out.edges],
                "dropped_edges": list(out.dropped_edges),
            }
            output_payload = payload_dict
            recent_graph_events.extend(out.events)
            if graph_accumulator is not None:
                graph_accumulator.append(
                    {
                        "window_lo": lo,
                        "window_hi_inclusive": hi - 1,
                        **payload_dict,
                    }
                )
        else:
            output_payload = None

        record = ReplayRecord(
            phase="extractor",
            turn_index=hi - 1,
            root_session_id=root_session_id,
            ts_ns=ts_ns,
            compose_kwargs=compose_kwargs_for_record,
            payload=payload,
            provider=[provider[0], dict(provider[1])] if provider else None,
            output=output_payload,
            status=status,
            error=error,
            latency_ms=latency_ms,
            extras={
                "turn_window_json": turn_window_json,
                "turn_texts": {str(k): v for k, v in state.turn_texts.items()},
                "window_lo": lo,
                "window_hi_inclusive": hi - 1,
            },
        )
        records.append(record)
        write_record(sink_path, record)

        if status != "ok":
            # Hold the cursor on failure — same semantics as live adapter.
            logger.warning(
                "extractor firing failed: status=%s error=%s", status, error
            )
            break
        lo = hi

    return records


# --------------------------------------------------------------------------
# DB I/O
# --------------------------------------------------------------------------


@dataclass
class DBRow:
    row_id: int
    exp_id: str
    dataset: str
    dataset_index: int | None
    agent_type: str | None
    model_name: str | None
    stage: str
    trajectories_json: str


def iter_rows(
    db_path: Path,
    *,
    exp_id: str | None,
    where: str | None,
    limit: int | None,
    ids: list[int] | None,
) -> Iterator[DBRow]:
    sql = (
        "SELECT id, exp_id, dataset, dataset_index, agent_type, model_name, "
        "stage, trajectories FROM evaluation_data WHERE trajectories IS NOT NULL"
    )
    params: list[Any] = []
    if exp_id is not None:
        sql += " AND exp_id = ?"
        params.append(exp_id)
    if ids:
        sql += " AND id IN (" + ",".join("?" * len(ids)) + ")"
        params.extend(ids)
    if where:
        sql += " AND (" + where + ")"
    sql += " ORDER BY id"
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(sql, params):
            yield DBRow(
                row_id=int(row["id"]),
                exp_id=str(row["exp_id"]),
                dataset=str(row["dataset"]),
                dataset_index=row["dataset_index"],
                agent_type=row["agent_type"],
                model_name=row["model_name"],
                stage=str(row["stage"]),
                trajectories_json=str(row["trajectories"]),
            )
    finally:
        conn.close()


def _events_from_row(row: DBRow) -> list[dict[str, Any]]:
    try:
        data = json.loads(row.trajectories_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"row {row.row_id}: trajectories not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        events = data.get("events")
        if isinstance(events, list):
            return events
    if isinstance(data, list):
        return data
    raise ValueError(
        f"row {row.row_id}: unexpected trajectories shape: {type(data).__name__}"
    )


def _root_session_id_for(row: DBRow) -> str:
    """Deterministic 32-hex id per row so reruns overwrite cleanly."""
    blob = f"evaldb|{row.exp_id}|{row.row_id}".encode()
    return hashlib.sha256(blob).hexdigest()[:32]


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


def build_provider(
    provider_id: str | None,
    model: str | None,
) -> tuple[str, dict[str, Any]]:
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY

    pid = provider_id or os.environ.get("AGENTM_PROVIDER") or "openai"
    mdl = model or os.environ.get("AGENTM_MODEL")
    config: dict[str, Any] = {}
    if mdl:
        config["model"] = mdl
    module, values = DEFAULT_PROVIDER_REGISTRY.build(pid, config)
    return module, values


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Fold rcabench-style eval.db trajectories into llmharness event "
        "graphs by running the v3.1 extractor over each row."
    ),
)


@app.command()
def extract(
    db: Annotated[Path, typer.Option("--db", help="Path to eval.db (read-only).")],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help=(
                "Output directory. Each row writes to "
                "<out-dir>/<exp_id>/<row_id>/ with records.jsonl, graph.json, "
                "and meta.json."
            ),
        ),
    ],
    exp_id: Annotated[
        str | None, typer.Option("--exp-id", help="Filter evaluation_data.exp_id.")
    ] = None,
    where: Annotated[
        str | None,
        typer.Option(
            "--where",
            help="Extra SQL fragment AND'd into the WHERE clause "
            "(e.g. \"stage='judged' AND model_name LIKE 'claude%'\").",
        ),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Cap row count; 0 = no cap.")
    ] = 0,
    row_ids: Annotated[
        list[int],
        typer.Option("--id", help="Specific evaluation_data.id rows (repeatable)."),
    ] = [],  # noqa: B006 — typer needs mutable default
    window: Annotated[
        int,
        typer.Option(
            "--window", help="Turn-window size; 0 = single-shot whole trajectory."
        ),
    ] = 0,
    provider_id: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="Provider id (default: $AGENTM_PROVIDER, else 'openai').",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model id (default: $AGENTM_MODEL)."),
    ] = None,
    cwd: Annotated[
        Path | None,
        typer.Option(
            "--cwd", help="Working dir for the spawned AgentSession (defaults to $PWD)."
        ),
    ] = None,
    prompt_name: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            help=(
                "Named extractor prompt variant (file under "
                "audit/extractor/prompts/) or absolute path. Defaults to v3.1."
            ),
        ),
    ] = None,
    prompt_override: Annotated[
        Path | None,
        typer.Option(
            "--prompt-override",
            help="Literal prompt file; takes precedence over --prompt.",
        ),
    ] = None,
    inspect_only: Annotated[
        bool,
        typer.Option(
            "--inspect",
            help="Print folded turns for each row and exit without calling LLM.",
        ),
    ] = False,
    witness_retry: Annotated[
        int,
        typer.Option(
            "--witness-retry",
            help=(
                "How many times to bounce a submission back to the LLM when "
                "individual edges fail witness; 0 disables (V3.1 single-shot). "
                "Each retry costs one extra LLM call per affected firing."
            ),
        ),
    ] = 1,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            "-j",
            help="Max trajectories processed in parallel (each row independent).",
        ),
    ] = 1,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Extract event graphs from one or more eval.db rows."""
    _autoload_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not db.is_file():
        typer.echo(f"db file not found: {db}", err=True)
        raise typer.Exit(2)

    cwd_abs = str((cwd or Path.cwd()).resolve())
    out_root = out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if prompt_override is not None:
        base_prompt = prompt_override.read_text(encoding="utf-8")
    elif prompt_name:
        base_prompt = load_extractor_prompt(prompt_name)
    else:
        base_prompt = load_extractor_prompt()
    compose_kwargs_for_record: dict[str, Any] = {"base_prompt": base_prompt}

    if inspect_only:
        provider_tuple: tuple[str, dict[str, Any]] | None = None
    else:
        try:
            provider_tuple = build_provider(provider_id, model)
        except KeyError as exc:
            typer.echo(f"provider resolution failed: {exc}", err=True)
            raise typer.Exit(2) from exc
        # Don't echo api_key into logs even when verbose.
        safe = {k: ("***" if "key" in k.lower() else v) for k, v in provider_tuple[1].items()}
        logger.info("provider: %s %s", provider_tuple[0], safe)

    rows = list(
        iter_rows(
            db,
            exp_id=exp_id,
            where=where,
            limit=limit if limit > 0 else None,
            ids=row_ids or None,
        )
    )
    if not rows:
        typer.echo("no rows matched the filter", err=True)
        raise typer.Exit(1)

    logger.info(
        "processing %d rows -> %s (concurrency=%d)",
        len(rows),
        out_root,
        max(concurrency, 1),
    )

    sem = asyncio.Semaphore(max(concurrency, 1))

    async def _process_row(row: DBRow) -> tuple[int, int]:
        async with sem:
            try:
                raw_events = _events_from_row(row)
            except ValueError as exc:
                logger.warning("skip row %d: %s", row.row_id, exc)
                return 0, 0
            folded = fold_events(raw_events)
            logger.info(
                "row=%d exp=%s agent=%s model=%s turns=%d",
                row.row_id,
                row.exp_id,
                row.agent_type,
                row.model_name,
                len(folded.turns),
            )
            if inspect_only:
                typer.echo(
                    json.dumps(
                        {
                            "row_id": row.row_id,
                            "exp_id": row.exp_id,
                            "turns": folded.turns,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                )
                return 0, 0
            if not folded.turns:
                logger.warning("row %d folded to 0 turns; skipping", row.row_id)
                return 0, 0

            row_dir = out_root / row.exp_id / str(row.row_id)
            row_dir.mkdir(parents=True, exist_ok=True)
            records_path = row_dir / "records.jsonl"
            with records_path.open("w", encoding="utf-8"):
                pass  # truncate for idempotent reruns

            assert provider_tuple is not None
            graph_firings: list[dict[str, Any]] = []
            t0 = time.monotonic()
            recs = await extract_trajectory(
                folded,
                cwd=cwd_abs,
                provider=provider_tuple,
                base_prompt=base_prompt,
                window=window,
                root_session_id=_root_session_id_for(row),
                sink_path=records_path,
                compose_kwargs_for_record=compose_kwargs_for_record,
                witness_retry_budget=witness_retry,
                graph_accumulator=graph_firings,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            terminal_ok = bool(recs) and recs[-1].status == "ok"
            total_events = sum(len(f["events"]) for f in graph_firings)
            total_edges = sum(len(f["edges"]) for f in graph_firings)
            total_dropped = sum(len(f["dropped_edges"]) for f in graph_firings)

            graph_path = row_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "root_session_id": _root_session_id_for(row),
                        "row_id": row.row_id,
                        "exp_id": row.exp_id,
                        "window": window,
                        "firings": graph_firings,
                        "totals": {
                            "events": total_events,
                            "edges": total_edges,
                            "dropped_edges": total_dropped,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

            meta_path = row_dir / "meta.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "row_id": row.row_id,
                        "exp_id": row.exp_id,
                        "dataset": row.dataset,
                        "dataset_index": row.dataset_index,
                        "agent_type": row.agent_type,
                        "model_name": row.model_name,
                        "stage": row.stage,
                        "n_turns": len(folded.turns),
                        "n_firings": len(recs),
                        "terminal_ok": terminal_ok,
                        "elapsed_ms": elapsed_ms,
                        "provider": [
                            provider_tuple[0],
                            {
                                k: ("***" if "key" in k.lower() else v)
                                for k, v in provider_tuple[1].items()
                            },
                        ],
                        "witness_retry_budget": witness_retry,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

            return (1 if terminal_ok else 0), len(recs)

    async def _run() -> tuple[int, int]:
        results = await asyncio.gather(
            *[_process_row(r) for r in rows], return_exceptions=True
        )
        ok = 0
        fired = 0
        for r in results:
            if isinstance(r, BaseException):
                logger.error("row task failed: %s", r)
                continue
            ok += r[0]
            fired += r[1]
        return ok, fired

    ok, fired = asyncio.run(_run())
    typer.echo(
        f"done: rows_ok={ok}/{len(rows)} firings={fired} out_dir={out_root}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
