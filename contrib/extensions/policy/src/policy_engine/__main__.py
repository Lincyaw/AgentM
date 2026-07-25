# code-health: ignore-file[AM025] -- replaying untyped trajectory payloads
"""Policy engine CLI — the evolution loop.

Commands map to steps of the evolution loop:

    compile     ② when_notes → trigger expressions + vocabulary
    deploy      ③ regenerate tagger prompt from vocabulary (automatic)
    replay      ④ replay one session through the live gating logic
    evaluate    ④ replay all sessions → per-item fire rates + gate stats
    select      ⑤ prune low-fitness items + unused predicates
    evolve      run the full loop: compile → evaluate → select

``replay``/``evaluate`` mirror the runtime's suppression gates, so a change to
``min_work_turns`` / ``max_stop_injections`` can be backtested against recorded
trajectories before it ships.

Step ① (mine) lives in the pattern_miner scenario.
Step ⑥ (diversify) is part of the miner's distill step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import typer
import yaml
from loguru import logger

from .pg_query import PgQuerySource
from .triggers import ChecklistItem, TriggerEngine, load_items

app = typer.Typer(
    name="policy_engine",
    add_completion=False,
    no_args_is_help=True,
    help="Policy engine evolution loop.",
)

_PKG = Path(__file__).parent
DEFAULT_CHECKLIST = str(_PKG / "checklist.yaml")
DEFAULT_VOCAB = str(_PKG / "vocabulary.yaml")

# Suppression gates. Defaults must mirror PolicyEngineConfig so a replay
# reproduces what the live atom would have done.
CHECKLIST_OPT = typer.Option(DEFAULT_CHECKLIST, "--checklist")
VOCAB_OPT = typer.Option(DEFAULT_VOCAB, "--vocab")
SCHEMA_OPT = typer.Option("harbor_live", "--schema")
MODEL_OPT = typer.Option(None, "--model", help="model profile from config.toml")
MAX_INJECTIONS_OPT = typer.Option(5, "--max-injections")
MAX_STOP_OPT = typer.Option(2, "--max-stop-injections")
MIN_WORK_OPT = typer.Option(1, "--min-work-turns")


@app.command("compile")
def cmd_compile(
    checklist: str = CHECKLIST_OPT,
    vocab_path_arg: str = VOCAB_OPT,
    model: str | None = MODEL_OPT,
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Compile when_notes into trigger expressions."""
    from .compile import (
        build_compiler_prompt,
        load_vocabulary,
        merge_vocabulary,
        parse_compiler_result,
        save_vocabulary,
        update_checklist_triggers,
    )

    checklist_path = Path(checklist)
    vocab_path = Path(vocab_path_arg)
    raw = yaml.safe_load(checklist_path.read_text(encoding="utf-8"))
    items = [i for i in raw.get("items", []) if i.get("deliver") != "offline"]
    vocab = load_vocabulary(vocab_path)

    print(f"Compiling {len(items)} items, {len(vocab)} existing predicates")

    compiled: dict[str, str] = {}

    for item in items:
        item_id = item.get("id", "")
        when_note = item.get("when_note", "")
        if not when_note:
            print(f"  {item_id}: no when_note, skipping")
            continue

        prompt = build_compiler_prompt(when_note, vocab)
        print(f"  {item_id}: ", end="", flush=True)

        if dry_run:
            print("(dry run)")
            continue

        result = _call_llm(prompt, manifest="compiler", model=model)
        if result is None:
            print("FAILED")
            continue

        parsed = parse_compiler_result(result)
        if parsed is None:
            print("PARSE ERROR")
            continue

        compiled[item_id] = parsed.trigger_expr
        vocab = merge_vocabulary(vocab, parsed.new_predicates)
        new_str = (
            f" +{len(parsed.new_predicates)} predicates"
            if parsed.new_predicates
            else ""
        )
        print(f"{parsed.trigger_expr}{new_str}")

    if not dry_run and compiled:
        save_vocabulary(vocab, vocab_path)
        update_checklist_triggers(checklist_path, compiled)
        print(f"\nVocabulary: {len(vocab)} predicates → {vocab_path}")
        print(f"Checklist: {len(compiled)} triggers updated → {checklist_path}")


@dataclass(slots=True)
class _Emission:
    turn_index: int
    stopping: bool
    item_id: str


@dataclass(slots=True)
class _Suppression:
    turn_index: int
    reason: str


def _load_turns(
    source: PgQuerySource, schema: str
) -> list[tuple[int, bool, frozenset[str]]]:
    """Return (turn_index, has_tool_calls, tags) per turn, in order."""
    rows = source.query(
        "SELECT t.turn_index, "
        "  EXISTS (SELECT 1 FROM jsonb_array_elements("
        "    coalesce(t.turn_json->'response'->'content', '[]'::jsonb)) c "
        "    WHERE c->>'type' = 'tool_call') AS has_tools, "
        "  coalesce(a.tags, ARRAY[]::text[]) AS tags "
        f"FROM {schema}.agentm_trajectory_turns t "  # noqa: S608
        "LEFT JOIN policy.turn_annotations a "
        "  ON a.session_id = t.session_id AND a.turn_index = t.turn_index "
        "WHERE t.session_id = %(session_id)s ORDER BY t.turn_index"
    )
    return [(int(r[0]), bool(r[1]), frozenset(r[2] or ())) for r in rows]


def _simulate(
    turns: list[tuple[int, bool, frozenset[str]]],
    items: dict[str, ChecklistItem],
    *,
    max_injections: int,
    max_stop_injections: int,
    min_work_turns: int,
) -> tuple[list[_Emission], list[_Suppression]]:
    """Replay the live gating logic over a recorded session.

    Mirrors ``_Runtime._suppressed`` / ``TriggerEngine.next_triggered`` so a
    gate setting can be backtested against real trajectories before shipping.
    """
    engine = TriggerEngine(items=items)
    emissions: list[_Emission] = []
    suppressions: list[_Suppression] = []
    active: set[str] = set()
    injections = 0
    stop_injections = 0
    work_turns = 0
    work_at_inject = 0

    for turn_index, has_tools, tags in turns:
        active.update(tags)
        work_now = work_turns + (1 if has_tools else 0)
        stopping = not has_tools

        reason = ""
        if injections >= max_injections:
            reason = "budget spent"
        elif stopping and stop_injections >= max_stop_injections:
            reason = "stop-inject budget spent"
        elif injections and work_now - work_at_inject < min_work_turns:
            reason = f"only {work_now - work_at_inject} work turn(s) since last inject"

        if not reason:
            item = engine.next_triggered(
                stopping=stopping, active_tags=frozenset(active)
            )
            if item is not None:
                injections += 1
                if stopping:
                    stop_injections += 1
                work_at_inject = work_now
                emissions.append(_Emission(turn_index, stopping, item.item_id))
        elif engine.would_trigger(stopping=stopping, active_tags=frozenset(active)):
            suppressions.append(_Suppression(turn_index, reason))

        work_turns = work_now

    return emissions, suppressions


@app.command("replay")
def cmd_replay(
    dsn: str,
    session_id: str,
    schema: str = SCHEMA_OPT,
    checklist: str = CHECKLIST_OPT,
    max_injections: int = MAX_INJECTIONS_OPT,
    max_stop_injections: int = MAX_STOP_OPT,
    min_work_turns: int = MIN_WORK_OPT,
) -> None:
    """Replay one session through the live gating logic."""
    items = load_items(Path(checklist))
    source = PgQuerySource(dsn, session_id)
    turns = _load_turns(source, schema)
    source.close()

    if not turns:
        print(f"No turns for session {session_id} in schema {schema}")
        raise typer.Exit(code=1)

    emissions, suppressions = _simulate(
        turns,
        items,
        max_injections=max_injections,
        max_stop_injections=max_stop_injections,
        min_work_turns=min_work_turns,
    )

    tags = sorted({t for _, _, ts in turns for t in ts})
    print(f"Session: {session_id}  ({len(turns)} turns)")
    print(f"Active tags: {tags}")

    print(f"\n=== Injections ({len(emissions)}) ===")
    for e in emissions:
        kind = "stop" if e.stopping else "continuous"
        print(f"  t{e.turn_index:<4} [{kind:10s}] {e.item_id}")

    print(f"\n=== Suppressed ({len(suppressions)}) ===")
    for s in suppressions:
        print(f"  t{s.turn_index:<4} {s.reason}")


def _list_sessions(dsn: str, schema: str) -> list[str]:
    """List session IDs from a trajectory schema."""
    from agentm.storage.sql import create_sql_engine  # noqa: PLC0415

    engine = create_sql_engine(dsn)
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            f"SELECT id FROM {schema}.agentm_trajectory_sessions "  # noqa: S608
            "ORDER BY id"
        )
        session_ids = [str(row[0]) for row in rows]
    engine.dispose()
    return session_ids


@app.command("tag")
def cmd_tag(
    dsn: str,
    schema: str = SCHEMA_OPT,
    model: str | None = MODEL_OPT,
    force: bool = typer.Option(False, "--force", help="re-tag already tagged sessions"),
) -> None:
    """Batch-tag all sessions: run tagger LLM on each turn, write to PG."""
    from .tagger import _get_tagger_prompt, _parse_tagger_result, write_annotation

    tagger_system = _get_tagger_prompt()
    print(f"Tagger prompt: {len(tagger_system)} chars")

    session_ids = _list_sessions(dsn, schema)

    print(f"Tagging {len(session_ids)} sessions")

    for sid in session_ids:
        source = PgQuerySource(dsn, sid)

        # Check if already tagged
        existing = source.query(
            "SELECT COUNT(*) FROM policy.turn_annotations "
            "WHERE session_id = %(session_id)s"
        )
        if existing and existing[0][0] > 0 and not force:
            print(f"  {sid}: already tagged ({existing[0][0]} turns), skip")
            source.close()
            continue

        # Load turns from trajectory
        turns = source.query(
            f"SELECT turn_index, turn_json "  # noqa: S608
            f"FROM {schema}.agentm_trajectory_turns "
            "WHERE session_id = %(session_id)s ORDER BY turn_index"
        )

        tagged = 0
        for turn_index, turn_json in turns:
            turn_content = _format_turn_for_tagger(turn_json)
            result = _call_llm(
                turn_content,
                manifest=None,
                model=model,
                system_override=tagger_system,
            )
            if result is None:
                continue
            annotation = _parse_tagger_result(
                result, session_id=sid, turn_index=turn_index
            )
            if annotation is not None:
                write_annotation(source, annotation)
                tagged += 1

        print(f"  {sid}: tagged {tagged}/{len(turns)} turns")
        source.close()


def _format_turn_for_tagger(turn_json: object) -> str:
    """Format one trajectory turn for the tagger (offline batch mode)."""
    import json as _json  # noqa: PLC0415

    if not isinstance(turn_json, dict):
        return ""
    parts: list[str] = []

    response = turn_json.get("response", {})
    if isinstance(response, dict):
        for block in response.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(f"Assistant reasoning:\n{text[:2000]}")

    for tr in turn_json.get("tool_results", []):
        if not isinstance(tr, dict):
            continue
        call = tr.get("call", {})
        result = tr.get("result", {})
        if not isinstance(call, dict) or not isinstance(result, dict):
            continue
        name = call.get("name", "?")
        args = call.get("arguments", {})
        result_text = ""
        for block in result.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                result_text += block.get("text", "")[:600]
        is_error = result.get("is_error", False)
        parts.append(f"\nTool call: {name}")
        parts.append(f"  Args: {_json.dumps(args, default=str)[:400]}")
        if result_text:
            parts.append(
                f"  Result ({'ERROR' if is_error else 'ok'}): {result_text[:600]}"
            )

    return "\n".join(parts)


@app.command("evaluate")
def cmd_evaluate(
    dsn: str,
    schema: str = SCHEMA_OPT,
    checklist: str = CHECKLIST_OPT,
    max_injections: int = MAX_INJECTIONS_OPT,
    max_stop_injections: int = MAX_STOP_OPT,
    min_work_turns: int = MIN_WORK_OPT,
) -> None:
    """Replay all sessions, report per-item fire rates and gate stats."""
    items = load_items(Path(checklist))
    session_ids = _list_sessions(dsn, schema)
    print(f"Evaluating {len(session_ids)} sessions against {len(items)} items")

    fires: dict[str, int] = {item_id: 0 for item_id in items}
    total_injects = 0
    stop_injects = 0
    total_suppressed = 0
    suppressed_by: dict[str, int] = {}
    scored = 0

    for sid in session_ids:
        source = PgQuerySource(dsn, sid)
        turns = _load_turns(source, schema)
        source.close()
        if not turns:
            continue
        scored += 1
        emissions, suppressions = _simulate(
            turns,
            items,
            max_injections=max_injections,
            max_stop_injections=max_stop_injections,
            min_work_turns=min_work_turns,
        )
        for e in emissions:
            total_injects += 1
            if e.stopping:
                stop_injects += 1
            fires[e.item_id] = fires.get(e.item_id, 0) + 1
        for s in suppressions:
            total_suppressed += 1
            suppressed_by[s.reason.split(" since")[0]] = (
                suppressed_by.get(s.reason.split(" since")[0], 0) + 1
            )

    print(f"\n=== Injections over {scored} sessions ===")
    print(f"  total       {total_injects}")
    print(f"  stop        {stop_injects}")
    print(f"  continuous  {total_injects - stop_injects}")
    print(f"  suppressed  {total_suppressed}")
    for reason, count in sorted(suppressed_by.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")

    print("\n=== Item fire rates ===")
    for item_id, count in sorted(fires.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / max(scored, 1)
        print(f"  {item_id:40s} {count:3d}/{scored} ({pct:.0f}%)")


@app.command("select")
def cmd_select(
    checklist: str = CHECKLIST_OPT,
    vocab_path_arg: str = VOCAB_OPT,
    fitness_file: str | None = typer.Option(
        None, "--fitness", help="JSON file: {item_id: score}"
    ),
    threshold: float = typer.Option(0.1, "--threshold"),
) -> None:
    """Prune low-fitness items and unused predicates."""
    from .compile import load_vocabulary, prune_items, prune_vocabulary, save_vocabulary

    checklist_path = Path(checklist)
    vocab_path = Path(vocab_path_arg)

    # For now, fitness is manual: read from a JSON file
    fitness_path = Path(fitness_file) if fitness_file else None
    if fitness_path is None or not fitness_path.is_file():
        print("No fitness data; use --fitness <path.json> with {item_id: score}")
        raise typer.Exit(code=1)
    fitness = json.loads(fitness_path.read_text())

    pruned = prune_items(checklist_path, fitness, threshold=threshold)
    print(f"Pruned {len(pruned)} items: {pruned}")

    vocab = load_vocabulary(vocab_path)
    before = len(vocab)
    vocab = prune_vocabulary(vocab, checklist_path)
    save_vocabulary(vocab, vocab_path)
    print(f"Vocabulary: {before} → {len(vocab)} predicates")


@app.command("evolve")
def cmd_evolve(
    dsn: str,
    schema: str = SCHEMA_OPT,
    checklist: str = CHECKLIST_OPT,
    vocab_path_arg: str = VOCAB_OPT,
    model: str | None = MODEL_OPT,
    max_injections: int = MAX_INJECTIONS_OPT,
    max_stop_injections: int = MAX_STOP_OPT,
    min_work_turns: int = MIN_WORK_OPT,
) -> None:
    """Run the full evolution loop: compile → evaluate → (select is manual)."""
    print("=== Step 1: Compile ===")
    cmd_compile(
        checklist=checklist,
        vocab_path_arg=vocab_path_arg,
        model=model,
        dry_run=False,
    )

    print("\n=== Step 2: Evaluate ===")
    cmd_evaluate(
        dsn=dsn,
        schema=schema,
        checklist=checklist,
        max_injections=max_injections,
        max_stop_injections=max_stop_injections,
        min_work_turns=min_work_turns,
    )

    print(
        "\n=== Step 3: Select ===\n"
        "Run manually after reviewing evaluate output:\n"
        f"  python -m policy_engine select --checklist {checklist} "
        f"--vocab {vocab_path_arg} --fitness <fitness.json>"
    )


# -- LLM calling (offline, not through SDK spawn) -----------------------------


def _call_llm(
    prompt: str,
    *,
    manifest: str | None,
    model: str | None,
    system_override: str | None = None,
) -> str | None:
    try:
        import tomllib  # noqa: PLC0415

        from openai import OpenAI  # noqa: PLC0415

        if system_override:
            system = system_override
        elif manifest:
            manifest_path = Path(__file__).parent / "agents" / f"{manifest}.yaml"
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            system = str(raw.get("system", ""))
        else:
            system = ""

        config = tomllib.loads(Path.home().joinpath(".agentm/config.toml").read_text())
        model_name = model or "litellm-dsv4flash"
        mc = config["models"][model_name]
        client = OpenAI(api_key=mc["api_key"], base_url=mc["base_url"])

        resp = client.chat.completions.create(
            model=mc["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0,
        )
        return resp.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM call failed: {}", exc)
        return None


if __name__ == "__main__":
    app()
