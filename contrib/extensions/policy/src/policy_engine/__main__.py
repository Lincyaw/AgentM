# code-health: ignore-file[AM025] -- replaying untyped trajectory payloads
"""Policy engine CLI — the evolution loop.

Commands map to steps of the evolution loop:

    compile     ② when_notes → trigger expressions + vocabulary
    deploy      ③ regenerate tagger prompt from vocabulary (automatic)
    replay      ④ replay one session: tagger + signals → emissions
    evaluate    ④ evaluate across all sessions → per-item fitness
    select      ⑤ prune low-fitness items + unused predicates
    evolve      run the full loop: compile → evaluate → select

Step ① (mine) lives in the pattern_miner scenario.
Step ⑥ (diversify) is part of the miner's distill step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger


def cmd_compile(args: argparse.Namespace) -> int:
    """Compile when_notes into trigger expressions."""
    from .compile import (
        build_compiler_prompt,
        load_vocabulary,
        merge_vocabulary,
        parse_compiler_result,
        save_vocabulary,
        update_checklist_triggers,
    )

    checklist_path = Path(args.checklist)
    vocab_path = Path(args.vocab)
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

        if args.dry_run:
            print("(dry run)")
            continue

        result = _call_llm(prompt, manifest="compiler", model=args.model)
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

    if not args.dry_run and compiled:
        save_vocabulary(vocab, vocab_path)
        update_checklist_triggers(checklist_path, compiled)
        print(f"\nVocabulary: {len(vocab)} predicates → {vocab_path}")
        print(f"Checklist: {len(compiled)} triggers updated → {checklist_path}")

    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay one session: evaluate signals + predicate triggers."""
    from .pg_query import PgQuerySource
    from .triggers import TriggerEngine, load_items, load_signals

    items = load_items(Path(args.checklist))
    signals = load_signals(Path(args.signals))
    engine = TriggerEngine(items=items, signals=signals)
    source = PgQuerySource(args.dsn, args.session_id)

    # Load active tags for this session
    tag_rows = source.query(
        "SELECT DISTINCT unnest(tags) FROM policy.turn_annotations "
        "WHERE session_id = %(session_id)s"
    )
    active_tags = frozenset(row[0] for row in tag_rows if row[0])

    print(f"Session: {args.session_id}")
    print(f"Active tags: {sorted(active_tags)}")

    print("\n=== Signals ===")
    for name in signals:
        result = engine.signal_true(source, name)
        evidence = engine.evidence(source, name) if result else ()
        print(f"  {name}: {result}")
        for fact in evidence[:3]:
            print(f"    {fact}")

    print("\n=== Inject emissions ===")
    count = 0
    for stopping in (False, True):
        firing = engine.evaluate_inject(
            source, stopping=stopping, active_tags=active_tags
        )
        while firing is not None:
            count += 1
            print(
                f"  [{firing.item.item_id}] "
                f"signal={firing.item.gate.signal} "
                f"trigger={firing.item.gate.trigger} "
                f"stopping={stopping}"
            )
            for fact in firing.facts[:2]:
                print(f"    {fact}")
            firing = engine.evaluate_inject(
                source, stopping=stopping, active_tags=active_tags
            )
    print(f"  total: {count}")

    print("\n=== Critic items ===")
    critic = engine.open_critic_items(source, active_tags=active_tags)
    print(f"  total: {len(critic)}")
    for item in critic[:10]:
        print(f"  [{item.item_id}] trigger={item.gate.trigger}")

    source.close()
    return 0


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


def cmd_tag(args: argparse.Namespace) -> int:
    """Batch-tag all sessions: run tagger LLM on each turn, write to PG."""
    from .pg_query import PgQuerySource
    from .tagger import _get_tagger_prompt, _parse_tagger_result, write_annotation

    tagger_system = _get_tagger_prompt()
    print(f"Tagger prompt: {len(tagger_system)} chars")

    session_ids = _list_sessions(args.dsn, args.schema)

    print(f"Tagging {len(session_ids)} sessions")

    for sid in session_ids:
        source = PgQuerySource(args.dsn, sid)

        # Check if already tagged
        existing = source.query(
            "SELECT COUNT(*) FROM policy.turn_annotations "
            "WHERE session_id = %(session_id)s"
        )
        if existing and existing[0][0] > 0 and not args.force:
            print(f"  {sid}: already tagged ({existing[0][0]} turns), skip")
            source.close()
            continue

        # Load turns from trajectory
        turns = source.query(
            f"SELECT turn_index, turn_json "  # noqa: S608
            f"FROM {args.schema}.agentm_trajectory_turns "
            "WHERE session_id = %(session_id)s ORDER BY turn_index"
        )

        tagged = 0
        for turn_index, turn_json in turns:
            turn_content = _format_turn_for_tagger(turn_json)
            result = _call_llm(
                turn_content,
                manifest=None,
                model=args.model,
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

    return 0


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


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate signals across all sessions, report per-signal fire rates."""
    from .pg_query import PgQuerySource
    from .triggers import TriggerEngine, load_items, load_signals

    items = load_items(Path(args.checklist))
    signals = load_signals(Path(args.signals))

    session_ids = _list_sessions(args.dsn, args.schema)

    print(f"Evaluating {len(session_ids)} sessions, {len(signals)} signals")

    signal_fires: dict[str, int] = {name: 0 for name in signals}
    for sid in session_ids:
        source = PgQuerySource(args.dsn, sid)
        engine = TriggerEngine(items=items, signals=signals)
        for name in signals:
            if engine.signal_true(source, name):
                signal_fires[name] += 1
        source.close()

    print("\n=== Signal fire rates ===")
    for name, count in sorted(signal_fires.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / max(len(session_ids), 1)
        print(f"  {name:30s} {count:3d}/{len(session_ids)} ({pct:.0f}%)")

    return 0


def cmd_select(args: argparse.Namespace) -> int:
    """Prune low-fitness items and unused predicates."""
    from .compile import load_vocabulary, prune_items, prune_vocabulary, save_vocabulary

    checklist_path = Path(args.checklist)
    vocab_path = Path(args.vocab)

    # For now, fitness is manual: read from a JSON file
    fitness_path = Path(args.fitness) if args.fitness else None
    if fitness_path and fitness_path.is_file():
        fitness = json.loads(fitness_path.read_text())
    else:
        print("No fitness data; use --fitness <path.json> with {item_id: score}")
        return 1

    pruned = prune_items(checklist_path, fitness, threshold=args.threshold)
    print(f"Pruned {len(pruned)} items: {pruned}")

    vocab = load_vocabulary(vocab_path)
    before = len(vocab)
    vocab = prune_vocabulary(vocab, checklist_path)
    save_vocabulary(vocab, vocab_path)
    print(f"Vocabulary: {before} → {len(vocab)} predicates")

    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    """Run the full evolution loop: compile → evaluate → (select is manual)."""
    print("=== Step 1: Compile ===")
    args.dry_run = False
    ret = cmd_compile(args)
    if ret != 0:
        return ret

    print("\n=== Step 2: Evaluate ===")
    ret = cmd_evaluate(args)
    if ret != 0:
        return ret

    print(
        "\n=== Step 3: Select ===\n"
        "Run manually after reviewing evaluate output:\n"
        f"  python -m policy_engine select {args.checklist} {args.vocab} "
        "--fitness <fitness.json>"
    )
    return 0


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


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="policy_engine",
        description="Policy engine evolution loop CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pkg = Path(__file__).parent
    default_checklist = str(pkg / "checklist.yaml")
    default_signals = str(pkg / "signals.yaml")
    default_vocab = str(pkg / "vocabulary.yaml")

    # compile
    p = sub.add_parser("compile", help="compile when_notes → trigger expressions")
    p.add_argument("--checklist", default=default_checklist)
    p.add_argument("--vocab", default=default_vocab)
    p.add_argument("--model", default=None, help="model profile from config.toml")
    p.add_argument("--dry-run", action="store_true")

    # tag
    p = sub.add_parser("tag", help="batch-tag all sessions with tagger LLM")
    p.add_argument("dsn")
    p.add_argument("--schema", default="harbor_live")
    p.add_argument("--model", default=None)
    p.add_argument(
        "--force", action="store_true", help="re-tag already tagged sessions"
    )

    # replay
    p = sub.add_parser("replay", help="replay one session")
    p.add_argument("dsn")
    p.add_argument("session_id")
    p.add_argument("--checklist", default=default_checklist)
    p.add_argument("--signals", default=default_signals)

    # evaluate
    p = sub.add_parser("evaluate", help="evaluate signals across all sessions")
    p.add_argument("dsn")
    p.add_argument("--schema", default="harbor_live")
    p.add_argument("--checklist", default=default_checklist)
    p.add_argument("--signals", default=default_signals)

    # select
    p = sub.add_parser("select", help="prune low-fitness items")
    p.add_argument("--checklist", default=default_checklist)
    p.add_argument("--vocab", default=default_vocab)
    p.add_argument("--fitness", help="JSON file: {item_id: score}")
    p.add_argument("--threshold", type=float, default=0.1)

    # evolve
    p = sub.add_parser("evolve", help="full loop: compile → tag → evaluate")
    p.add_argument("dsn")
    p.add_argument("--schema", default="harbor_live")
    p.add_argument("--checklist", default=default_checklist)
    p.add_argument("--signals", default=default_signals)
    p.add_argument("--vocab", default=default_vocab)
    p.add_argument("--model", default=None)

    args = parser.parse_args()
    handlers = {
        "compile": cmd_compile,
        "tag": cmd_tag,
        "replay": cmd_replay,
        "evaluate": cmd_evaluate,
        "select": cmd_select,
        "evolve": cmd_evolve,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
