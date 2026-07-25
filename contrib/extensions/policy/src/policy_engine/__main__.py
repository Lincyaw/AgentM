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
``max_injections`` can be backtested against recorded trajectories before it
ships. Stop-checkpoint checks now run through the ``submit`` tool and cannot be
replayed from trajectories recorded before that tool existed.

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

from agentm.core.abi import ProviderConfig

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

# Defaults must mirror PolicyEngineConfig so a replay reproduces what the live
# atom would have done.
CHECKLIST_OPT = typer.Option(DEFAULT_CHECKLIST, "--checklist")
VOCAB_OPT = typer.Option(DEFAULT_VOCAB, "--vocab")
SCHEMA_OPT = typer.Option("harbor_live", "--schema")
MODEL_OPT = typer.Option(None, "--model", help="model profile from config.toml")
MAX_INJECTIONS_OPT = typer.Option(5, "--max-injections")
MAX_STOP_CHECKS_OPT = typer.Option(3, "--max-stop-checks")


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
    max_stop_checks: int,
) -> tuple[list[_Emission], list[_Suppression]]:
    """Replay the mid-work injection path over a recorded session.

    Faithful for continuous checks: those still fire from ``_on_decide`` on the
    same condition, so replaying recorded tags reproduces them exactly.

    Not faithful for stop-checkpoint checks. Those are now raised by the
    ``submit`` tool, and no recorded trajectory contains a submit call — the
    agent had no such tool. What is reported instead is which stop-checkpoint
    items were *matching* the first time the agent ended a turn without tool
    calls: an upper bound on what a first submit would have been rejected on,
    not a prediction of how the session would then have gone.
    """
    engine = TriggerEngine(items=items)
    emissions: list[_Emission] = []
    pending: list[_Suppression] = []
    active: set[str] = set()
    injections = 0
    first_stop_seen = False

    for turn_index, has_tools, tags in turns:
        active.update(tags)
        frozen = frozenset(active)

        if not has_tools:
            if not first_stop_seen:
                first_stop_seen = True
                for _ in range(max_stop_checks):
                    item = engine.next_triggered(stopping=True, active_tags=frozen)
                    if item is None:
                        break
                    pending.append(_Suppression(turn_index, item.item_id))
            continue

        if injections >= max_injections:
            continue
        item = engine.next_triggered(stopping=False, active_tags=frozen)
        if item is not None:
            injections += 1
            emissions.append(_Emission(turn_index, False, item.item_id))

    return emissions, pending


@app.command("replay")
def cmd_replay(
    dsn: str,
    session_id: str,
    schema: str = SCHEMA_OPT,
    checklist: str = CHECKLIST_OPT,
    max_injections: int = MAX_INJECTIONS_OPT,
    max_stop_checks: int = MAX_STOP_CHECKS_OPT,
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
        max_stop_checks=max_stop_checks,
    )

    tags = sorted({t for _, _, ts in turns for t in ts})
    print(f"Session: {session_id}  ({len(turns)} turns)")
    print(f"Active tags: {tags}")

    print(f"\n=== Injections ({len(emissions)}) ===")
    for e in emissions:
        kind = "stop" if e.stopping else "continuous"
        print(f"  t{e.turn_index:<4} [{kind:10s}] {e.item_id}")

    print(f"\n=== Would be raised at first submit ({len(suppressions)}) ===")
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
    interval: int = typer.Option(5, "--interval", help="turns per tagger call"),
) -> None:
    """Replay recorded sessions through the tagger and write annotations.

    Exercises the same ``TaggerConversation`` the atom runs live, against
    trajectories already in the store, so a vocabulary or prompt change can be
    judged on real runs without spending a sandbox.
    """
    import asyncio  # noqa: PLC0415

    asyncio.run(_tag_sessions(dsn, schema, model, force, interval))


async def _build_provider(model: str | None) -> ProviderConfig:
    """A session carrying nothing but the provider, for offline tagging.

    Goes through the same resolver the CLI uses, so the provider comes from
    AGENTM_HOME/config.toml exactly as it would in a live run — no second copy
    of credential lookup lives here.
    """
    from agentm import AgentSession, AgentSessionConfig  # noqa: PLC0415
    from agentm.config import DefaultSessionSpecResolver  # noqa: PLC0415

    session = await AgentSession.create(
        AgentSessionConfig(
            purpose="tagger",
            spec_resolver=DefaultSessionSpecResolver(),
        )
    )
    provider = session.get_provider(model or None)
    if provider is None:
        raise RuntimeError(
            f"provider {model or '<active>'} is not registered; "
            "set default_model in AGENTM_HOME/config.toml or pass --model"
        )
    return provider


async def _tag_sessions(
    dsn: str, schema: str, model: str | None, force: bool, interval: int
) -> None:
    from .tagger import (  # noqa: PLC0415
        TaggerConversation,
        render_turn,
        write_annotation,
    )

    provider = await _build_provider(model)
    print(f"Provider: {provider.name} ({provider.model.id})")

    session_ids = _list_sessions(dsn, schema)
    print(f"Tagging {len(session_ids)} sessions")

    for sid in session_ids:
        source = PgQuerySource(dsn, sid)
        existing = source.query(
            "SELECT COUNT(*) FROM policy.turn_annotations "
            "WHERE session_id = %(session_id)s"
        )
        if existing and existing[0][0] > 0 and not force:
            print(f"  {sid}: already tagged ({existing[0][0]} turns), skip")
            source.close()
            continue

        turns = source.query(
            f"SELECT turn_index, turn_json "  # noqa: S608
            f"FROM {schema}.agentm_trajectory_turns "
            "WHERE session_id = %(session_id)s ORDER BY turn_index"
        )
        conversation = TaggerConversation(
            session_id=sid, stream_fn=provider.stream_fn, model=provider.model
        )

        tagged = 0
        batch: list[str] = []
        for turn_index, turn_json in turns:
            assistant_text, tool_calls, task_text = _turn_for_tagger(turn_json)
            if not assistant_text and not tool_calls:
                continue
            batch.append(
                render_turn(
                    turn_index,
                    assistant_text,
                    tool_calls,
                    task_text=task_text if turn_index == 0 else "",
                )
            )
            if len(batch) < interval:
                continue
            annotation = await conversation.annotate(batch, turn_index=turn_index)
            batch = []
            if annotation is not None:
                write_annotation(source, annotation)
                tagged += 1
        if batch:
            annotation = await conversation.annotate(batch, turn_index=turns[-1][0])
            if annotation is not None:
                write_annotation(source, annotation)
                tagged += 1

        print(
            f"  {sid}: {tagged} batch(es) over {len(turns)} turns, "
            f"{len(conversation.seen_tags)} distinct tags"
        )
        source.close()


def _turn_for_tagger(
    turn_json: object,
) -> tuple[str, list[dict[str, object]], str]:
    """Recover (assistant_text, tool_calls, task_text) from a stored turn.

    Mirrors what the live atom collects from ToolResultEvent, so an offline
    replay renders identically to a live run.
    """
    if not isinstance(turn_json, dict):
        return "", [], ""

    assistant_parts: list[str] = []
    response = turn_json.get("response", {})
    if isinstance(response, dict):
        for block in response.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    assistant_parts.append(str(text))

    tool_calls: list[dict[str, object]] = []
    for tr in turn_json.get("tool_results", []):
        if not isinstance(tr, dict):
            continue
        call = tr.get("call", {})
        result = tr.get("result", {})
        if not isinstance(call, dict) or not isinstance(result, dict):
            continue
        result_text = ""
        for block in result.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                result_text += str(block.get("text", ""))
        arguments = call.get("arguments")
        tool_calls.append(
            {
                "name": str(call.get("name", "?")),
                "arguments": arguments if isinstance(arguments, dict) else {},
                "result_text": result_text[:2000],
                "is_error": bool(result.get("is_error", False)),
            }
        )

    task_text = ""
    trigger = turn_json.get("trigger_metadata")
    if isinstance(trigger, dict):
        meta = trigger.get("meta")
        if isinstance(meta, dict):
            task_text = str(meta.get("text", ""))

    return "\n".join(assistant_parts), tool_calls, task_text


@app.command("evaluate")
def cmd_evaluate(
    dsn: str,
    schema: str = SCHEMA_OPT,
    checklist: str = CHECKLIST_OPT,
    max_injections: int = MAX_INJECTIONS_OPT,
    max_stop_checks: int = MAX_STOP_CHECKS_OPT,
) -> None:
    """Replay all sessions, report per-item fire rates and gate stats."""
    items = load_items(Path(checklist))
    session_ids = _list_sessions(dsn, schema)
    print(f"Evaluating {len(session_ids)} sessions against {len(items)} items")

    fires: dict[str, int] = {item_id: 0 for item_id in items}
    at_submit: dict[str, int] = {}
    total_injects = 0
    total_at_submit = 0
    scored = 0

    for sid in session_ids:
        source = PgQuerySource(dsn, sid)
        turns = _load_turns(source, schema)
        source.close()
        if not turns:
            continue
        scored += 1
        emissions, pending = _simulate(
            turns,
            items,
            max_injections=max_injections,
            max_stop_checks=max_stop_checks,
        )
        for e in emissions:
            total_injects += 1
            fires[e.item_id] = fires.get(e.item_id, 0) + 1
        for s in pending:
            total_at_submit += 1
            at_submit[s.reason] = at_submit.get(s.reason, 0) + 1

    print(f"\n=== Over {scored} sessions ===")
    print(f"  mid-work injections      {total_injects}")
    print(f"  raised at first submit   {total_at_submit}")

    print("\n=== Mid-work item fire rates ===")
    for item_id, count in sorted(fires.items(), key=lambda x: -x[1]):
        if not count:
            continue
        pct = 100.0 * count / max(scored, 1)
        print(f"  {item_id:40s} {count:3d}/{scored} ({pct:.0f}%)")

    print("\n=== Items raised at first submit ===")
    for item_id, count in sorted(at_submit.items(), key=lambda x: -x[1]):
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
    max_stop_checks: int = MAX_STOP_CHECKS_OPT,
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
        max_stop_checks=max_stop_checks,
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
