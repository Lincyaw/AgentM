# code-health: ignore-file[AM025] -- replaying untyped trajectory payloads
"""``python -m policy_engine`` — the loop, plus two offline maintenance jobs.

The evolution loop lives under ``loop`` (collect, diagnose, abstract, compile,
replay, select, install), one artifact per stage; see ``loop --help``. What
remains at the top level is maintenance that runs against recorded state rather
than a batch:

    tag      replay recorded sessions through the live tagger, writing
             ``policy.turn_annotations`` — for judging a vocabulary or prompt
             change on real runs without spending a sandbox
    prune    retire low-fitness checklist items and the predicates nothing
             references any more

The old ``replay``/``evaluate`` simulators are gone: they modelled the
injection path without preconditions, so they structurally could not evaluate
what ``loop compile`` produces. ``loop check-gates`` asks the same question
against real recorded sessions, and ``loop replay`` measures the real thing.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from agentm.core.abi import ProviderConfig
from policy_engine.shared.pg_query import PgQuerySource

app = typer.Typer(
    name="policy_engine",
    add_completion=False,
    no_args_is_help=True,
    help="Policy engine: the evolution loop, plus offline tagging and pruning.",
)

from .loop.cli import app as _loop_app

app.add_typer(_loop_app, name="loop")

_PKG = Path(__file__).parent / "runtime"
DEFAULT_CHECKLIST = str(_PKG / "checklist.yaml")
DEFAULT_VOCAB = str(_PKG / "vocabulary.yaml")

SCHEMA_OPT = typer.Option("harbor_live", "--schema")


def _fork_point(source: PgQuerySource, schema: str) -> int:
    """The last turn this session inherited from a fork source, or -1.

    ``-1`` rather than ``None`` so callers can compare against it without a
    branch: no fork means nothing was inherited, and every turn index is above
    it.
    """
    rows = source.query(
        f"SELECT meta_json->>'fork_point' FROM {schema}.agentm_trajectory_sessions "
        "WHERE id = %(session_id)s"
    )
    if not rows or rows[0][0] is None:
        return -1
    raw = str(rows[0][0])
    return int(raw) if raw.isdigit() else -1


def _list_sessions(dsn: str, schema: str) -> list[str]:
    """List session IDs from a trajectory schema."""
    from agentm.storage.sql import create_sql_engine

    engine = create_sql_engine(dsn)
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            f"SELECT id FROM {schema}.agentm_trajectory_sessions ORDER BY id"
        )
        session_ids = [str(row[0]) for row in rows]
    engine.dispose()
    return session_ids


@app.command("tag")
def cmd_tag(
    dsn: str,
    schema: str = SCHEMA_OPT,
    model: str | None = typer.Option(
        None, "--model", help="model profile from config.toml"
    ),
    force: bool = typer.Option(False, "--force", help="re-tag already tagged sessions"),
    interval: int = typer.Option(5, "--interval", help="turns per tagger call"),
) -> None:
    """Replay recorded sessions through the tagger and write annotations.

    Exercises the same ``TaggerConversation`` the atom runs live, against
    trajectories already in the store, so a vocabulary or prompt change can be
    judged on real runs without spending a sandbox.
    """
    import asyncio

    asyncio.run(_tag_sessions(dsn, schema, model, force, interval))


async def _build_provider(model: str | None) -> ProviderConfig:
    """A session carrying nothing but the provider, for offline tagging.

    Goes through the same resolver the CLI uses, so the provider comes from
    AGENTM_HOME/config.toml exactly as it would in a live run — no second copy
    of credential lookup lives here.
    """
    from agentm import AgentSession, AgentSessionConfig
    from agentm.config import DefaultSessionSpecResolver

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
    from policy_engine.runtime.record import render_turn
    from policy_engine.runtime.tagger import (
        TaggerConversation,
        tagger_system_prompt,
        write_annotation,
    )

    system = tagger_system_prompt()
    if system is None:
        typer.echo("no vocabulary.yaml: nothing to tag against", err=True)
        raise typer.Exit(1)

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

        # A fork is stored holding a copy of its source's prefix, so those
        # turns are in the store under both ids. Tagging both copies pays the
        # model twice for one stretch of work and writes a second annotation
        # row per turn; the runtime reads tags down the fork chain, so the
        # copy earns nothing either way.
        turns = source.query(
            f"SELECT turn_index, turn_json "
            f"FROM {schema}.agentm_trajectory_turns "
            "WHERE session_id = %(session_id)s "
            "  AND turn_index > %(inherited_through)s ORDER BY turn_index",
            {"inherited_through": _fork_point(source, schema)},
        )
        conversation = TaggerConversation(
            session_id=sid,
            stream_fn=provider.stream_fn,
            model=provider.model,
            system=system,
        )

        tagged = 0
        seen: set[str] = set()
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
                seen.update(annotation.tags)
                tagged += 1
        if batch:
            annotation = await conversation.annotate(batch, turn_index=turns[-1][0])
            if annotation is not None:
                write_annotation(source, annotation)
                seen.update(annotation.tags)
                tagged += 1

        print(
            f"  {sid}: {tagged} batch(es) over {len(turns)} turns, "
            f"{len(seen)} distinct tags"
        )
        source.close()


def _turn_for_tagger(
    turn_json: object,
) -> tuple[str, list[dict[str, object]], str]:
    """Recover (assistant_text, tool_calls, task_text) from a stored turn.

    The dict shape and the rendering both come from ``runtime.record``; what
    lives here is only the walk over raw ``turn_json``, which a live session
    never has to do.
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


@app.command("prune")
def cmd_prune(
    checklist: str = typer.Option(DEFAULT_CHECKLIST, "--checklist"),
    vocab_path_arg: str = typer.Option(DEFAULT_VOCAB, "--vocab"),
    fitness_file: str = typer.Option(
        ..., "--fitness", help="JSON file: {item_id: score}"
    ),
    threshold: float = typer.Option(0.1, "--threshold"),
) -> None:
    """Retire low-fitness items, then the predicates nothing references."""
    from policy_engine.shared.vocabulary import (
        load_vocabulary,
        prune_items,
        prune_vocabulary,
        save_vocabulary,
    )

    checklist_path = Path(checklist)
    vocab_path = Path(vocab_path_arg)

    fitness_path = Path(fitness_file)
    if not fitness_path.is_file():
        print(f"no fitness file at {fitness_path}")
        raise typer.Exit(code=1)
    fitness = json.loads(fitness_path.read_text())

    pruned = prune_items(checklist_path, fitness, threshold=threshold)
    print(f"Pruned {len(pruned)} items: {pruned}")

    vocab = load_vocabulary(vocab_path)
    before = len(vocab)
    vocab = prune_vocabulary(vocab, checklist_path)
    save_vocabulary(vocab, vocab_path)
    print(f"Vocabulary: {before} → {len(vocab)} predicates")


if __name__ == "__main__":
    app()
