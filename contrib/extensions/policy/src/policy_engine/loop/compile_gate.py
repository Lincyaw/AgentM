"""Stage 4: ``Candidate`` -> ``CompiledCandidate``.

The candidate arrives with its firing conditions in plain language. This stage
turns them into the two things the runtime can evaluate: a boolean expression
over tagger predicates, and SQL over the session's recorded actions.

Any precondition it produces is executed against real recorded sessions before
being accepted. That check is cheap and it caught a real class of mistake
immediately: a word-boundary pattern for "test" matches neither ``pytest`` nor
``--group testing``, so an item gated on it would have been silently inert. A
precondition that no recorded session satisfies is reported, not shipped.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from agentm.core.abi import JsonValue
from policy_engine.shared import facts
from policy_engine.shared.pg_query import PgQuerySource
from policy_engine.shared.vocabulary import PredicateDef, load_vocabulary

from .contracts import Candidate, CompiledCandidate, GatePayload
from .runner import AgentRun, ResultTool, fan_out, load_stage_manifest

RESULT_TOOL = ResultTool(
    name="submit_gate",
    description="Record the firing condition for this item. Call once.",
    payload=GatePayload,
)


def build_prompt(candidate: Candidate, vocabulary: Mapping[str, PredicateDef]) -> str:
    vocab = (
        "\n".join(f"- {name}: {p.definition}" for name, p in sorted(vocabulary.items()))
        or "(empty -- propose what you need)"
    )
    return "\n".join(
        [
            "## The check this gates",
            candidate.check,
            "",
            "## What it observes when it runs",
            candidate.observation or "(not recorded)",
            "",
            "## When it should fire, in plain language",
            candidate.when_note or "(unconditional)",
            "",
            "## What must be observably true first, in plain language",
            candidate.precondition_note or "(nothing)",
            "",
            f"## Checkpoint\n{candidate.checkpoint}",
            "",
            "## Predicate vocabulary",
            vocab,
            "",
            "## Fact tables available to the precondition",
            facts.documentation(),
        ]
    )


async def compile_one(
    candidate: Candidate,
    vocabulary: Mapping[str, PredicateDef],
    *,
    provider: str = "",
    user_config: str = "",
    agent: AgentRun | None = None,
) -> CompiledCandidate | None:
    run = agent or AgentRun(
        manifest=load_stage_manifest("compiler"),
        result_tool=RESULT_TOOL,
        provider=provider,
        user_config=user_config,
    )
    payload = await run.run(build_prompt(candidate, vocabulary))
    if payload is None:
        return None
    return _to_compiled(payload, candidate)


def _to_compiled(
    payload: Mapping[str, JsonValue], candidate: Candidate
) -> CompiledCandidate:
    enriched: dict[str, JsonValue] = dict(payload)
    enriched["candidate"] = dict(candidate.to_json())
    enriched["item_id"] = f"{candidate.dimension or 'item'}_{uuid.uuid4().hex[:6]}"
    return CompiledCandidate.from_json(enriched)


async def compile_candidates(
    candidates: Sequence[Candidate],
    *,
    vocabulary_path: Path,
    provider: str = "",
    user_config: str = "",
    concurrency: int = 4,
) -> list[CompiledCandidate]:
    vocabulary = load_vocabulary(vocabulary_path)

    async def one(candidate: Candidate) -> CompiledCandidate | None:
        return await compile_one(
            candidate, vocabulary, provider=provider, user_config=user_config
        )

    return await fan_out(
        candidates, one, concurrency=concurrency, label="compile", noun="gate(s)"
    )


# -- checking a precondition before it ships ----------------------------------


def check_precondition(
    sql: str,
    *,
    schema: str,
    session_ids: Sequence[str],
    source: PgQuerySource,
) -> tuple[dict[str, bool], str]:
    """Which of these sessions the precondition holds for, and any error.

    Run before a gate is accepted. An item whose precondition is true nowhere is
    inert; one that is true everywhere is not gating anything. Both are worth
    seeing before spending a replay.

    Takes an open ``source`` rather than a DSN: a caller checks every compiled
    item against the same sessions, and building an engine per item made the
    connection cost scale with the checklist for no reason.
    """
    if not sql.strip():
        return ({sid: True for sid in session_ids}, "")
    try:
        expanded = facts.expand(sql, schema)
    except ValueError as exc:
        return ({}, str(exc))

    outcome: dict[str, bool] = {}
    try:
        for session_id in session_ids:
            rows = source.query(expanded, {"session_id": session_id})
            outcome[session_id] = bool(rows)
    except Exception as exc:  # noqa: BLE001 - a bad precondition is a result
        return (outcome, f"{type(exc).__name__}: {exc}")
    return (outcome, "")


__all__ = [
    "RESULT_TOOL",
    "build_prompt",
    "check_precondition",
    "compile_candidates",
    "compile_one",
]
