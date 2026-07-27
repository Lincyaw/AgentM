"""Stage 3: ``Diagnosis`` -> ``Candidate``, by agent.

Two guards live here rather than in the prompt, because a prompt is a request
and these have to hold.

**The abstractor is kept blind.** It never sees the reference solution's idea,
and evidence quoted out of the oracle patch is dropped before the prompt is
built. Given either, it writes a check that names the specific defect: perfect
on the case it came from, worthless on any other, and indistinguishable from a
good check to everything downstream.

The blindfold is structural for the reference idea and filtered for evidence,
but the diagnosis's own prose passes through unexamined -- and the diagnoser did
read the reference. What protects the result is not this function: it is
measuring whether a candidate transfers to cases it did not come from. Treat the
blindfold as removing the easy leak, not as a guarantee.

**Cases with nothing to learn are not abstracted at all.** A grading artifact or
a nondeterministic harness has no check behind it, and asking for one produces a
plausible check aimed at a defect that is not there.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence

from loguru import logger

from agentm.core.abi import JsonValue

from .contracts import Candidate, Diagnosis, Evidence
from .runner import AgentRun, ResultTool, fan_out, load_stage_manifest

RESULT_TOOL = ResultTool(
    name="submit_candidate",
    description="Record the check this diagnosis implies. Call once.",
    parameters={  # code-health: ignore[AM011]
        "type": "object",
        "properties": {
            "check": {
                "type": "string",
                "description": "The message the agent will receive. One doubt, "
                "plainly put. No file, function or value from this task.",
            },
            "observation": {
                "type": "string",
                "description": "What running this check actually produces: a "
                "number, an output, an exit status. Empty means the check is "
                "reflective, and the candidate will be discarded.",
            },
            "when_note": {
                "type": "string",
                "description": "In plain language, what must be true of the "
                "session for this to be worth sending.",
            },
            "precondition_note": {
                "type": "string",
                "description": "In plain language, what must be observably true "
                "of the session's recorded actions. Empty means unconditional.",
            },
            "checkpoint": {
                "type": "string",
                "enum": ("continuous", "stop"),
                "description": "continuous: mid-work. stop: when the agent wraps up.",
            },
            "dimension": {
                "type": "string",
                "description": "Which representation-chain dimension this "
                "guards, D1 to D7.",
            },
        },
        "required": ("check", "observation", "when_note", "checkpoint"),
    },
)

#: Evidence whose source looks like the reference solution. Dropped before the
#: prompt is built: a quote from the oracle patch is the answer.
_ORACLE_MARKERS = ("oracle", "judge/oracle", "reference")


def visible_evidence(diagnosis: Diagnosis) -> tuple[Evidence, ...]:
    return tuple(
        item
        for item in diagnosis.evidence
        if not any(marker in item.source.lower() for marker in _ORACLE_MARKERS)
    )


def build_prompt(diagnosis: Diagnosis) -> str:
    lines = [
        "# One diagnosis",
        "",
        "## What grading required",
        diagnosis.required_behaviour or "(not recorded)",
        "",
        "## What the agent built instead",
        diagnosis.agent_idea or "(not recorded)",
        "",
        "## Why that is wrong",
        diagnosis.mechanism or "(not recorded)",
        "",
        f"## Cause class\n{diagnosis.cause_class or '(unclassified)'}",
    ]
    if diagnosis.decision:
        lines += ["", "## The choice being made", diagnosis.decision]
    if diagnosis.unasked_question:
        lines += [
            "",
            "## The question it did not put to itself",
            diagnosis.unasked_question,
        ]
    if diagnosis.discriminating_answer:
        lines += [
            "",
            "## What asking it would have shown",
            diagnosis.discriminating_answer,
        ]
    if diagnosis.lesson:
        lines += [
            "",
            "## What carries to the next task in this repository",
            diagnosis.lesson,
        ]
    if diagnosis.divergence_turn >= 0 and diagnosis.divergence_quote:
        lines += [
            "",
            f"## Where the approach became fixed (turn {diagnosis.divergence_turn})",
            diagnosis.divergence_quote.strip()[:1200],
        ]
    shown = visible_evidence(diagnosis)
    if shown:
        lines += ["", "## Evidence"]
        for item in shown[:8]:
            lines.append(f"- {item.source}: {item.quote.strip()[:400]}")
    lines += [
        "",
        (
            "Write the check that would have changed this outcome, for the class "
            "of mistake rather than for this task. The question it did not ask "
            "itself is the raw material: your job is to put it in a form that "
            "can be sent before anyone knows it is needed."
        ),
    ]
    return "\n".join(lines)


async def abstract_one(
    diagnosis: Diagnosis,
    *,
    provider: str = "",
    user_config: str = "",
    agent: AgentRun | None = None,
) -> Candidate | None:
    if not diagnosis.worth_abstracting:
        logger.info(
            "abstract: skipping {} ({}, reachable={})",
            diagnosis.case_id,
            diagnosis.cause_class or "unclassified",
            diagnosis.reachable,
        )
        return None

    run = agent or AgentRun(
        manifest=load_stage_manifest("abstractor"),
        result_tool=RESULT_TOOL,
        provider=provider,
        user_config=user_config,
    )
    payload = await run.run(build_prompt(diagnosis))
    if payload is None:
        return None

    candidate = _to_candidate(payload, diagnosis)
    rejection = candidate.rejection()
    if rejection:
        logger.info(
            "abstract: rejected candidate from {}: {}", diagnosis.case_id, rejection
        )
        return None
    return candidate


def _to_candidate(payload: Mapping[str, JsonValue], diagnosis: Diagnosis) -> Candidate:
    enriched: dict[str, JsonValue] = dict(payload)
    enriched["candidate_id"] = f"cand-{uuid.uuid4().hex[:10]}"
    enriched["from_task"] = diagnosis.task_name
    enriched["from_session"] = diagnosis.case_id
    enriched["from_diagnosis"] = diagnosis.diagnosis_id
    enriched["from_turn"] = diagnosis.divergence_turn
    return Candidate.from_json(enriched)


async def abstract(
    diagnoses: Sequence[Diagnosis],
    *,
    provider: str = "",
    user_config: str = "",
    concurrency: int = 4,
) -> list[Candidate]:
    async def one(diagnosis: Diagnosis) -> Candidate | None:
        return await abstract_one(diagnosis, provider=provider, user_config=user_config)

    return await fan_out(
        diagnoses, one, concurrency=concurrency, label="abstract", noun="candidate(s)"
    )


__all__ = [
    "RESULT_TOOL",
    "abstract",
    "abstract_one",
    "build_prompt",
    "visible_evidence",
]
