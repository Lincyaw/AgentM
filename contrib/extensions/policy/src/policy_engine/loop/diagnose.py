"""Stage 2: ``FailureCase`` -> ``Diagnosis``, by agent.

The prompt hands over pointers, not contents. Patches run to thousands of lines
and the trajectory to hundreds of turns; pasting either wastes the budget on
material the agent can read selectively with the tools it already has. What is
inlined is only what it cannot look up: the task as the agent received it, and
the assertion text the graders produced.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence

from loguru import logger

from agentm.core.abi import JsonValue

from .contracts import CAUSE_CLASSES, Diagnosis, FailureCase
from .protocols import CaseSource
from .runner import AgentRun, ResultTool, fan_out, load_stage_manifest

RESULT_TOOL = ResultTool(
    name="submit_diagnosis",
    description="Record why this attempt diverged. Call once, when you can "
    "state the mechanism concretely.",
    parameters={  # code-health: ignore[AM011]
        "type": "object",
        "properties": {
            "required_behaviour": {
                "type": "string",
                "description": "What grading demands, quoted from the failing "
                "assertions.",
            },
            "reference_idea": {
                "type": "string",
                "description": "The reference solution's central idea, one sentence.",
            },
            "agent_idea": {
                "type": "string",
                "description": "What the agent built instead, in the same terms.",
            },
            "divergence_turn": {
                "type": "integer",
                "description": "Turn index where the approach stopped being "
                "open. -1 if you cannot locate it.",
            },
            "divergence_quote": {
                "type": "string",
                "description": "What was said or done at that turn.",
            },
            "cause_class": {
                "type": "string",
                "enum": CAUSE_CLASSES,
            },
            "mechanism": {
                "type": "string",
                "description": "Why it is wrong, concretely: what breaks, under "
                "what condition. Not 'it did not test enough'.",
            },
            "decision": {
                "type": "string",
                "description": "What was being chosen at the divergence turn, "
                "in the agent's own terms.",
            },
            "unasked_question": {
                "type": "string",
                "description": "The step not taken: a question the agent could "
                "have asked itself at that moment, answerable only by running "
                "something. Must be askable without knowing the answer -- if it "
                "names the defect, it is hindsight, not a question.",
            },
            "discriminating_answer": {
                "type": "string",
                "description": "What asking it would have produced, and what the "
                "agent believed instead. If the answer would look the same "
                "whether or not the agent was right, this is not the missing "
                "step; find the question whose answer differs.",
            },
            "lesson": {
                "type": "string",
                "description": "What someone starting a different task in this "
                "same repository should know because of this. May name this "
                "repository's conventions and traps; may not name this task's "
                "defect. Empty if this case teaches nothing.",
            },
            "reachable": {
                "type": "boolean",
                "description": "Could anything said to the agent before it "
                "submitted have changed this outcome?",
            },
            "reachable_rationale": {
                "type": "string",
                "description": "When not reachable, why not. A justified no is "
                "worth more than an invented yes.",
            },
            "evidence": {
                "type": "array",
                "description": "Every claim with a quote and where it came from.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ("source", "quote"),
                },
            },
        },
        "required": (
            "required_behaviour",
            "reference_idea",
            "agent_idea",
            "cause_class",
            "mechanism",
            "decision",
            "unasked_question",
            "discriminating_answer",
            "lesson",
            "reachable",
            "evidence",
        ),
    },
)


def build_prompt(case: FailureCase, *, environment: str) -> str:
    """The case, rendered. Every benchmark-specific pointer arrives already
    written by the adapter, so this function never learns what a verifier
    produces or how to read a trajectory."""
    lines = [
        f"# Task: {case.task_name}",
        "",
        "## What the agent was asked",
        case.instruction or "(not recorded)",
        "",
        "## What failed",
    ]
    if case.failing_assertions:
        for assertion in case.failing_assertions:
            lines.append(f"- [{assertion.kind}] {assertion.assertion_id}")
            if assertion.message:
                lines.append(f"      {assertion.message.strip()[:600]}")
    else:
        lines.append("(no assertion detail recorded; read the graded output)")

    lines += ["", "## Scores", f"this attempt: {case.metrics.render()}"]
    if len(case.sibling_metrics) > 1:
        others = ", ".join(m.render() for m in case.sibling_metrics)
        lines.append(f"all attempts of this task: {others}")
    if case.cohort_note:
        lines += ["", "## How the attempts compare", case.cohort_note]

    if case.evidence:
        lines += ["", "## Where to look"]
        for ref in case.evidence:
            lines.append(f"- {ref.label}: {ref.locator}")
            if ref.note:
                lines.append(f"      {ref.note}")

    lines += ["", "## Your shell"]
    lines.append(
        environment
        or "There is no machine for this attempt, so you have the artefacts "
        "above and not the repository. Some causes are only visible in the "
        "code; where that bites, say so rather than inferring a mechanism from "
        "the diff."
    )
    return "\n".join(lines)


async def diagnose_one(
    case: FailureCase,
    *,
    provider: str = "",
    user_config: str = "",
    agent: AgentRun | None = None,
    source: CaseSource | None = None,
) -> Diagnosis | None:
    """One diagnosis, in the attempt's own environment where that is possible.

    Without the sandbox the agent sees patches and a trajectory but not the
    repository, and some causes live only there: one hand analysis turned on a
    test already in the repository that endorsed the wrong behaviour, which
    appears in neither diff. So the sandbox is tried first and its absence is
    told to the agent rather than hidden, because a diagnosis made without the
    code should not read like one made with it.
    """
    env = await source.open_environment(case) if source is not None else None

    try:
        run = agent or AgentRun(
            manifest=load_stage_manifest("diagnoser"),
            result_tool=RESULT_TOOL,
            provider=provider,
            user_config=user_config,
            cwd=env.work_dir if env is not None else ".",
            operations=env.operations if env is not None else None,
            writer=env.writer if env is not None else None,
        )
        payload = await run.run(
            build_prompt(case, environment=env.description if env else "")
        )
    finally:
        if env is not None:
            await env.operations.close()

    if payload is None:
        return None
    return _to_diagnosis(payload, case)


def _to_diagnosis(payload: Mapping[str, JsonValue], case: FailureCase) -> Diagnosis:
    enriched: dict[str, JsonValue] = dict(payload)
    enriched["diagnosis_id"] = f"dx-{uuid.uuid4().hex[:10]}"
    enriched["case_id"] = case.case_id
    enriched["task_name"] = case.task_name
    return Diagnosis.from_json(enriched)


async def diagnose(
    cases: Sequence[FailureCase],
    *,
    provider: str = "",
    user_config: str = "",
    concurrency: int = 4,
    source: CaseSource | None = None,
) -> list[Diagnosis]:
    """Cases are independent, so they run together. Concurrency is bounded
    because each one holds a session, a model connection and, when the fork
    succeeds, a sandbox on the cluster."""

    async def one(case: FailureCase) -> Diagnosis | None:
        logger.info("diagnose: {}", case.case_id)
        return await diagnose_one(
            case,
            provider=provider,
            user_config=user_config,
            source=source,
        )

    return await fan_out(
        cases, one, concurrency=concurrency, label="diagnose", noun="diagnosis(es)"
    )


__all__ = ["RESULT_TOOL", "build_prompt", "diagnose", "diagnose_one"]
