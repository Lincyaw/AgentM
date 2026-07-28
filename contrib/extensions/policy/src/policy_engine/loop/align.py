# code-health: ignore-file[AM025] -- a review's report and the model's verdict
# both arrive as untyped JSON; the isinstance checks here are that boundary.
"""Did the review find what grading punished?

The loop needs a signal, and the score is not one. A score moves only when two
independent things go right: the review finds the defect the tests punish, and
the agent then fixes it correctly. Across twenty-one measured runs it never
moved, which leaves no way to tell a review that was one step away from one
that was in the wrong subsystem.

This asks the first question by itself. The graded failures are known -- they
are in the case that started the run -- and the review's report is in the child
session it wrote. Comparing them has an answer every time, which is the whole
point: a dense signal that can order things the score cannot.

What it cannot do is stand in for the score. A review can find the right defect
and the agent still not fix it, and this will call that a success. It measures
the reviewer, and it should only ever be read as measuring the reviewer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loguru import logger

from agentm.core.abi import JsonValue

from .contracts import ALIGNMENTS, Alignment, Assertion, FailureCase
from .runner import AgentRun, ResultTool, fan_out, load_stage_manifest

RESULT_TOOL = ResultTool(
    name="submit_alignment",
    description="Record whether the review found what grading punished. Call once.",
    parameters={  # code-health: ignore[AM011]
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ALIGNMENTS,
                "description": "same: fixing the review's finding makes the "
                "assertion pass. adjacent: same code, different defect. "
                "elsewhere: real but unrelated. none: no case reported.",
            },
            "finding": {
                "type": "string",
                "description": "The review's strongest finding, in one line.",
            },
            "graded_failure": {
                "type": "string",
                "description": "The assertion you compared it against.",
            },
            "reason": {
                "type": "string",
                "description": "Why that verdict. For 'same', the line from "
                "the finding to the assertion no longer failing.",
            },
        },
        "required": ("verdict", "reason"),
    },
)


def build_prompt(assertions: Sequence[Assertion], report: str) -> str:
    lines = ["# What the graded tests reported", ""]
    if assertions:
        for item in assertions:
            lines.append(f"- [{item.kind}] {item.assertion_id}")
            if item.message.strip():
                lines.append(f"  {item.message.strip()[:600]}")
    else:
        lines.append("(none recorded)")
    lines += ["", "# What the review reported", "", report.strip()[:12000]]
    return "\n".join(lines)


async def align_one(
    case: FailureCase,
    report: str,
    *,
    candidate_id: str = "",
    provider: str = "",
    user_config: str = "",
    agent: AgentRun | None = None,
) -> Alignment | None:
    if not report.strip():
        return Alignment(
            case_id=case.case_id,
            candidate_id=candidate_id,
            verdict="none",
            reason="the review produced no report",
        )

    run = agent or AgentRun(
        manifest=load_stage_manifest("aligner"),
        result_tool=RESULT_TOOL,
        provider=provider,
        user_config=user_config,
    )
    payload = await run.run(build_prompt(case.failing_assertions, report))
    if payload is None:
        return None

    verdict = _text(payload, "verdict")
    if verdict not in ALIGNMENTS:
        logger.warning("align: {} returned verdict {!r}", case.case_id, verdict)
        verdict = "none"
    return Alignment(
        case_id=case.case_id,
        candidate_id=candidate_id,
        verdict=verdict,
        finding=_text(payload, "finding"),
        graded_failure=_text(payload, "graded_failure"),
        reason=_text(payload, "reason"),
    )


async def align(
    pairs: Sequence[tuple[FailureCase, str, str]],
    *,
    provider: str = "",
    user_config: str = "",
    concurrency: int = 6,
) -> list[Alignment]:
    """Align every (case, review report, candidate id) triple."""

    async def one(item: tuple[FailureCase, str, str]) -> Alignment | None:
        case, report, candidate_id = item
        return await align_one(
            case,
            report,
            candidate_id=candidate_id,
            provider=provider,
            user_config=user_config,
        )

    return await fan_out(
        pairs, one, concurrency=concurrency, label="align", noun="alignment(s)"
    )


def tally(alignments: Sequence[Alignment]) -> dict[str, int]:
    """How many of each verdict, in the order that means best to worst."""
    counts = dict.fromkeys(ALIGNMENTS, 0)
    for item in alignments:
        counts[item.verdict] = counts.get(item.verdict, 0) + 1
    return counts


def _text(raw: Mapping[str, JsonValue], key: str) -> str:
    value = raw.get(key)
    return value.strip() if isinstance(value, str) else ""


__all__ = ["RESULT_TOOL", "align", "align_one", "build_prompt", "tally"]
