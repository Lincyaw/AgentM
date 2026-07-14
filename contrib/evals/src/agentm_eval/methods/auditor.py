"""LLM-harness auditor method adapter.

Single entry point for running the cognitive-audit auditor offline.
Delegates to the atom's ``build_auditor_config`` / ``build_auditor_prompt``
/ ``run_auditor_session`` — no duplicated session creation.

Usage::

    from agentm_eval.methods.auditor import run_auditor, AuditResult

    result = await run_auditor(trajectory, model="azure-gpt")
    for v in result.verdicts:
        print(v.reminder_text if v.surface_reminder else "(no surface)")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentm.core.abi import AgentMessage
from loguru import logger

from llmharness.schema import Verdict
from llmharness.state import CumulativeAuditState


@dataclass(slots=True)
class AuditResult:
    """Outcome of running the auditor over a trajectory."""

    verdicts: list[Verdict] = field(default_factory=list)
    surfaces: list[Verdict] = field(default_factory=list)
    state: CumulativeAuditState = field(default_factory=CumulativeAuditState.fresh)
    session_ids: list[str] = field(default_factory=list)


# Canonical home is the trajectory_index package (pure TrajectoryIndex -> dict);
# re-exported here so existing eval callers keep importing it from this module.


_PHASE1_DISCRIMINATE = (
    "Before localizing, investigate the trajectory with your tools "
    "(list_turns, get_turn, list_claim_checks) and answer ONE question: is "
    "this a TERMINAL-COMMIT error or an UPSTREAM-PROCESS error?\n\n"
    "- TERMINAL-COMMIT: the agent's earlier steps were justified given the "
    "evidence they had, and the mistake is the final step committing to a "
    "wrong or unsupported answer. The error is narrow — essentially the last "
    "step.\n"
    "- UPSTREAM-PROCESS: the mistake was made earlier — a search that did not "
    "verify, a source trusted without checking, a decision to conclude on "
    "insufficient evidence — and later steps (including the final one) merely "
    "carry it forward. The final answer may even be correct; the error is a "
    "chain of upstream steps.\n\n"
    "Read the task and the turns, then state the mode and the specific "
    "step(s) you believe carry the error, with a one-line reason. Do NOT call "
    "submit_verdict yet."
)

_PHASE2_LOCALIZE = (
    "Now call submit_verdict exactly once, localizing according to the mode "
    "you determined:\n"
    "- If TERMINAL-COMMIT: set matched_event_ids to the committing step(s) "
    "only. Be narrow — do not pad with earlier steps that were justified at "
    "the time.\n"
    "- If UPSTREAM-PROCESS: set matched_event_ids to the upstream step(s) "
    "where the error entered and is carried forward. Include the final step "
    "only if it independently errs.\n"
    "Fill reminder_text, evidence, and continuation_notes as specified."
)


def _critic_prompt(verdict_raw: dict[str, Any]) -> str:
    ids = verdict_raw.get("matched_event_ids", [])
    reminder = verdict_raw.get("reminder_text", "")
    surfaced = bool(verdict_raw.get("surface_reminder"))
    claim = (
        f"Another auditor reviewed this trajectory and localized the error to "
        f"turn(s) {ids} with this reasoning:\n\n{reminder!r}\n\n"
        if surfaced
        else "Another auditor reviewed this trajectory and found no error.\n\n"
    )
    return (
        claim
        + "You are a critic. Independently verify that localization by reading "
        "the trajectory (list_turns, get_turn, list_claim_checks). Test each "
        "claim adversarially:\n"
        "- For every flagged turn, decide whether that turn is ITSELF an error "
        "given the evidence available at that point — or whether it was a "
        "justified action (a reasonable search, a correct intermediate result) "
        "that merely came before a later mistake. A turn that was warranted "
        "when taken is not an error even if the final answer is wrong; remove "
        "it.\n"
        "- Watch for over-attribution: an unsupported final answer does not by "
        "itself mean an earlier step was wrong. Do not blame upstream steps "
        "unless you can point to the specific unjustified move in that step.\n"
        "- Conversely, if the flagged turns were all justified and the real "
        "mistake is a step the other auditor missed, name that step instead.\n\n"
        "Then call submit_verdict once with the corrected localization: "
        "matched_event_ids should contain only the turns that themselves carry "
        "an error. If after review the agent's work is sound at every step, set "
        "surface_reminder=false."
    )


async def run_auditor(
    messages: list[AgentMessage],
    *,
    model: str | None = None,
    provider: tuple[str, dict[str, Any]] | None = None,
    cwd: str = ".",
    auditor_prompt: str = "index",
    audit_interval: int | None = None,
    index_path: str = "",
    two_phase: bool = False,
    critic: bool = False,
) -> AuditResult:
    """Run the auditor over a trajectory.

    Accepts typed ``AgentMessage`` objects; serialization is handled
    internally. When ``audit_interval`` is None (default), fires the
    auditor once (post-hoc). When set, fires every N turns with state
    accumulation (online-simulation mode).
    """
    from agentm.core.runtime import AgentSession
    from llmharness.atom import (
        build_auditor_config,
        build_auditor_prompt,
        run_auditor_session,
    )

    cumulative = CumulativeAuditState.fresh()
    result = AuditResult(state=cumulative)

    def _absorb(verdict_raw: dict[str, Any] | None, sid: str) -> None:
        result.session_ids.append(sid)
        if not isinstance(verdict_raw, dict):
            return
        verdict = Verdict.from_dict(verdict_raw)
        cumulative.absorb_auditor_verdict(verdict.to_dict())
        result.verdicts.append(verdict)
        if verdict.surface_reminder and verdict.reminder_text:
            result.surfaces.append(verdict)

    if audit_interval is None:
        config = build_auditor_config(
            cwd=cwd, model=model, provider=provider,
            auditor_prompt=auditor_prompt,
            messages=messages,
            index_path=index_path,
        )
        if two_phase:
            phase1 = f"{_PHASE1_DISCRIMINATE}\n\n{build_auditor_prompt()}"
            verdict_raw, sid = await run_auditor_session(
                config, phase1, followup_prompt=_PHASE2_LOCALIZE,
                spawn=AgentSession.create,
            )
        else:
            prompt = build_auditor_prompt()
            verdict_raw, sid = await run_auditor_session(config, prompt, spawn=AgentSession.create)

        if critic and isinstance(verdict_raw, dict):
            # Independent critic re-localizes: prunes turns that were justified
            # when taken, guards against over-attributing an unsupported final
            # answer to upstream steps. Its verdict replaces the auditor's.
            result.session_ids.append(sid)
            critic_config = build_auditor_config(
                cwd=cwd, model=model, provider=provider,
                auditor_prompt=auditor_prompt,
                messages=messages,
                index_path=index_path,
            )
            revised, csid = await run_auditor_session(
                critic_config, _critic_prompt(verdict_raw),
                spawn=AgentSession.create,
            )
            _absorb(revised if isinstance(revised, dict) else verdict_raw, csid)
            return result

        _absorb(verdict_raw, sid)
        return result

    n_turns = len(messages)
    for turn in range(1, n_turns + 1):
        if turn % audit_interval != 0:
            continue
        try:
            config = build_auditor_config(
                cwd=cwd, model=model, provider=provider,
                auditor_prompt=auditor_prompt,
                continuation_notes=list(cumulative.last_continuation_notes),
                messages=messages[:turn],
                index_path=index_path,
            )
            prompt = build_auditor_prompt(list(cumulative.last_continuation_notes))
            verdict_raw, sid = await run_auditor_session(config, prompt, spawn=AgentSession.create)
            _absorb(verdict_raw, sid)
        except Exception as exc:
            logger.warning("auditor firing at turn {} failed: {}", turn, exc)

    return result
