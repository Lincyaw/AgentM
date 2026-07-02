"""Treatment factory + oracle builder (doc §7).

For each prefix we emit the content ladder (doc §7.2): from CONTINUE (no info)
through TYPE / TYPE+TARGET / EVIDENCE up to the oracle. The oracle is the anchor
that defines G*_t. Per the locked decision (DESIGN §4.1):

  ORACLE_GROUNDED = strong model + GT, restricted to prefix-consistent
  information, channel-respecting. It steers toward the GT root cause but cites
  only evidence visible in the prefix (turn indices <= the fork turn) and stays a
  typed VERIFY/REPLAN nudge, not the answer.

  ORACLE_DIAG = raw GT diagnosis injected directly. A separate, higher ceiling
  (actor/state-recovery), not the realizable channel.

No-future-leakage is enforced: a grounded oracle whose cited evidence turns fall
outside the prefix is dropped rather than allowed to leak.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agentm.core.abi import AgentMessage, AssistantMessage, ToolCallBlock
from loguru import logger

from ..model import (
    ActionType,
    ContentLevel,
    Intervention,
    LadderRung,
    PrefixPoint,
    Treatment,
)
from .adapter import GroundTruth

_ACTION_VERB = {
    ActionType.GENERIC: "there may be an issue — please double-check before continuing",
    ActionType.VERIFY: "re-verify a key requirement, assumption, or test interpretation",
    ActionType.ADVISE: "here is a risk area and a suggested correction direction",
    ActionType.REPLAN: "pause the current direction and re-plan from here",
    ActionType.FINAL_AUDIT: "audit your conclusion against the requirements and the evidence before submitting",
}

Condition = tuple[ContentLevel, ActionType]

# "oracle-landscape": sweep ACTIONS at oracle-level targeting (GT service as
# target) + controls. Answers "does rescue opportunity exist, and which action
# type works?" — each action isolates a different intervention strength.
ORACLE_LANDSCAPE: tuple[Condition, ...] = (
    (ContentLevel.CONTINUE, ActionType.CONTINUE),       # fork + resume, no injection
    (ContentLevel.PLACEBO, ActionType.CONTINUE),         # token-matched neutral text
    (ContentLevel.TYPE_TARGET, ActionType.GENERIC),      # bare alarm, no specifics
    (ContentLevel.TYPE_TARGET, ActionType.VERIFY),       # ask to verify GT service
    (ContentLevel.TYPE_TARGET, ActionType.ADVISE),       # point out risk + direction
    (ContentLevel.TYPE_TARGET, ActionType.REPLAN),       # ask to re-plan around GT service
    (ContentLevel.TYPE_TARGET, ActionType.FINAL_AUDIT),  # ask to audit against GT service before submit
    (ContentLevel.ORACLE_DIAG, ActionType.ADVISE),       # give the answer directly (actor ceiling)
)

# "content-ladder": fix action to VERIFY, sweep INFORMATION from none to
# oracle. Answers "what is the minimal effective information?"
CONTENT_LADDER: tuple[Condition, ...] = (
    (ContentLevel.CONTINUE, ActionType.CONTINUE),
    (ContentLevel.PLACEBO, ActionType.CONTINUE),
    (ContentLevel.GENERIC, ActionType.VERIFY),
    (ContentLevel.TYPE, ActionType.VERIFY),
    (ContentLevel.TYPE_TARGET, ActionType.VERIFY),
    (ContentLevel.EVIDENCE, ActionType.VERIFY),
    (ContentLevel.ORACLE_DIAG, ActionType.VERIFY),
)

PRESETS: dict[str, tuple[Condition, ...]] = {
    "oracle-landscape": ORACLE_LANDSCAPE,
    "content-ladder": CONTENT_LADDER,
}


class OracleBuilder(Protocol):
    """Produces the ORACLE_GROUNDED intervention for a prefix, or None."""

    async def build(
        self,
        *,
        prefix: PrefixPoint,
        messages: list[AgentMessage],
        gt: GroundTruth,
    ) -> Intervention | None:
        ...


class TreatmentFactory:
    """Build the doc's condition set for a prefix (doc §7.2 / §8)."""

    def __init__(
        self,
        *,
        conditions: tuple[Condition, ...] = ORACLE_LANDSCAPE,
        oracle: OracleBuilder | None = None,
    ) -> None:
        self.conditions = conditions
        self.oracle = oracle

    async def build(
        self,
        prefix: PrefixPoint,
        messages: list[AgentMessage],
        gt: GroundTruth,
    ) -> list[Treatment]:
        treatments: list[Treatment] = []
        # A representative targeted message, so PLACEBO can token-match its length.
        evidence_msg = self._evidence_message(gt, ActionType.VERIFY)
        for level, action in self.conditions:
            if level is ContentLevel.ORACLE_GROUNDED:
                if self.oracle is not None:
                    grounded = await self._grounded(prefix, messages, gt)
                    if grounded is not None:
                        treatments.append(grounded)
                continue
            treatment = self._deterministic(level, action, gt, evidence_msg)
            if treatment is not None:
                treatments.append(treatment)
        return treatments

    def _deterministic(
        self, level: ContentLevel, action: ActionType, gt: GroundTruth, evidence_msg: str
    ) -> Treatment | None:
        target = gt.primary_target
        needs_target = level in {ContentLevel.TYPE_TARGET, ContentLevel.EVIDENCE, ContentLevel.ORACLE_DIAG}
        if needs_target and not target:
            return None

        # --- controls (no action label) ---
        if level is ContentLevel.CONTINUE:
            return self._make(level, ActionType.CONTINUE, "", gt, LadderRung.CHANNEL)
        if level is ContentLevel.PLACEBO:
            return self._make(level, ActionType.CONTINUE, _placebo(evidence_msg), gt, LadderRung.CHANNEL)

        # --- action × content composition ---
        msg = _render_message(action, level, gt)
        rung = LadderRung.ACTOR if level is ContentLevel.ORACLE_DIAG else LadderRung.CHANNEL
        return self._make(level, action, msg, gt, rung, target=target if needs_target else None)

    async def _grounded(
        self, prefix: PrefixPoint, messages: list[AgentMessage], gt: GroundTruth
    ) -> Treatment | None:
        if self.oracle is None:
            return None
        intervention = await self.oracle.build(prefix=prefix, messages=messages, gt=gt)
        if intervention is None:
            return None
        return Treatment(
            treatment_id=f"{intervention.action.value}:{ContentLevel.ORACLE_GROUNDED.value}",
            content_level=ContentLevel.ORACLE_GROUNDED,
            action=intervention.action,
            intervention=intervention,
            rung=LadderRung.CHANNEL,
        )

    def _make(
        self,
        level: ContentLevel,
        action: ActionType,
        message: str,
        gt: GroundTruth,
        rung: LadderRung,
        *,
        target: str | None = None,
    ) -> Treatment:
        target_dict = {"service": target} if target else {}
        treatment_id = (
            level.value
            if action is ActionType.CONTINUE
            else f"{action.value}:{level.value}"
        )
        intervention = Intervention(
            action=action,
            condition_id=level.value,
            content_level=level.value,
            message=message,
            target=target_dict,
            evidence={},
            strength="advisory",
            valid_until={},
            metadata={"gt_targets": list(gt.targets)},
        )
        return Treatment(
            treatment_id=treatment_id,
            content_level=level,
            action=action,
            intervention=intervention,
            rung=rung,
        )

    def _evidence_message(self, gt: GroundTruth, action: ActionType) -> str:
        return _render_message(action, ContentLevel.EVIDENCE, gt)


def _render_message(action: ActionType, level: ContentLevel, gt: GroundTruth) -> str:
    """Compose an intervention message from the action × content axes."""

    target = gt.primary_target
    verb = _ACTION_VERB.get(action, "re-check your reasoning")

    # GENERIC content: bare alarm, no target/evidence
    if level is ContentLevel.GENERIC:
        return f"[{action.value}] {verb}."

    # TYPE: action label only, no target
    if level is ContentLevel.TYPE:
        return f"[{action.value}] Before you conclude, {verb}."

    # TYPE_TARGET: action + concrete object
    if level is ContentLevel.TYPE_TARGET:
        if not target:
            return f"[{action.value}] Before you conclude, {verb}."
        return f"[{action.value}] Before you conclude, {verb}, focusing on `{target}`."

    # EVIDENCE: action + target + prefix-visible evidence pointer
    if level is ContentLevel.EVIDENCE:
        if not target:
            return f"[{action.value}] Before you conclude, {verb}."
        return (
            f"[{action.value}] Before you conclude, {verb}, focusing on `{target}`. "
            f"The evidence gathered so far points at `{target}` in a way the investigation "
            "has not yet examined closely."
        )

    # ORACLE_DIAG: give the answer directly
    if level is ContentLevel.ORACLE_DIAG:
        return (
            f"[{action.value}] The verified root cause for this incident is "
            f"{gt.summary}. Re-orient your investigation accordingly before concluding."
        )

    return f"[{action.value}] {verb}."


# --- oracle implementations -------------------------------------------------


@dataclass(slots=True)
class StrongModelOracle:
    """ORACLE_GROUNDED via a strong model that knows GT but must stay prefix-grounded.

    Degrades to ``None`` on any error so the pipeline can still run the
    deterministic ladder. Evidence turns the model cites are validated against the
    prefix; an out-of-prefix citation drops the row (no future leakage).
    """

    provider: tuple[str, dict[str, Any]]
    max_prefix_turns: int = 12

    async def build(
        self,
        *,
        prefix: PrefixPoint,
        messages: list[AgentMessage],
        gt: GroundTruth,
    ) -> Intervention | None:
        try:
            rendered = _render_prefix(messages, max_turns=self.max_prefix_turns)
            system = _ORACLE_SYSTEM
            user = _oracle_user_prompt(rendered, gt, prefix.turn_index)
            text = await _run_text_model(self.provider, system=system, user=user)
            obj = _parse_json_object(text)
            if obj is None:
                return None
            message = str(obj.get("message") or "").strip()
            if not message:
                return None
            evidence_turns = [
                int(value)
                for value in obj.get("evidence_turns") or []
                if isinstance(value, int) and not isinstance(value, bool)
            ]
            if any(turn < 0 or turn > prefix.turn_index for turn in evidence_turns):
                logger.warning(
                    "StrongModelOracle dropped a row: evidence turns {} exceed prefix turn {}",
                    evidence_turns,
                    prefix.turn_index,
                )
                return None
            action = _parse_action(obj.get("action"))
            target = str(obj.get("target") or gt.primary_target or "").strip()
            return Intervention(
                action=action,
                condition_id=ContentLevel.ORACLE_GROUNDED.value,
                content_level=ContentLevel.ORACLE_GROUNDED.value,
                message=message,
                target={"service": target} if target else {},
                evidence={"trajectory_event_ids": [str(t) for t in evidence_turns]},
                strength="advisory",
                valid_until={},
                metadata={"oracle": "strong_model", "gt_targets": list(gt.targets)},
            )
        except Exception as exc:  # noqa: BLE001 -- oracle failure must not sink the prefix
            logger.warning("StrongModelOracle failed for {}: {}", prefix.prefix_id, exc)
            return None


_ORACLE_SYSTEM = (
    "You are a perfect overseer auditing an in-progress root-cause investigation. "
    "You privately know the verified ground truth. Your job is to emit the single "
    "best low-bandwidth nudge that steers the investigator toward the true root "
    "cause, subject to two hard rules: (1) cite only evidence that is already "
    "visible in the shown prefix; never reference anything the investigator could "
    "not yet know; (2) stay a short typed nudge (VERIFY / ADVISE / REPLAN /"
    "FINAL_AUDIT), not the answer. Output one JSON object only."
)


def _oracle_user_prompt(rendered_prefix: str, gt: GroundTruth, turn_index: int) -> str:
    return json.dumps(
        {
            "instruction": (
                "Return JSON {action, target, message, evidence_turns}. "
                "action is one of VERIFY/ADVISE/REPLAN/FINAL_AUDIT. target is the "
                "service to steer toward. message is the nudge text shown to the "
                "investigator (do not reveal that you know ground truth). "
                "evidence_turns is a list of prefix turn indices (<= "
                f"{turn_index}) that justify the nudge."
            ),
            "ground_truth": {
                "targets": list(gt.targets),
                "fault_kinds": list(gt.fault_kinds),
                "summary": gt.summary,
            },
            "visible_prefix": rendered_prefix,
            "current_turn_index": turn_index,
        },
        ensure_ascii=False,
    )


async def _run_text_model(
    provider: tuple[str, dict[str, Any]], *, system: str, user: str
) -> str:
    from agentm.core.abi import AgentSessionConfig
    from agentm.core.runtime import AgentSession

    config = AgentSessionConfig(
        cwd=str(Path.cwd()),
        provider=provider,
        extensions=[
            ("agentm.extensions.builtin.observability", {}),
            ("agentm.extensions.builtin.system_prompt", {"prompt": system}),
        ],
        purpose="rescue_window_oracle",
    )
    session = await AgentSession.create(config)
    try:
        messages = await session.prompt(user)
    finally:
        with contextlib.suppress(Exception):
            await session.shutdown()
    return _assistant_text(messages)


def _assistant_text(messages: list[AgentMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _render_prefix(messages: list[AgentMessage], *, max_turns: int) -> str:
    lines: list[str] = []
    ordinal = -1
    for message in messages:
        role = getattr(message, "role", "?")
        if isinstance(message, AssistantMessage):
            ordinal += 1
            tools = [
                block.name
                for block in message.content
                if isinstance(block, ToolCallBlock)
            ]
            text = " ".join(
                getattr(block, "text", "")
                for block in message.content
                if getattr(block, "text", None)
            )
            lines.append(f"[turn {ordinal}] assistant: {text[:400]} tools={tools}")
        else:
            text = _flatten_text(message)
            if text:
                lines.append(f"  {role}: {text[:300]}")
    return "\n".join(lines[-max_turns * 3 :])


def _flatten_text(message: AgentMessage) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None) or getattr(block, "content", None)
        if isinstance(text, str):
            parts.append(text)
    return " ".join(parts)


def _parse_action(value: Any) -> ActionType:
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return ActionType(value.upper())
    return ActionType.VERIFY


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    with contextlib.suppress(json.JSONDecodeError):
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    return None


def _placebo(reference: str) -> str:
    base = (
        "Please continue your investigation at your own pace. There is no need to "
        "rush; thoroughness is appreciated."
    )
    while len(base) < len(reference):
        base += " Continue carefully and methodically."
    return base[: max(len(reference), len(base))] if reference else base
