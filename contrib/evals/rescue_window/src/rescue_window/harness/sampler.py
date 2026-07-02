"""Prefix sampler (doc §6.4).

Prefixes must not come only from a critic's trigger points, or detector bias
leaks into the problem definition. We sample stratified fork points along each
trajectory and keep the sampling stratum (and weight) for analysis.

Fork points are keyed by ``turn_index`` — the 0-based assistant-turn ordinal,
which is exactly what the session store's fork resolves against (one turn
summary per assistant LLM turn; ``resolve_turn_leaf_id`` maps the ordinal to the
k-th assistant message). So the sampler derives everything from the message
list: the number of assistant turns is the trajectory length.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from agentm.core.abi import AgentMessage, AssistantMessage, ToolCallBlock

from ..model import ForkPoint, PrefixPoint
from .corpus import TrajectoryRef


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """Which strata to draw per trajectory."""

    relative_progress: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    include_pre_final: bool = True
    detect_events: tuple[str, ...] = ("first_tool_error", "repeated_tool_error")
    n_random: int = 0
    dense: bool = False
    dense_count: int | None = None  # evenly-spaced k points; None = every turn
    min_turn: int = 1  # never fork before the first assistant turn completes
    seed: int = 0


class PrefixSampler:
    """Draw stratified ``PrefixPoint``s from a trajectory's messages."""

    def __init__(
        self, policy: SamplingPolicy | None = None, *, final_tool: str | None = None
    ) -> None:
        self.policy = policy or SamplingPolicy()
        self.final_tool = final_tool  # scenario's terminal tool, for pre-final stratum

    def sample(
        self, ref: TrajectoryRef, messages: list[AgentMessage]
    ) -> list[PrefixPoint]:
        n_turns = _assistant_turn_count(messages)
        if n_turns <= self.policy.min_turn:
            return []
        eligible = list(range(self.policy.min_turn, n_turns))  # fork before final turn
        chosen: dict[int, str] = {}

        def _add(turn: int, stratum: str) -> None:
            if turn in eligible and turn not in chosen:
                chosen[turn] = stratum

        if self.policy.dense:
            for turn in _dense_points(eligible, self.policy.dense_count):
                _add(turn, "dense")
        for frac in self.policy.relative_progress:
            _add(_progress_turn(frac, n_turns, self.policy.min_turn), f"progress:{frac}")
        if self.policy.include_pre_final and self.final_tool:
            final = _final_turn(messages, self.final_tool)
            if final is not None and final - 1 >= self.policy.min_turn:
                _add(final - 1, "pre_final")
        for turn, name in _detect_event_turns(messages, self.policy.detect_events).items():
            _add(turn, f"event:{name}")
        if self.policy.n_random > 0:
            rng = random.Random(f"{ref.trajectory_id}:{self.policy.seed}")
            pool = [turn for turn in eligible if turn not in chosen]
            rng.shuffle(pool)
            for turn in pool[: self.policy.n_random]:
                _add(turn, "random")

        points: list[PrefixPoint] = []
        for turn, stratum in sorted(chosen.items()):
            points.append(
                PrefixPoint(
                    trajectory_id=ref.trajectory_id,
                    case_id=ref.case_id,
                    repository_id=ref.repository_id,
                    prefix_id=f"{ref.trajectory_id}#t{turn}",
                    fork_point=ForkPoint(turn_index=turn),
                    turn_index=turn,
                    progress=round(turn / n_turns, 4),
                    stratum=stratum,
                    event=stratum.split(":", 1)[1] if stratum.startswith("event:") else None,
                    weight=1.0,
                    remaining_budget={"turns_elapsed": turn, "total_turns": n_turns},
                    metadata={},
                )
            )
        return points


def _assistant_turn_count(messages: list[AgentMessage]) -> int:
    return sum(1 for message in messages if isinstance(message, AssistantMessage))


def _progress_turn(frac: float, n_turns: int, min_turn: int) -> int:
    turn = round(frac * (n_turns - 1))
    return max(min_turn, min(turn, n_turns - 1))


def _dense_points(eligible: list[int], dense_count: int | None) -> list[int]:
    if not eligible:
        return []
    if dense_count is None or dense_count >= len(eligible):
        return list(eligible)
    if dense_count <= 1:
        return [eligible[len(eligible) // 2]]
    step = (len(eligible) - 1) / (dense_count - 1)
    return sorted({eligible[round(i * step)] for i in range(dense_count)})


def _assistant_ordinal_of(messages: list[AgentMessage], predicate) -> int | None:  # type: ignore[no-untyped-def]
    ordinal = -1
    for message in messages:
        if isinstance(message, AssistantMessage):
            ordinal += 1
            if predicate(message):
                return ordinal
    return None


def _final_turn(messages: list[AgentMessage], final_tool: str) -> int | None:
    def _has_final(message: AssistantMessage) -> bool:
        return any(
            isinstance(block, ToolCallBlock) and block.name == final_tool
            for block in message.content
        )

    return _assistant_ordinal_of(messages, _has_final)


def _detect_event_turns(
    messages: list[AgentMessage], events: tuple[str, ...]
) -> dict[int, str]:
    """Best-effort event strata. Currently tool-error driven (doc §6.4)."""

    if not events:
        return {}
    error_turns: list[int] = []
    ordinal = -1
    for index, message in enumerate(messages):
        if not isinstance(message, AssistantMessage):
            continue
        ordinal += 1
        tail = messages[index + 1 : _next_assistant_index(messages, index)]
        if any(_message_signals_error(item) for item in tail):
            error_turns.append(ordinal)

    out: dict[int, str] = {}
    if "first_tool_error" in events and error_turns:
        out[error_turns[0]] = "first_tool_error"
    if "repeated_tool_error" in events:
        for prev, cur in zip(error_turns, error_turns[1:]):
            if cur - prev == 1:
                out[cur] = "repeated_tool_error"
                break
    return out


def _next_assistant_index(messages: list[AgentMessage], start: int) -> int:
    for index in range(start + 1, len(messages)):
        if isinstance(messages[index], AssistantMessage):
            return index
    return len(messages)


def _message_signals_error(message: AgentMessage) -> bool:
    """Heuristic: a tool-result-bearing message whose payload looks like an error."""

    for block in getattr(message, "content", []) or []:
        is_error = getattr(block, "is_error", None)
        if is_error is True:
            return True
        text = getattr(block, "text", None) or getattr(block, "content", None)
        if isinstance(text, str) and _looks_like_error(text):
            return True
    return False


def _looks_like_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ("error", "exception", "traceback", "failed", "binder error")
    )
