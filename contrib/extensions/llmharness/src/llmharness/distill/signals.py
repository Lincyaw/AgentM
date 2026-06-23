"""Deterministic process-reward signals for child rollouts.

This module is the **process-reward contract** between llmharness (which
captures child rollouts) and external trainers (e.g. ``rca-autorl``'s
GRPO/PPO/DPO drivers). It is intentionally:

* **Pure** — no I/O, no LLM, no atom-infra imports. Operates only on
  plain ``dict``-shaped ``ToolEvent`` lists so an external trainer can
  pull it in without dragging the audit-index machinery along.
* **Deterministic** — same inputs ⇒ same outputs; safe to call inside
  a training loop's reward function.
* **Defensive** — empty / malformed input collapses to an all-zero
  dict rather than raising, so a partially-observed rollout never
  crashes the trainer.

Two phase faces:

* :func:`extractor_process_reward` — for extractor-child rollouts; the
  terminal-good signal is a successful ``finalize_extraction`` call.
* :func:`auditor_process_reward` — for auditor-child rollouts; the
  terminal-good signal is a successful ``submit_verdict`` call.

The composite scalars are bounded so they sit in a stable range for the
trainer's reward normalization. The exact weights are baked here — if
they shift, that's a breaking change that bumps the package version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:  # avoid pulling agentm.core.abi at import time
    from ..replay.engine import PhaseResult

__all__ = [
    "ToolEvent",
    "auditor_process_reward",
    "extractor_process_reward",
    "tool_events_from_phase_result",
]

_EXTRACTOR_FINALIZE_TOOL = "finalize_extraction"
_AUDITOR_SUBMIT_TOOL = "submit_verdict"

class ToolEvent(TypedDict):
    """One tool_call / tool_result pair observed during a child rollout.

    The trainer constructs these from a replay record's
    ``raw_assistant_messages`` (tool_call side) joined with the matching
    tool_result. ``is_error`` is the tool_result's error flag;
    ``error_text`` carries the result text when ``is_error`` is true.
    """

    tool_name: str
    args: dict[str, object]
    is_error: bool
    error_text: str | None

def _success_rate(events: list[ToolEvent]) -> float:
    if not events:
        return 0.0
    ok = sum(1 for e in events if not e.get("is_error", False))
    return ok / len(events)

def _terminal_success(events: list[ToolEvent], expected_tool: str) -> int:
    """1 iff the last event is ``expected_tool`` AND it succeeded."""
    if not events:
        return 0
    last = events[-1]
    if last.get("tool_name") != expected_tool:
        return 0
    if last.get("is_error", False):
        return 0
    return 1

def _efficiency_penalty(events: list[ToolEvent], budget: int) -> float:
    if budget <= 0:
        return 0.0
    return min(1.0, len(events) / budget)

def extractor_process_reward(
    tool_events: list[ToolEvent],
    *,
    max_steps_budget: int = 32,
) -> dict[str, float]:
    """Deterministic process reward for one extractor child rollout.

    Composite formula::

        reward = 0.5 * finalize_success
               + 0.3 * witness_pass_rate
               - 0.2 * efficiency_penalty

    Bounded in ``[-0.2, 0.8]``. Empty input → all-zero dict.

    Returns:
        Dict with keys ``reward``, ``witness_pass_rate``,
        ``finalize_success``, ``efficiency_penalty``.
    """
    if not tool_events:
        return {
            "reward": 0.0,
            "witness_pass_rate": 0.0,
            "finalize_success": 0.0,
            "efficiency_penalty": 0.0,
        }
    witness_pass_rate = _success_rate(tool_events)
    finalize_success = float(_terminal_success(tool_events, _EXTRACTOR_FINALIZE_TOOL))
    efficiency_penalty = _efficiency_penalty(tool_events, max_steps_budget)
    reward = 0.5 * finalize_success + 0.3 * witness_pass_rate - 0.2 * efficiency_penalty
    return {
        "reward": reward,
        "witness_pass_rate": witness_pass_rate,
        "finalize_success": finalize_success,
        "efficiency_penalty": efficiency_penalty,
    }

def auditor_process_reward(
    tool_events: list[ToolEvent],
    *,
    max_steps_budget: int = 8,
) -> dict[str, float]:
    """Deterministic process reward for one auditor child rollout.

    Composite formula::

        reward = 0.7 * verdict_submitted
               + 0.2 * schema_valid_rate
               - 0.1 * efficiency_penalty

    Bounded in ``[-0.1, 1.0]``. Empty input → all-zero dict.

    Returns:
        Dict with keys ``reward``, ``schema_valid_rate``,
        ``verdict_submitted``, ``efficiency_penalty``.
    """
    if not tool_events:
        return {
            "reward": 0.0,
            "schema_valid_rate": 0.0,
            "verdict_submitted": 0.0,
            "efficiency_penalty": 0.0,
        }
    schema_valid_rate = _success_rate(tool_events)
    verdict_submitted = float(_terminal_success(tool_events, _AUDITOR_SUBMIT_TOOL))
    efficiency_penalty = _efficiency_penalty(tool_events, max_steps_budget)
    reward = 0.7 * verdict_submitted + 0.2 * schema_valid_rate - 0.1 * efficiency_penalty
    return {
        "reward": reward,
        "schema_valid_rate": schema_valid_rate,
        "verdict_submitted": verdict_submitted,
        "efficiency_penalty": efficiency_penalty,
    }

def tool_events_from_phase_result(result: PhaseResult) -> list[ToolEvent]:
    """Walk ``result.messages`` and pair tool_call with tool_result.

    Returns the canonical :class:`ToolEvent` list that
    :func:`extractor_process_reward` / :func:`auditor_process_reward`
    consume. Pairing prefers ``ToolResultBlock.tool_call_id`` match;
    when an id can't be matched (older providers / synthetic messages)
    we fall back to per-tool-call FIFO ordering across the trajectory.

    Skips system / user / assistant-text / thinking content. Defensive:
    empty ``result.messages`` returns ``[]``. Tool calls without any
    matching result are still emitted with ``is_error=False`` and
    ``error_text=None`` so the rollout's step count stays honest.

    The pairing strategy intentionally mirrors what the live audit loop
    does — see :func:`llmharness.runtime.child_collect.terminal_tool_arguments`;
    here we keep every tool_call, not just the terminal one.
    """
    # Local import: keeping ``agentm.core.abi`` out of module-load time
    # so train_signals stays cheap to import in the trainer's hot path.
    from agentm.core.abi import (
        AssistantMessage,
        TextContent,
        ToolCallBlock,
        ToolResultBlock,
        ToolResultMessage,
    )

    messages = result.messages
    if not messages:
        return []

    # First pass: collect tool calls in order, indexed by their id.
    calls: list[ToolCallBlock] = []
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock):
                calls.append(block)

    if not calls:
        return []

    # Second pass: collect results, keyed by tool_call_id when possible.
    # ``results_by_id`` maps id -> (is_error, error_text); we keep the last
    # match if a provider somehow emits the same id twice.
    results_by_id: dict[str, tuple[bool, str | None]] = {}
    # Fallback queue of (is_error, error_text) when a result block has no
    # matching id; consumed in FIFO order by leftover unpaired calls.
    orphan_results: list[tuple[bool, str | None]] = []
    seen_call_ids = {c.id for c in calls if c.id}
    for msg in messages:
        if not isinstance(msg, ToolResultMessage):
            continue
        for r_block in msg.content:
            if not isinstance(r_block, ToolResultBlock):
                continue
            text_parts = [c.text for c in r_block.content if isinstance(c, TextContent)]
            error_text = "\n".join(text_parts) if r_block.is_error and text_parts else None
            entry = (bool(r_block.is_error), error_text)
            if r_block.tool_call_id and r_block.tool_call_id in seen_call_ids:
                results_by_id[r_block.tool_call_id] = entry
            else:
                orphan_results.append(entry)

    out: list[ToolEvent] = []
    for call in calls:
        if call.id and call.id in results_by_id:
            is_error, error_text = results_by_id[call.id]
        elif orphan_results:
            is_error, error_text = orphan_results.pop(0)
        else:
            is_error, error_text = False, None
        out.append(
            ToolEvent(
                tool_name=call.name,
                args=dict(call.arguments),
                is_error=is_error,
                error_text=error_text,
            )
        )
    return out
