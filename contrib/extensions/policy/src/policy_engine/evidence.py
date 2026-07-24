"""Targeted evidence gathering for critic verification.

Given a checklist item's trigger predicates, finds the relevant turns
from the trajectory and formats them as critic input.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from .pg_query import PgQuerySource
from .triggers import ChecklistItem

_PREDICATE_RE = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")


def extract_predicates(trigger_expr: str) -> frozenset[str]:
    """Extract predicate names from a trigger expression."""
    if trigger_expr == "always":
        return frozenset()
    tokens = _PREDICATE_RE.findall(trigger_expr)
    return frozenset(t for t in tokens if t not in {"AND", "OR", "NOT"})


def gather_evidence(
    source: PgQuerySource,
    item: ChecklistItem,
    *,
    schema: str = "harbor_live",
    max_turns: int = 10,
) -> str:
    """Build a focused evidence prompt for one checklist item.

    Finds turns whose tagger annotations match the item's trigger
    predicates, then loads those turns' full content from the trajectory.
    """
    predicates = extract_predicates(item.gate.trigger)

    # Find turns that have any of the trigger predicates
    if predicates:
        relevant_turns = source.query(
            "SELECT turn_index, tags FROM policy.turn_annotations "
            "WHERE session_id = %(session_id)s "
            "AND tags && %(predicates)s "
            "ORDER BY turn_index",
            {"session_id": source.session_id, "predicates": list(predicates)},
        )
    else:
        # trigger=always: take the last few turns
        relevant_turns = source.query(
            "SELECT turn_index, tags FROM policy.turn_annotations "
            "WHERE session_id = %(session_id)s "
            "ORDER BY turn_index DESC LIMIT %(limit)s",
            {"session_id": source.session_id, "limit": max_turns},
        )
        relevant_turns = list(reversed(relevant_turns))

    if not relevant_turns:
        return ""

    turn_indices = [r[0] for r in relevant_turns[:max_turns]]

    # Load turn content from trajectory
    parts = [f"Review question:\n{item.check}\n"]
    parts.append(f"Evidence from session {source.session_id}:\n")

    for turn_idx in turn_indices:
        turn_rows = source.query(
            f"SELECT turn_json FROM {schema}.agentm_trajectory_turns "  # noqa: S608
            "WHERE session_id = %(session_id)s AND turn_index = %(turn_idx)s",
            {"session_id": source.session_id, "turn_idx": turn_idx},
        )
        if not turn_rows:
            continue

        turn_json = turn_rows[0][0]
        turn_text = _format_turn(turn_idx, turn_json)
        if turn_text:
            parts.append(turn_text)

    return "\n".join(parts)


def _format_turn(turn_index: int, turn_json: object) -> str:
    if not isinstance(turn_json, Mapping):
        return ""

    parts = [f"\n--- Turn {turn_index} ---"]

    response = turn_json.get("response")
    if isinstance(response, Mapping):
        for block in response.get("content", []):
            if isinstance(block, Mapping) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(f"Agent reasoning: {text[:1000]}")

    for tr in turn_json.get("tool_results", []):
        if not isinstance(tr, Mapping):
            continue
        call = tr.get("call", {})
        result = tr.get("result", {})
        if not isinstance(call, Mapping) or not isinstance(result, Mapping):
            continue

        name = call.get("name", "?")
        args = call.get("arguments", {})
        is_error = result.get("is_error", False)
        result_text = ""
        for block in result.get("content", []):
            if isinstance(block, Mapping) and block.get("type") == "text":
                result_text += block.get("text", "")[:500]

        args_str = json.dumps(args, default=str)[:300]
        parts.append(f"Tool: {name}({args_str})")
        if result_text:
            status = "ERROR" if is_error else "ok"
            parts.append(f"  Result ({status}): {result_text[:500]}")

    return "\n".join(parts)
