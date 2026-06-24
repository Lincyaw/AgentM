"""Compaction + retry-context helpers for gate-rejected discovery agents.

A gate rejection spawns a fresh child session; these helpers summarize the
previous attempt compactly so the retry sees a map of what was already
checked without replaying an entire trajectory.
"""
from __future__ import annotations

import json
from typing import Any

from .schema import GateDecision


def truncate_text(value: object, limit: int = 1600) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def compact_query(query: object) -> dict[str, str]:
    if not isinstance(query, dict):
        return {}
    statement = ""
    raw_statement = query.get("statement")
    if raw_statement is not None:
        statement = truncate_text(raw_statement, 1200)
    language = query.get("language")
    return {
        "language": str(language or "sql"),
        "statement": statement,
    }


def compact_evidence_item(item: object) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"value": truncate_text(item, 400)}
    compact: dict[str, Any] = {}
    if item.get("explanation"):
        compact["explanation"] = truncate_text(item["explanation"], 500)
    query = compact_query(item.get("query"))
    if query:
        compact["query"] = query
    return compact


def compact_agent_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep retry context useful without replaying an entire trajectory."""
    compact: dict[str, Any] = {}
    for key in (
        "verdict",
        "effect_target",
        "predicate",
        "claim",
        "rationale",
        "investigation_coverage",
    ):
        value = result.get(key)
        if value:
            compact[key] = value if isinstance(value, dict) else truncate_text(value)
    evidence = result.get("evidence")
    if isinstance(evidence, list):
        compact["evidence"] = [
            compact_evidence_item(item) for item in evidence[:10]
        ]
        if len(evidence) > 10:
            compact["evidence_truncated"] = len(evidence) - 10
    relationship = result.get("relationship")
    if isinstance(relationship, dict):
        compact["relationship"] = compact_evidence_item(relationship)
    return compact


def build_retry_context(
    *,
    base_context: str,
    label: str,
    submitted_result: dict[str, Any],
    gate: GateDecision,
) -> str:
    """Build the context passed to the next clean retry session."""
    payload = {
        "failed_attempt": label,
        "previous_submitted_result": compact_agent_result(submitted_result),
        "gate_decision": {
            "accepted": gate.accepted,
            "confidence": gate.confidence,
            "missing_checks": gate.missing_checks,
            "rationale": truncate_text(gate.rationale),
            "retry_prompt": truncate_text(gate.retry_prompt or "", 2400),
        },
    }
    block = (
        "## Previous attempt context\n"
        "This is a new child session, but the previous attempt is summarized below. "
        "Use it as a map of what was already checked and why gate rejected it. "
        "Do not simply repeat the previous evidence; rerun or repair SQL as needed, "
        "then fill only the missing checks.\n"
        "```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        + "\n```"
    )
    return (base_context + "\n\n" if base_context else "") + block
