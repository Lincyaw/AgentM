"""Shared utilities for tool modules: token budgeting, JSON serialization."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

# Common token budget for all tool result payloads.
TOKEN_LIMIT = 5000


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text* using a rough char/3 heuristic."""
    return (len(text) + 2) // 3


def serialize_for_json(obj: Any) -> Any:
    """Recursively convert datetime / NaN / Inf values to JSON-safe equivalents.

    - ``datetime`` -> ISO-format string
    - ``float('nan')`` / ``float('inf')`` -> ``None``
    - ``dict`` and ``list`` are traversed recursively
    - Everything else is returned unchanged.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj


def enforce_token_budget(payload: str, context: str) -> str:
    """Enforce token budget on a JSON tool-result string.

    When *payload* fits within ``TOKEN_LIMIT``, returns it unchanged.
    When it exceeds the limit, truncates the data and appends metadata
    so the agent can refine its query.

    Handles three shapes:
    - **list** — keeps a prefix of rows and reports total vs returned counts.
    - **dict** — truncates each list-valued key independently.
    - **scalar / unknown** — returns a budget-exceeded error object.
    """
    if estimate_tokens(payload) <= TOKEN_LIMIT:
        return payload

    estimated_tokens = estimate_tokens(payload)
    ratio = TOKEN_LIMIT / estimated_tokens
    parsed = json.loads(payload)

    if isinstance(parsed, list):
        original_count = len(parsed)
        keep = max(1, int(original_count * ratio * 0.8))
        truncated = parsed[:keep]
        result = {
            "_truncated": True,
            "_total_rows": original_count,
            "_rows_returned": keep,
            "_context": context,
            "_suggestion": (
                "Add more filters, a narrower time range, or specify a LIMIT "
                "to get complete data."
            ),
            "data": truncated,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    if isinstance(parsed, dict):
        # For dict payloads (e.g. call graphs with {nodes, edges}), truncate each list
        truncated_dict: dict[str, Any] = {}
        meta: dict[str, Any] = {"_truncated": True, "_context": context}
        for k, v in parsed.items():
            if isinstance(v, list):
                keep = max(1, int(len(v) * ratio * 0.8))
                truncated_dict[k] = v[:keep]
                meta[f"_{k}_total"] = len(v)
                meta[f"_{k}_returned"] = keep
            else:
                truncated_dict[k] = v
        meta["_suggestion"] = "Add more filters or a narrower time range."
        truncated_dict.update(meta)
        return json.dumps(truncated_dict, ensure_ascii=False, indent=2)

    # Scalar or unknown — return warning only
    warning = {
        "error": "Result exceeds token budget",
        "context": context,
        "estimated_tokens": estimated_tokens,
        "token_limit": TOKEN_LIMIT,
        "suggestion": "Add more filters or reduce the LIMIT.",
    }
    return json.dumps(warning, ensure_ascii=False, indent=2)
