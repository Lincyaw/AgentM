"""Trailing-JSON extractor shared by the P0 ``agentm_bridge`` (REQ-007) and
the V0 cognitive-audit AgentM adapter (REQ-017).

Both consumers drive an AgentM session that ends with a JSON object as the
final assistant text and need the same lenient parser:

1. Prefer fenced ``​```json`` blocks (with or without language tag).
2. Fall back to the *last* balanced ``{...}`` / ``[...]`` block in the text.

Kept independent of either caller so the audit adapter doesn't need to
reach into the subprocess-CLI bridge layer (or vice versa).
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S)


def extract_json(text: str) -> Any | None:
    """Pull the last JSON object/array from ``text``.

    Returns ``None`` if no parseable block can be found. Tries fenced
    blocks first (most reliable when the model uses code fences), then
    falls back to the last balanced ``{...}`` / ``[...]`` span.
    """

    for snippet in reversed(_FENCE_RE.findall(text)):
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    snippet = _last_balanced_block(text)
    if snippet is None:
        return None
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _last_balanced_block(text: str) -> str | None:
    last: str | None = None
    stack: list[str] = []
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            stack.pop()
            if not stack and start >= 0:
                last = text[start : i + 1]
                start = -1
    return last


__all__ = ["extract_json"]
