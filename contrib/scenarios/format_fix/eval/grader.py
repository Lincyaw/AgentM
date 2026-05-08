"""Deterministic grader for the format_fix eval suite.

``grade(task, output) -> float`` is the contract ``tool_eval_run`` calls.
We extract the first JSON-parseable substring from ``output`` and
deep-equal against ``task['expected']['value']``. Score is 1.0 on
match, 0.0 otherwise.

The substring extraction tolerates the agent occasionally wrapping its
answer in commentary or markdown fences — we strip code fences and look
for the outermost ``{...}`` or ``[...]`` block.
"""

from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def grade(task: dict[str, Any], output: str) -> float:
    expected = (task.get("expected") or {}).get("value")
    if expected is None:
        return 0.0
    parsed = _parse_first_json(output)
    if parsed is None:
        return 0.0
    return 1.0 if parsed == expected else 0.0


def _parse_first_json(output: str) -> Any:
    if not isinstance(output, str):
        return None
    # 1. Strip ```json ... ``` fences if present, keep the inner block.
    m = _FENCE_RE.search(output)
    candidates: list[str] = []
    if m is not None:
        candidates.append(m.group(1).strip())
    candidates.append(output.strip())

    for candidate in candidates:
        # Try to parse the whole candidate first.
        for blob in _candidate_substrings(candidate):
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                continue
    return None


def _candidate_substrings(text: str) -> list[str]:
    """Yield the original string plus every outermost ``{...}`` /
    ``[...]`` substring. Supports the agent prepending or appending
    commentary."""
    out = [text]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            out.append(text[start : end + 1])
    return out
