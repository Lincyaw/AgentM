"""Deterministic grader for the format_fix eval suite.

``grade(task, output) -> GradeResult`` is the contract ``tool_eval_run``
calls. We extract the first JSON-parseable substring from ``output`` and
deep-equal against ``task['expected']['value']``. Score is 1.0 on
match, 0.0 otherwise.

Returns the μ_f feedback shape (design §3.2): score + diagnostic
``feedback_text`` (and ``module_feedback`` keyed on the production atom
the grader can plausibly fault, ``tool_normalize_json``). The
``tool_eval_run`` adapter accepts bare floats too, so older third-party
graders keep working — but we use the rich shape here so B-2 / B-5 have
real signal to consume.

The substring extraction tolerates the agent occasionally wrapping its
answer in commentary or markdown fences — we strip code fences and look
for the outermost ``{...}`` or ``[...]`` block.
"""

from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def grade(task: dict[str, Any], output: str) -> dict[str, Any]:
    expected = (task.get("expected") or {}).get("value")
    if expected is None:
        return _result(
            0.0,
            feedback_text=(
                "task is missing expected.value; cannot grade — fix the "
                "eval YAML"
            ),
            module="eval_suite",
        )
    parsed = _parse_first_json(output)
    if parsed is None:
        return _result(
            0.0,
            feedback_text=(
                "agent output did not contain a JSON-parseable substring; "
                "tool_normalize_json likely produced invalid JSON or the "
                "agent dropped the tool's reply on the floor"
            ),
            module="tool_normalize_json",
        )
    if parsed != expected:
        return _result(
            0.0,
            feedback_text=(
                f"parsed JSON did not deep-equal expected: got "
                f"{json.dumps(parsed, sort_keys=True)!r} expected "
                f"{json.dumps(expected, sort_keys=True)!r}"
            ),
            module="tool_normalize_json",
        )
    return _result(1.0, feedback_text="match", module="tool_normalize_json")


def _result(
    score: float, *, feedback_text: str, module: str
) -> dict[str, Any]:
    return {
        "score": score,
        "dimensions": {},
        "feedback_text": feedback_text,
        "module_feedback": {module: feedback_text} if feedback_text else {},
    }


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
