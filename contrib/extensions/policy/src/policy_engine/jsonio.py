"""Reading JSON back out of a model reply.

Asking for JSON does not reliably get only JSON. A model with something to say
says it — before the object, after it, or wrapped in a fence — and a strict
``json.loads`` then drops the whole answer. On one 11-session run that cost 19%
of turn annotations, and the losses were not spread evenly: they landed on the
turns where the agent had written the most, which are the turns worth reading.
"""

from __future__ import annotations


def json_object(text: str) -> str:
    """The outermost JSON object in *text*, however it was wrapped."""
    clean = text.strip()
    if clean.startswith("```"):
        body = clean.split("\n", 1)[-1]
        clean = body.rsplit("```", 1)[0].strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end <= start:
        return clean
    return clean[start : end + 1]


__all__ = ["json_object"]
