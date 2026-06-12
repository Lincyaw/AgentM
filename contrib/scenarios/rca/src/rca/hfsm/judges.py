"""Judge port — pure module (not an atom).

Phase 2 C1 of the rca_hfsm scenario. Defines the ``Judge`` Protocol every
judgment atom implements, plus the supporting data types (``JudgeContext``,
``Verdict``) and helpers (``make_unclear``, ``canonical_cache_key``,
``JudgeToolSchema``).

Design references:

* :doc:`/.claude/designs/llm-native-judges.md` §3 (the port shape).
* :doc:`/.claude/designs/llm-native-judges.md` §3.4 (failure modes — the
  ``unclear`` fallback that judge atoms emit when the LLM is unreachable
  twice in a row).
* CLAUDE.md "no preset enums for subjective dimensions" — ``Verdict``'s
  three text fields are free-text. Callers may pattern-match on canonical
  strings the prompt requests (e.g. ``"satisfied"``, ``"refuted"``) but
  the field type is not ``Literal``-constrained.

This module is *not* an atom: it has no ``MANIFEST`` and no ``install``. It
is the same shape as ``schema.py`` / ``updates.py`` — pure data the judge
atoms import.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeContext:
    """Structured slice of L1 graph state passed to ``Judge.judge``.

    ``graph_slice`` is the structured payload the judge reasons over (e.g.
    ``{"prediction": ..., "checks": [...]}`` for ``judge.satisfied``).
    ``operands`` carries the specific inputs to the judgment (e.g.
    ``{"hypothesis_id": "H7"}`` for ``judge.coverage``).

    The fields are typed ``dict[str, Any]`` — each judge kind documents the
    concrete keys it expects in its prompt file under
    ``contrib/scenarios/rca_hfsm/prompts/judges/<kind>.md``.
    """

    graph_slice: dict[str, Any]
    operands: dict[str, Any]


@dataclass(frozen=True)
class Verdict:
    """The judgment a ``Judge`` returns.

    Per CLAUDE.md "no preset enums for subjective dimensions" every field is
    free-text. The prompt (loaded from ``prompts/judges/<kind>.md``) asks the
    LLM to emit one of the canonical strings for that judge, but the schema
    keeps the field a plain ``str``.
    """

    verdict: str
    reason: str
    confidence: str


@dataclass(frozen=True)
class JudgeToolSchema:
    """JSON Schema fragment for the ``submit_verdict`` tool.

    Returned by :func:`build_submit_verdict_tool_schema`. The judge atom
    constructs a ``Tool`` whose ``parameters`` mirror this schema and asks
    the LLM (via the prompt) to call ``submit_verdict`` exactly once.
    """

    name: str
    description: str
    parameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SUBMIT_VERDICT_TOOL_NAME = "submit_verdict"


def make_unclear(reason: str) -> Verdict:
    """Construct the canonical ``unclear`` verdict.

    Used by LLM-mode judges on the two-retry failure path (design §3.4) so
    every caller can pattern-match on the same shape regardless of why the
    judge bailed out.
    """

    return Verdict(verdict="unclear", reason=reason, confidence="none")


def canonical_cache_key(ctx: JudgeContext) -> str:
    """Stable ``sha256`` of the sorted-keys JSON encoding of ``ctx``.

    Used by both stub and LLM judge implementations as the per-session LRU
    cache key. ``default=str`` lets the encoder coerce dataclasses and
    other non-JSON-native values to their ``repr`` — judges only need a
    stable key, not a roundtrippable encoding.
    """

    payload = {
        "graph_slice": ctx.graph_slice,
        "operands": ctx.operands,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_submit_verdict_tool_schema(kind: str) -> JudgeToolSchema:
    """Return the canonical ``submit_verdict`` schema for ``kind``.

    The same tool name (``submit_verdict``) is used for every judge — the
    judge kind is communicated to the LLM through the prompt. Each judge
    builds a private :class:`agentm.core.abi.Tool` whose ``parameters``
    field comes from ``JudgeToolSchema.parameters``.
    """

    description = (
        f"Submit the verdict for the rca_hfsm '{kind}' judgment. Call "
        "this tool exactly once with a short free-text verdict, a "
        "free-text reason, and a free-text confidence label."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "description": (
                    "Free-text verdict label. Use one of the canonical "
                    "values listed in the prompt for this judge."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Free-text rationale grounded in the supplied graph slice.",
            },
            "confidence": {
                "type": "string",
                "description": "Free-text confidence label (e.g. 'high', 'medium', 'low').",
            },
        },
        "required": ["verdict", "reason", "confidence"],
        "additionalProperties": False,
    }
    return JudgeToolSchema(
        name=SUBMIT_VERDICT_TOOL_NAME,
        description=description,
        parameters=parameters,
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Judge(Protocol):
    """A single judgment surface.

    Implementations register themselves via
    ``api.set_service("rca.judge.<kind>", impl)`` from inside a atom.
    The service name's ``<kind>`` suffix matches the implementation's
    ``kind`` attribute.
    """

    kind: str

    def judge(self, context: JudgeContext) -> Verdict: ...


__all__ = [
    "Judge",
    "JudgeContext",
    "JudgeToolSchema",
    "SUBMIT_VERDICT_TOOL_NAME",
    "Verdict",
    "build_submit_verdict_tool_schema",
    "canonical_cache_key",
    "make_unclear",
]
