"""Parameter schemas + descriptions for the rca_hfsm evidence tools.

Pure data module (no ``MANIFEST``, no ``install``). Lives alongside
``schema.py`` / ``updates.py`` so the evidence-tools atom stays under the
single-file size envelope. Importing pure scenario-local modules from
an atom is permitted by the atom-to-atom rule (rule names only other
atom files; pure modules are pure data).

Two contracts the LLM sees are load-bearing here:

* ``attach_check.verdict_proposal``'s description steers the model toward
  the gate's free-text sentinels (``triggered`` / ``supports`` /
  ``steelman``) without making the field an enum (CLAUDE.md "no preset
  enums for subjective dimensions").
* ``propose_hypothesis.predictions``'s description repeats the §6.2
  negative-prediction-required rule so the LLM gets the rule both from
  the gate's rejection reason AND from the parameter schema.

Changes to the descriptions are a public-contract change — they're what
the LLM reads on every call.
"""

from __future__ import annotations

from typing import Any, Final


_STR: Final[dict[str, str]] = {"type": "string"}
_STR_ARR: Final[dict[str, Any]] = {"type": "array", "items": _STR}


VERDICT_GUIDANCE: Final[str] = (
    "Free-text verdict. For NEGATIVE predictions, state whether observations "
    "'triggered' or did 'not trigger' the prediction. For POSITIVE predictions, "
    "state whether observations 'support' or 'do not support' the prediction. "
    "For steelman-mode checks, mention 'steelman' so the gate's §7.2 refute "
    "path recognises the attempt — word-boundary matched in updates.py."
)


PARAMS: Final[dict[str, dict[str, Any]]] = {
    "record_symptom": {
        "type": "object",
        "properties": {
            "text": {**_STR, "description": "Free-form observed problem."},
            "source": {
                **_STR,
                "description": "Origin tag (free-text).",
                "default": "user_intake",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    "record_observation": {
        "type": "object",
        "properties": {
            "text": {**_STR, "description": "Rendered evidence (cited verbatim)."},
            "source_tool_call": {
                **_STR,
                "description": "tool_call_id that produced this fact.",
            },
            "related_symptoms": {**_STR_ARR, "default": []},
            "related_predictions": {**_STR_ARR, "default": []},
        },
        "required": ["text", "source_tool_call"],
        "additionalProperties": False,
    },
    "propose_hypothesis": {
        "type": "object",
        "properties": {
            "claim": {**_STR, "description": "Free-form hypothesis statement."},
            "rationale": {**_STR, "default": ""},
            "predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": _STR,
                        "polarity": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                        },
                        "test_plan": _STR,
                    },
                    "required": ["claim", "polarity"],
                    "additionalProperties": False,
                },
                "description": (
                    "Each Hypothesis MUST declare at least one polarity='negative' "
                    "prediction or the gate rejects it as unfalsifiable (design §6.2)."
                ),
            },
        },
        "required": ["claim", "predictions"],
        "additionalProperties": False,
    },
    "attach_check": {
        "type": "object",
        "properties": {
            "hypothesis_id": _STR,
            "prediction_id": _STR,
            "worker_session_id": {
                **_STR,
                "description": (
                    "Distinct values across positive checks satisfy the §7.1 "
                    "independence requirement at confirm time."
                ),
            },
            "mode": {
                **_STR,
                "description": (
                    "Free-text mode label. Use 'verify' for normal falsification "
                    "checks; use 'steelman' when the worker actively tried to "
                    "FIND supporting evidence."
                ),
                "default": "verify",
            },
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": _STR,
                        "source_tool_call": _STR,
                        "related_symptoms": _STR_ARR,
                        "related_predictions": _STR_ARR,
                    },
                    "required": ["text", "source_tool_call"],
                    "additionalProperties": False,
                },
                "description": (
                    "Citable facts the worker found. These — NOT the interpretation — "
                    "drive the orchestrator's re-derived update decision (design §6)."
                ),
            },
            "interpretation": {
                "type": "object",
                "properties": {
                    "proposed_update": _STR,
                    "reasoning": _STR,
                    "confidence": _STR,
                },
                "required": ["proposed_update", "reasoning", "confidence"],
                "additionalProperties": False,
                "description": (
                    "Worker's advisory verdict. Recorded in the trace but does not "
                    "enter the graph; orchestrator re-derives from observations alone."
                ),
            },
            "verdict_proposal": {
                **_STR,
                "description": VERDICT_GUIDANCE,
                "default": "",
            },
        },
        "required": [
            "hypothesis_id",
            "prediction_id",
            "worker_session_id",
            "observations",
            "interpretation",
        ],
        "additionalProperties": False,
    },
    "propose_update": {
        "type": "object",
        "properties": {
            "op": {
                **_STR,
                "description": (
                    "One of: confirm, refute, refine, split, merge, supersede, suspend."
                ),
            },
            "target_id": _STR,
            "reason": {**_STR, "default": ""},
            "child": {
                "type": "object",
                "description": "Payload for refine / supersede.",
            },
            "children": {"type": "array", "items": {"type": "object"}, "default": []},
            "sources": {**_STR_ARR, "default": []},
        },
        "required": ["op"],
        "additionalProperties": False,
    },
}


DESCRIPTIONS: Final[dict[str, str]] = {
    "record_symptom": (
        "Record an observed problem on the RCA hypothesis graph. INTAKE-stage tool."
    ),
    "record_observation": (
        "Append a citable fact to the ObservationLog. Use related_symptoms / "
        "related_predictions to link the fact to the graph."
    ),
    "propose_hypothesis": (
        "Propose a hypothesis with its predictions. At least one prediction MUST "
        "have polarity='negative' or the gate rejects it (design §6.2)."
    ),
    "attach_check": (
        "Record a CheckResult on a prediction. Observations are facts (ingested "
        "into L1); interpretation is advisory (trace only). See verdict_proposal "
        "description for the gate's sentinel vocabulary."
    ),
    "propose_update": (
        "Escape hatch for explicit graph operators. Routes through the same gate "
        "as the dedicated tools and surfaces downgrade-to-refine with reasons."
    ),
}
