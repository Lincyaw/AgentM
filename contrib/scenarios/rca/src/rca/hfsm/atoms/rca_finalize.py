"""``rca_finalize`` — coverage-gated termination tool (design §7 FINALIZE state).

Registers ``submit_final_report``. The only legal way out of the FINALIZE
state. On invocation:

* Queries L1 via ``rca.hgraph.read.get_unexplained_symptoms()``.
* If the list is non-empty: rejects the report (``ToolResult`` with
  ``is_error=True``). The text lists the unexplained symptom ids so the
  LLM has a concrete next step (record observations linking them, or
  refine the hypothesis to cover them).
* If the structural coverage check passes, consults
  ``rca.judge.investigation_genuine`` (Phase 2 C5). The judge inspects
  the trajectory shape — symptoms recorded, hypotheses proposed,
  observations gathered, gate mutations applied vs downgraded, proposed
  final report — and returns ``genuine_investigation`` / ``speculation``
  / ``unclear``. Only ``genuine_investigation`` lets the report through;
  the other two verdicts produce a rejection whose text carries the
  judge's free-text reason verbatim so the orchestrator can course-
  correct (e.g. "the LLM didn't record any symptoms before concluding —
  please call record_symptom and gather evidence first").
* If both checks pass: returns :class:`ToolTerminate` so the agent loop
  exits cleanly with reason ``rca_hfsm:final-report-submitted``.

The C5 judge moves the "did the investigator actually investigate?"
question to LLM judgment. The C4 eval showed the structural coverage
check was vacuously true on empty symptom sets, letting the orchestrator
short-circuit the FSM by submitting reports without ever calling
``record_symptom``. The judge sees that the trajectory is empty and
returns ``speculation``; no hardcoded ``len(symptoms) > 0`` rule lives
in this atom.

§11 contract: stdlib + ``agentm.core.abi.*`` + ``agentm.extensions``. No
atom-to-atom imports; L1 is reached via ``api.get_service('rca.hgraph.read')``
and the judge via ``api.get_service('rca.judge.investigation_genuine')``.
"""

from __future__ import annotations

from typing import Any, Final

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest

from rca.hfsm.judges import JudgeContext

MANIFEST = ExtensionManifest(
    name="rca_finalize",
    description=(
        "Terminate the rca_hfsm trace via submit_final_report; gated by "
        "the FINALIZE structural coverage check and by the "
        "rca.judge.investigation_genuine judgment that the trajectory "
        "shows a genuine investigation (not speculation)."
    ),
    registers=("tool:submit_final_report",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    requires=("rca_hgraph_store",),
)

_PARAMS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "root_cause": {
            "type": "string",
            "description": (
                "Free-form statement of the root cause. Cite the confirmed "
                "hypothesis claim and the evidence chain."
            ),
        },
        "supporting_observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Observation ids (from L1) that anchor the root-cause claim."
            ),
            "default": [],
        },
        "refuted_alternatives": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Hypothesis ids the investigation refuted, so reviewers can "
                "audit the elimination path."
            ),
            "default": [],
        },
    },
    "required": ["root_cause"],
    "additionalProperties": False,
}

_GENUINE = "genuine_investigation"

def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])

def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)

def _hypothesis_summary(read_handle: Any) -> list[dict[str, Any]]:
    """Compose a structured summary of every hypothesis the store knows.

    Walks ``get_open_leaves`` + ``get_refuted_branches`` + ``get_confirmed``.
    Each entry carries id, claim, status, a short list of prediction
    summaries (claim + polarity + check count), and a total check count.
    The judge prompt reasons over this; no caller-side ``len()`` checks
    decide the verdict.
    """

    seen: dict[str, dict[str, Any]] = {}
    sources: list[list[Any]] = []
    for getter_name in ("get_open_leaves", "get_refuted_branches", "get_confirmed"):
        getter = getattr(read_handle, getter_name, None)
        if callable(getter):
            try:
                sources.append(list(getter()))
            except Exception:  # noqa: BLE001 — defensive on store quirks
                sources.append([])
    for batch in sources:
        for h in batch:
            hid = getattr(h, "id", None)
            if hid is None or hid in seen:
                continue
            predictions = getattr(h, "predictions", []) or []
            preds_summary: list[str] = []
            checks_count = 0
            for p in predictions:
                pclaim = getattr(p, "claim", "")
                polarity = getattr(p, "polarity", "")
                checks = list(getattr(p, "checks", []) or [])
                checks_count += len(checks)
                preds_summary.append(
                    f"[{polarity}] {pclaim} (checks={len(checks)})"
                )
            seen[hid] = {
                "id": hid,
                "claim": getattr(h, "claim", ""),
                "status": getattr(h, "status", ""),
                "predictions_summary": preds_summary,
                "checks_count": checks_count,
            }
    return list(seen.values())

def _build_judge_context(
    read_handle: Any,
    mutations: dict[str, int],
    args: dict[str, Any],
) -> JudgeContext:
    symptoms = list(read_handle.get_symptoms() or [])
    symptom_payload = [
        {
            "id": getattr(s, "id", ""),
            "text": getattr(s, "text", ""),
            "source": getattr(s, "source", ""),
        }
        for s in symptoms
    ]
    # Observation count: structural, not subjective. Same shape the
    # judge prompt expects.
    obs_count = 0
    obs_attr = getattr(read_handle, "_state", None)
    if obs_attr is not None:
        obs_count = len(getattr(obs_attr, "observations", []) or [])

    supporting = [
        str(x) for x in (args.get("supporting_observations") or [])
        if isinstance(x, str)
    ]
    refuted = [
        str(x) for x in (args.get("refuted_alternatives") or [])
        if isinstance(x, str)
    ]
    graph_slice: dict[str, Any] = {
        "symptom_count": len(symptoms),
        "symptoms": symptom_payload,
        "hypotheses": _hypothesis_summary(read_handle),
        "observations_count": obs_count,
        "gate_mutations": {
            "applied": int(mutations.get("applied", 0)),
            "downgraded": int(mutations.get("downgraded", 0)),
        },
        "final_report": {
            "root_cause": str(args.get("root_cause", "")),
            "supporting_observations": supporting,
            "refuted_alternatives": refuted,
        },
    }
    return JudgeContext(graph_slice=graph_slice, operands={})

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    read_handle = api.get_service("rca.hgraph.read")
    if read_handle is None:
        raise RuntimeError(
            "rca_finalize: rca.hgraph.read is not published; install "
            "rca_hgraph_store before this atom"
        )
    judge = api.get_service("rca.judge.investigation_genuine")
    if judge is None:
        raise RuntimeError(
            "rca_finalize: rca.judge.investigation_genuine is not "
            "published; install judge_investigation_genuine before this atom"
        )

    # Subscribe to ``rca.graph.mutated`` so the judge sees how many gate
    # operations the trace produced. Counts are scoped to this install;
    # tests reset the store between cases.
    mutation_counts: dict[str, int] = {"applied": 0, "downgraded": 0}

    def _on_mutation(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        kind = payload.get("kind")
        if kind in mutation_counts:
            mutation_counts[kind] += 1
        elif isinstance(kind, str):
            # Defensive: future mutation kinds land here without losing the
            # count. Judge prompt only references applied/downgraded.
            mutation_counts[kind] = mutation_counts.get(kind, 0) + 1

    bus = getattr(api, "events", None)
    if bus is not None:
        api.on("rca.graph.mutated", _on_mutation)

    async def _submit(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        root_cause = str(args.get("root_cause", "")).strip()
        if not root_cause:
            return _error("status=rejected reason=root_cause must be non-empty")
        unexplained = read_handle.get_unexplained_symptoms()
        if unexplained:
            ids = sorted(s.id for s in unexplained)
            return _error(
                "status=rejected reason=unexplained symptoms remain; "
                f"link via record_observation or refine: {ids}"
            )
        # C5: defer the "did this trace actually investigate?" question to
        # the LLM judge. No structural len()-style call-site checks here —
        # the judge prompt sees the full trajectory shape and decides.
        verdict = judge.judge(_build_judge_context(read_handle, mutation_counts, args))
        if verdict.verdict != _GENUINE:
            return _error(
                "status=rejected reason="
                f"judge=investigation_genuine verdict={verdict.verdict} "
                f"reason={verdict.reason}"
            )
        supporting = [
            str(x) for x in (args.get("supporting_observations") or [])
            if isinstance(x, str)
        ]
        refuted = [
            str(x) for x in (args.get("refuted_alternatives") or [])
            if isinstance(x, str)
        ]
        body = (
            "status=finalized root_cause=" + root_cause
            + " supporting=" + ",".join(supporting)
            + " refuted=" + ",".join(refuted)
        )
        # Emit a bus event so observability sinks (and the smoke test) can
        # pick up the final report content without parsing the tool result.
        if bus is not None:
            bus.emit_sync(
                "rca.final_report",
                {
                    "root_cause": root_cause,
                    "supporting_observations": supporting,
                    "refuted_alternatives": refuted,
                },
            )
        return ToolTerminate(
            result=_ok(body),
            reason="rca_hfsm:final-report-submitted",
        )

    api.register_tool(
        FunctionTool(
            name="submit_final_report",
            description=(
                "Submit the final root-cause analysis. Rejected if any "
                "symptom remains unexplained (structural coverage) OR if "
                "the investigation_genuine judge determines the trajectory "
                "shows speculation rather than verified investigation. On "
                "acceptance the agent loop terminates."
            ),
            parameters=_PARAMS,
            fn=_submit,
            metadata={"idempotent": False},
        )
    )
