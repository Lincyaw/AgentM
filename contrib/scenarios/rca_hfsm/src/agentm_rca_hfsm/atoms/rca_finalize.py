"""``rca_finalize`` — coverage-gated termination tool (design §7 FINALIZE state).

Registers ``submit_final_report``. The only legal way out of the FINALIZE
state. On invocation:

* Queries L1 via ``rca.hgraph.read.get_unexplained_symptoms()``.
* If the list is non-empty: rejects the report (``ToolResult`` with
  ``is_error=True``). The text lists the unexplained symptom ids so the
  LLM has a concrete next step (record observations linking them, or
  refine the hypothesis to cover them).
* If empty: returns :class:`ToolTerminate` so the agent loop exits cleanly
  with reason ``rca_hfsm:final-report-submitted`` — analogous to the
  existing ``rca`` scenario's finalize atom.

The coverage check is the structural enforcement of acceptance #7. It is
NOT a free choice the LLM can override by re-phrasing the report — the
unexplained-symptoms view is a property of the L1 graph maintained by the
gate, not a prompt-level constraint.

§11 contract: stdlib + ``agentm.core.abi.*`` + ``agentm.extensions``. No
atom-to-atom imports; L1 is reached via ``api.get_service('rca.hgraph.read')``.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, ToolResult, ToolTerminate
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.abi.messages import TextContent
from agentm.extensions import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="rca_finalize",
    description=(
        "Terminate the rca_hfsm trace via submit_final_report; gated by the "
        "FINALIZE coverage check (every symptom must be linked to a "
        "satisfied prediction of a confirmed hypothesis)."
    ),
    registers=("tool:submit_final_report",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    requires=("rca_hgraph_store",),
)


_PARAMS: dict[str, Any] = {
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


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    read_handle = api.get_service("rca.hgraph.read")
    if read_handle is None:
        raise RuntimeError(
            "rca_finalize: rca.hgraph.read is not published; install "
            "rca_hgraph_store before this atom"
        )

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
        bus = api.events
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
                "symptom remains unexplained (coverage check). On acceptance "
                "the agent loop terminates."
            ),
            parameters=_PARAMS,
            fn=_submit,
            metadata={"idempotent": False},
        )
    )
