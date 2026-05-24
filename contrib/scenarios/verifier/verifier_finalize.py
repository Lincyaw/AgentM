"""Verifier-scenario termination protocol.

The verifier ends by calling ``submit_propagation_report``. Unlike the
RCA scenario (which submits a guess at the unknown root cause), the
verifier already knows what was injected — it submits:

* ``injection_effective`` — did the injection actually materialize? One
  of ``true`` / ``false`` / ``ambiguous`` (free-text rationale required).
* ``injection_evidence`` — SQL-backed claims that the injection target
  itself shows the expected anomaly inside the abnormal window.
* ``slo_impact`` — user-visible service-level regressions observed
  during the abnormal window.
* ``propagation_nodes`` — every component whose state changed because
  of the injection (NOT just the injection target). Each node carries a
  ``component`` (``container|name``, ``service|name``, ``span|name``,
  …), a free-text ``state`` description, the affected window, and SQL
  evidence. Free-text by design — see [[feedback_no_preset_subjective_labels]].
* ``propagation_edges`` — directed edges ``from`` → ``to`` with a
  free-text ``mechanism`` (e.g. "JDBC connection failure surfaces as
  500s in callers") and SQL evidence per edge.

Schema correctness is enforced by Pydantic here; vocabulary alignment
(component IDs must match strings observed in the parquets) is
enforced in the prompt.

Mirrors the rca ``finalize`` atom's loop-keepalive behaviour: while
the agent has not yet submitted, voluntarily ending a turn with
prose-only is rejected via an ``Inject``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.lib import pydantic_to_openai_tool_schema

from agentm.core.abi import (
    FunctionTool,
    ToolResult,
    ToolTerminate,
)
from agentm.core.abi.messages import TextContent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="verifier_finalize",
    description=(
        "Termination protocol: the verifier must call "
        "submit_propagation_report to end the investigation."
    ),
    registers=("tool:submit_propagation_report",),
)


# ---------------------------------------------------------------------------
# Pydantic schema — free-text where reasonable interpretations differ.
# ---------------------------------------------------------------------------


# Models declare ``extra="forbid"`` so runtime validation matches the
# ``additionalProperties: false`` advertised in the JSON schema, and
# every field is required (no defaults) so OpenAI strict mode is happy.
# Empty arrays are allowed at the type level (``list[...]`` accepts
# ``[]``); the agent passes ``[]`` explicitly when a section has no
# entries. ``pydantic_to_openai_tool_schema`` normalises the emitted
# schema (inlines ``$defs``, strips titles, forces strict mode).
_STRICT = ConfigDict(extra="forbid")


class SqlEvidence(BaseModel):
    model_config = _STRICT
    sql: str = Field(description="DuckDB SQL re-executable against case parquets.")
    claim: str = Field(description="<=25-word natural-language assertion the SQL backs.")


class PropagationEdge(BaseModel):
    """One service-to-service hop in the fault impact graph.

    Both ``from_service`` and ``to_service`` are bare service names as
    they appear in the parquet ``service_name`` column. The edge
    direction is fault-impact (upstream-failing → downstream-affected),
    NOT request-call direction.

    Synthetic traffic generators (``loadgenerator``, ``locust``,
    ``wrk2``, ``dsb-wrk2``, ``k6``) are not real services and must NOT
    appear in either ``from_service`` or ``to_service``.
    """

    model_config = _STRICT
    from_service: str = Field(
        description="Service whose degradation causes the downstream change. "
        "Do NOT use synthetic load generators (loadgenerator, locust, wrk2, "
        "dsb-wrk2, k6) — those are not services."
    )
    to_service: str = Field(
        description="Service that degraded because of ``from_service``. "
        "Do NOT use synthetic load generators."
    )
    evidence: list[SqlEvidence] = Field(
        description="At least one SQL+claim showing to_service's "
        "behaviour in the abnormal window differs from normal AND the "
        "timing is consistent with from_service being the cause."
    )


class VerifierReport(BaseModel):
    """Service-level fault propagation report.

    The verifier outputs the directed graph of services affected by a
    known injection, plus an effectiveness verdict on the injection
    itself. Lower-granularity targets (container / pod / span) are
    intentionally NOT part of the contract — only services and the
    SQL evidence that proves each propagation hop.
    """

    model_config = _STRICT
    injection_effective: Literal["true", "false", "ambiguous"]
    effectiveness_rationale: str = Field(
        description="Why effective / not / ambiguous, citing the abnormal "
        "vs normal window comparison."
    )
    propagation_edges: list[PropagationEdge] = Field(
        description="Service-to-service fault propagation hops. Empty "
        "list is acceptable when injection_effective='false'."
    )


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


@dataclass
class _State:
    submitted: bool = False


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    state = _State()
    del config

    async def _submit(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        try:
            report = VerifierReport.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "verifier_contract_validation_failed",
                                "detail": exc.errors(include_url=False),
                            },
                            ensure_ascii=False,
                        ),
                    )
                ],
                is_error=True,
            )

        state.submitted = True
        return ToolTerminate(
            result=ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=report.model_dump_json(by_alias=True),
                    )
                ]
            ),
            reason="verifier:propagation-report-submitted",
        )

    api.register_tool(
        FunctionTool(
            name="submit_propagation_report",
            description=(
                "Submit the verifier's final report: an effectiveness "
                "verdict on the injection itself plus the service-level "
                "fault-propagation graph. Each edge is a directed "
                "service-to-service hop in the fault-impact direction "
                "(upstream-failing → downstream-affected) with at least "
                "one SQL+claim evidence pair. Service names must match "
                "strings observed in the parquet service_name column."
            ),
            parameters=pydantic_to_openai_tool_schema(VerifierReport),
            fn=_submit,
        )
    )



__all__ = ["MANIFEST", "install"]
