"""Verifier-scenario termination protocol.

The verifier ends by calling ``submit_propagation_report``. Unlike the
RCA scenario (which submits a guess at the unknown root cause), the
verifier already knows what was injected — it submits:

* ``injections`` — one verdict per injected fault (a case may have
  several; judge each independently — all, some, or none may engage).
  Each carries ``target_service``, ``fault_kind``, ``verdict``
  (true/false/ambiguous) and a ``rationale`` citing the target's own
  normal-vs-abnormal data.
* ``propagation_nodes`` — every service judged to have degraded. Each
  node carries ``symptom_evidence`` — 1..N SQL+claim pairs from diverse
  signals (traces, metrics, logs) showing it got worse vs baseline. A
  flat/improved metric is not a symptom, so such a service is not a node.
* ``propagation_edges`` — directed fault-impact hops ``from_service`` →
  ``to_service`` between nodes. Each carries a ``relationship_sql``
  proving the two services are directly connected (trace parent/child
  call, either direction, or a shared k8s deployment/node) plus a
  ``claim``. Both endpoints must also be ``propagation_nodes``.

The graph is thus built only from queryable facts: symptomatic nodes
joined by proven relationships. Schema + edge↔node referential
integrity are enforced by Pydantic here; the driver re-executes every
SQL after submission to confirm each is queryable and returns rows.

Mirrors the rca ``finalize`` atom's loop-keepalive behaviour: while
the agent has not yet submitted, voluntarily ending a turn with
prose-only is rejected via an ``Inject``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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
    config_schema={
        "type": "object",
        "properties": {"data_dir": {"type": "string"}},
        "additionalProperties": False,
    },
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
    """One re-executable DuckDB SELECT plus the one-line claim it backs."""

    model_config = _STRICT
    sql: str = Field(
        description="ONE DuckDB SELECT, re-executed after submission (must run "
        "and return rows). Compare the NORMAL and ABNORMAL windows side by side "
        "(e.g. UNION ALL) when showing a symptom delta."
    )
    claim: str = Field(description="<=25-word assertion the rows justify.")


class InjectionVerdict(BaseModel):
    """Per-injection effectiveness. A case may have several injections; judge
    EACH independently — they can all engage, some, or none."""

    model_config = _STRICT
    target_service: str = Field(description="The injection target service.")
    fault_kind: str = Field(description="The injected fault_kind for this entry.")
    verdict: Literal["true", "false", "ambiguous"] = Field(
        description="Did THIS injection materially engage, judged from its "
        "target's own normal-vs-abnormal data?"
    )
    rationale: str = Field(
        description="Why, citing the target's own evidence. An injection that "
        "did not engage must not be used as a propagation source."
    )


class PropagationNode(BaseModel):
    """A service you have JUDGED to be genuinely dragged down by the fault(s),
    backed by one or more re-executable SQLs.

    One fault can surface across several signals, so a node may carry MULTIPLE
    evidence rows drawn from diverse sources (traces, metrics of various kinds,
    logs). A service whose overall picture is unchanged or improved is not a
    node, however much one cherry-picked metric dipped.
    """

    model_config = _STRICT
    service: str = Field(
        description="Bare service_name as in the parquet column. Not a "
        "synthetic load generator (loadgenerator, locust, wrk2, dsb-wrk2, k6)."
    )
    symptom_evidence: list[SqlEvidence] = Field(
        description="1..N SQL+claim pairs showing this service degraded in the "
        "abnormal window vs baseline, the way the mechanism predicts. Prefer "
        "DIVERSE signals (trace latency/throughput/errors, metrics, logs) — one "
        "fault often shows in several. Read the whole picture, not one metric."
    )


class PropagationEdge(BaseModel):
    """A directed fault-impact hop ``from_service`` → ``to_service``.

    Direction is fault-impact (upstream-failing → downstream-affected), the
    OPPOSITE of the request-call direction. BOTH endpoints must also appear in
    ``propagation_nodes`` (both must be proven symptomatic); the edge itself is
    proven by a SQL that shows the two services are directly connected — a
    trace parent/child call (either direction) or a shared k8s
    deployment/node relationship.

    Synthetic traffic generators (``loadgenerator``, ``locust``, ``wrk2``,
    ``dsb-wrk2``, ``k6``) are not real services and must NOT appear here.
    """

    model_config = _STRICT
    from_service: str = Field(
        description="Service whose degradation causes the downstream change. "
        "Must appear in propagation_nodes. Not a synthetic load generator."
    )
    to_service: str = Field(
        description="Service that degraded because of from_service. Must "
        "appear in propagation_nodes. Not a synthetic load generator."
    )
    relationship_sql: str = Field(
        description="ONE DuckDB SELECT that returns rows proving from_service "
        "and to_service are DIRECTLY connected — a trace parent/child call "
        "(either direction; look in the normal window) or a shared k8s "
        "deployment/node relationship. This proves the edge can carry impact."
    )
    claim: str = Field(
        description="<=25-word statement of the connection and why from's "
        "degradation reaches to (impact rides on the reverse call)."
    )


class VerifierReport(BaseModel):
    """Service-level fault propagation report — an SQL-backed graph.

    Every node and every edge is queryable: nodes carry a normal-vs-abnormal
    symptom SQL, edges carry a connection SQL. The graph is the set of
    symptomatic nodes joined by proven relationships. Lower-granularity
    targets (container / pod / span) are intentionally NOT part of the
    contract.
    """

    model_config = _STRICT
    injections: list[InjectionVerdict] = Field(
        description="One verdict per injected fault (a case may have several). "
        "Judge each independently — all, some, or none may have engaged."
    )
    propagation_nodes: list[PropagationNode] = Field(
        description="Every service judged to have degraded, each backed by "
        "symptom_evidence. Empty when no injection engaged."
    )
    propagation_edges: list[PropagationEdge] = Field(
        description="Directed fault-impact hops between propagation_nodes. "
        "Empty when no injection engaged."
    )

    @model_validator(mode="after")
    def _edges_reference_nodes(self) -> "VerifierReport":
        node_services = {n.service for n in self.propagation_nodes}
        edge_services = {e.from_service for e in self.propagation_edges} | {
            e.to_service for e in self.propagation_edges
        }
        dangling = sorted(edge_services - node_services)
        if dangling:
            raise ValueError(
                "every edge endpoint must also be a propagation_node (with its "
                f"own symptom_sql); these services are used in edges but are not "
                f"nodes: {dangling}"
            )
        return self


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


@dataclass
class _State:
    submitted: bool = False


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    state = _State()

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

        # No entry-tier "completeness" nudge: forcing the graph toward the
        # request entry tier pressured the agent into inventing symptomless
        # nodes just to reach further. Reachability is an OUTCOME of honest
        # first-principles tracing (see the methodology skill), not a goal the
        # termination protocol enforces.
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
                "Submit the verifier's final report: a per-injection "
                "effectiveness verdict (injections[]) plus an SQL-backed "
                "propagation graph. propagation_nodes = every degraded service, "
                "each with symptom_evidence (1..N SQLs over diverse signals) "
                "comparing normal vs abnormal. propagation_edges = directed "
                "fault-impact hops between those nodes, each with a "
                "relationship_sql proving the two services are directly "
                "connected. Every edge endpoint must also be a node. Service "
                "names must match the parquet service_name column. Every SQL "
                "is re-executed after submission — it must run and return rows."
            ),
            parameters=pydantic_to_openai_tool_schema(VerifierReport),
            fn=_submit,
        )
    )



__all__ = ["MANIFEST", "install"]
