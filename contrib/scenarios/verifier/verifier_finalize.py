"""Verifier-scenario termination protocol.

The verifier ends by calling ``submit_propagation_report``. Unlike the
RCA scenario (which submits a guess at the unknown root cause), the
verifier already knows what was injected — it submits:

* ``injection_effective`` — did the injection actually materialize? One
  of ``true`` / ``false`` / ``ambiguous`` (free-text rationale required,
  citing the injection target's own normal-vs-abnormal symptom).
* ``propagation_nodes`` — every service proven to have degraded. Each
  node carries a ``symptom_sql`` that compares the normal and abnormal
  windows (delta visible) plus a short ``claim``. A flat/improved metric
  is not a symptom, so such a service is not a node.
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
import os
from dataclasses import dataclass, field
from pathlib import Path
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


# Synthetic traffic generators are not real services and never form part of
# the entry tier (they sit *above* it). Kept in sync with PropagationEdge.
_SYNTHETIC = frozenset(
    {"loadgenerator", "load-generator", "load_generator", "locust", "wrk2", "dsb-wrk2", "k6"}
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


class PropagationNode(BaseModel):
    """A service shown — by re-executable SQL — to carry a fault symptom in
    the abnormal window.

    The graph is built from queryable facts: every node is a service whose
    symptom is proven by a single SQL that compares the NORMAL and ABNORMAL
    windows. A flat or improved metric is NOT a symptom — such a service is
    not a node.
    """

    model_config = _STRICT
    service: str = Field(
        description="Bare service_name as in the parquet column. Not a "
        "synthetic load generator (loadgenerator, locust, wrk2, dsb-wrk2, k6)."
    )
    symptom_sql: str = Field(
        description="ONE DuckDB SELECT that returns BOTH the normal and the "
        "abnormal window side by side (e.g. UNION ALL of the two windows) so "
        "the delta is visible — proving this service degraded (worse latency / "
        "errors / throughput collapse) in the abnormal window."
    )
    claim: str = Field(
        description="<=25-word delta the SQL shows, e.g. 'p99 0.15ms->0.25ms (+64%)'."
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
    injection_effective: Literal["true", "false", "ambiguous"]
    effectiveness_rationale: str = Field(
        description="Why effective / not / ambiguous, citing the abnormal "
        "vs normal window comparison of the injection TARGET's own symptom."
    )
    propagation_nodes: list[PropagationNode] = Field(
        description="Every service proven (by symptom_sql) to have degraded. "
        "Empty list is acceptable when injection_effective='false'."
    )
    propagation_edges: list[PropagationEdge] = Field(
        description="Directed fault-impact hops between propagation_nodes. "
        "Empty list is acceptable when injection_effective='false'."
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
    nudged: bool = False
    # ``None`` = not computed yet; a (possibly empty) set once resolved.
    entry_services: frozenset[str] | None = field(default=None)


def _resolve_data_dir(config: dict[str, Any]) -> Path | None:
    raw = config.get("data_dir") or os.environ.get("AGENTM_RCA_DATA_DIR")
    return Path(raw).resolve() if raw else None


def _entry_services(data_dir: Path | None) -> frozenset[str]:
    """Services with zero in-degree from *real* (non-synthetic) callers — the
    request entry tier the fault impact should ultimately reach.

    Computed from the normal-window trace call graph: a service is entry-tier
    if no other real service calls it (it is only driven by a synthetic load
    generator, or sits at the top of the call hierarchy). Topology-agnostic —
    yields ``frontend-proxy`` (otel-demo), ``ts-ui-dashboard`` (trainticket),
    ``frontend`` (hotel-reservation). Fails open (empty set) on any error so
    the gate can never block a submission due to missing/odd data.
    """
    if data_dir is None:
        return frozenset()
    nrm = data_dir / "normal_traces.parquet"
    if not nrm.exists():
        return frozenset()
    try:
        import duckdb
    except ImportError:
        return frozenset()
    syn = ",".join(f"'{s}'" for s in sorted(_SYNTHETIC))
    query = f"""
        WITH edges AS (
          SELECT DISTINCT p.service_name AS caller, ch.service_name AS callee
          FROM read_parquet(?) ch JOIN read_parquet(?) p
            ON ch.parent_span_id = p.span_id AND ch.trace_id = p.trace_id
          WHERE ch.service_name <> p.service_name
            AND lower(p.service_name) NOT IN ({syn})
        ),
        allsvc AS (
          SELECT DISTINCT service_name s FROM read_parquet(?)
          WHERE lower(service_name) NOT IN ({syn})
        )
        SELECT a.s FROM allsvc a
        WHERE NOT EXISTS (SELECT 1 FROM edges e WHERE e.callee = a.s)
    """
    try:
        path = str(nrm)
        rows = duckdb.execute(query, [path, path, path]).fetchall()
        return frozenset(r[0] for r in rows if r[0])
    except Exception:
        return frozenset()


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    state = _State()
    data_dir = _resolve_data_dir(config)

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

        # One-shot entry-tier nudge: if the injection materialised but the
        # propagation graph never reaches a user-facing entry service, the
        # agent has almost certainly stopped short of the full fan-out (the
        # common failure mode). Push it once to trace toward the entry tier;
        # if it resubmits unchanged, accept — the agent may have evidence the
        # impact genuinely dies out earlier. Soft by design: it never forces
        # specific edges (which would manufacture false positives), only asks
        # for one more pass toward the request entry.
        if report.injection_effective != "false" and not state.nudged:
            if state.entry_services is None:
                state.entry_services = _entry_services(data_dir)
            entry = state.entry_services
            graph_services = {e.from_service for e in report.propagation_edges} | {
                e.to_service for e in report.propagation_edges
            }
            if entry and not (graph_services & entry):
                state.nudged = True
                return ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "error": "propagation_graph_incomplete",
                                    "reason": (
                                        "The graph does not reach any user-facing "
                                        "entry service. Fault impact normally "
                                        "propagates outward until it hits the "
                                        "request entry tier."
                                    ),
                                    "entry_tier_services": sorted(entry),
                                    "current_graph_services": sorted(graph_services),
                                    "instruction": (
                                        "For each service already in your graph, "
                                        "examine its callers in the abnormal window "
                                        "and extend the graph along real failure "
                                        "edges (a caller whose spans fail because the "
                                        "callee failed) — hop by hop toward an entry "
                                        "service above. Do NOT add edges for services "
                                        "that merely show lower throughput; only add "
                                        "an edge backed by a causal SQL claim. If you "
                                        "have already confirmed via SQL that the "
                                        "impact genuinely stops where it is, call "
                                        "submit_propagation_report again unchanged and "
                                        "it will be accepted."
                                    ),
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
                "verdict on the injection plus an SQL-backed propagation "
                "graph. propagation_nodes = every degraded service, each with "
                "a symptom_sql comparing normal vs abnormal. propagation_edges "
                "= directed fault-impact hops between those nodes, each with a "
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
