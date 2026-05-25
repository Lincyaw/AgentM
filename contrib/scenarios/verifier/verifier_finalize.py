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
import os
from dataclasses import dataclass, field
from pathlib import Path
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
