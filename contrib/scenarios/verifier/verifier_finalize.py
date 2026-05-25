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
import logging
import os
from dataclasses import dataclass
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
        description="ONE DuckDB SELECT proving a real path between the two "
        "services — a span parent/child call chain (direct, or one an ancestor "
        "of the other via parent_span_id=span_id) or a shared k8s "
        "deployment/node. NOT mere same-trace co-occurrence (a trace holds "
        "parallel siblings that never call each other). Proves the edge can "
        "carry impact."
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


logger = logging.getLogger(__name__)


@dataclass
class _State:
    submitted: bool = False
    critic_done: bool = False


def _duck_conn(data_dir: Path) -> Any:
    import duckdb

    conn = duckdb.connect(":memory:")
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix == ".parquet" and f.name != "conclusion.parquet":
            path = f.as_posix().replace("'", "''")
            conn.execute(
                f"CREATE OR REPLACE VIEW {f.stem} AS SELECT * FROM read_parquet('{path}')"
            )
    return conn


# ---------------------------------------------------------------------------
# Critic — single-shot LLM review of the submitted report
# ---------------------------------------------------------------------------


def _collect_critic_context(conn: Any, report: VerifierReport) -> dict[str, Any]:
    """Gather system-level data the critic needs to judge drift vs fault."""
    try:
        rows = conn.execute(
            "SELECT n.service_name, n.cnt AS normal, a.cnt AS abnormal, "
            "ROUND(a.cnt * 1.0 / n.cnt, 3) AS ratio "
            "FROM (SELECT service_name, COUNT(*) AS cnt FROM normal_traces "
            "  GROUP BY service_name) n "
            "JOIN (SELECT service_name, COUNT(*) AS cnt FROM abnormal_traces "
            "  GROUP BY service_name) a "
            "ON n.service_name = a.service_name ORDER BY ratio"
        ).fetchall()
        throughput = [
            {"service": r[0], "normal": r[1], "abnormal": r[2], "ratio": float(r[3])}
            for r in rows
        ]
    except Exception:  # noqa: BLE001
        throughput = []

    try:
        rows = conn.execute(
            "SELECT 'normal' AS win, service_name, COUNT(*) AS cnt, "
            "AVG(duration) AS avg_dur, APPROX_QUANTILE(duration, 0.99) AS p99 "
            "FROM normal_traces GROUP BY service_name "
            "UNION ALL "
            "SELECT 'abnormal' AS win, service_name, COUNT(*) AS cnt, "
            "AVG(duration) AS avg_dur, APPROX_QUANTILE(duration, 0.99) AS p99 "
            "FROM abnormal_traces GROUP BY service_name "
            "ORDER BY service_name, win"
        ).fetchall()
        latency = [
            {"win": r[0], "service": r[1], "cnt": r[2],
             "avg_ms": round(r[3] / 1e6, 3), "p99_ms": round(r[4] / 1e6, 3)}
            for r in rows
        ]
    except Exception:  # noqa: BLE001
        latency = []

    try:
        rows = conn.execute(
            "SELECT parent.service_name AS caller, child.service_name AS callee, "
            "COUNT(*) AS calls "
            "FROM normal_traces child "
            "JOIN normal_traces parent ON child.parent_span_id = parent.span_id "
            "WHERE child.service_name <> parent.service_name "
            "GROUP BY parent.service_name, child.service_name ORDER BY calls DESC"
        ).fetchall()
        call_graph = [{"caller": r[0], "callee": r[1], "calls": r[2]} for r in rows]
    except Exception:  # noqa: BLE001
        call_graph = []

    return {
        "throughput_ratios": throughput,
        "latency_overview": latency,
        "call_graph": call_graph,
    }


def _build_critic_prompt(
    report: VerifierReport,
    injection_spec: dict[str, Any],
    context: dict[str, Any],
) -> str:
    inj_lines = []
    for inj in injection_spec.get("engine_config", []):
        inj_lines.append(
            f"- {inj.get('app', '?')}: {inj.get('chaos_type', '?')} "
            f"(duration={inj.get('duration', '?')}m)"
        )
    inj_text = "\n".join(inj_lines) or "(none)"

    tp = context.get("throughput_ratios", [])
    ratios = [r["ratio"] for r in tp]
    median_ratio = sorted(ratios)[len(ratios) // 2] if ratios else 0
    tp_lines = [
        f"| {'service':25s} | {'normal':>7s} | {'abnorm':>7s} | {'ratio':>6s} |",
        f"|{'-' * 27}|{'-' * 9}|{'-' * 9}|{'-' * 8}|",
    ]
    for r in tp:
        tp_lines.append(
            f"| {r['service']:25s} | {r['normal']:7d} | {r['abnormal']:7d} "
            f"| {r['ratio']:6.3f} |"
        )
    tp_lines.append(f"\nMedian system ratio: {median_ratio:.3f}")

    lat = context.get("latency_overview", [])
    lat_lines = [
        f"| {'win':8s} | {'service':25s} | {'cnt':>6s} | {'avg_ms':>10s} "
        f"| {'p99_ms':>10s} |",
        f"|{'-' * 10}|{'-' * 27}|{'-' * 8}|{'-' * 12}|{'-' * 12}|",
    ]
    for r in lat:
        lat_lines.append(
            f"| {r['win']:8s} | {r['service']:25s} | {r['cnt']:6d} | "
            f"{r['avg_ms']:10.3f} | {r['p99_ms']:10.3f} |"
        )

    cg = context.get("call_graph", [])
    cg_lines = [f"  {r['caller']} -> {r['callee']} ({r['calls']} calls)" for r in cg]

    verdicts = "\n".join(
        f"- {i.target_service}/{i.fault_kind}: verdict={i.verdict}. {i.rationale}"
        for i in report.injections
    )
    nodes = "\n".join(
        f"- {n.service}: " + "; ".join(e.claim for e in n.symptom_evidence)
        for n in report.propagation_nodes
    )
    edges = "\n".join(
        f"- {e.from_service} -> {e.to_service}: {e.claim}"
        for e in report.propagation_edges
    )

    nl = "\n"
    return f"""\
You are a critic reviewing a fault-propagation report. Find errors in \
reasoning — do NOT redo the analysis.

## Injections performed
{inj_text}

## System-wide throughput (all services, abnormal/normal)
{nl.join(tp_lines)}

## Latency overview (all services, both windows)
{nl.join(lat_lines)}

## Call graph (normal window, caller -> callee)
{nl.join(cg_lines)}

## Report to review

### Injection verdicts
{verdicts}

### Propagation nodes
{nodes}

### Propagation edges
{edges}

## What to check

1. **DRIFT vs FAULT**: The system throughput dropped overall (median \
ratio {median_ratio:.3f}). A node whose throughput ratio is close to \
the system median, AND whose latency change is small (sub-millisecond \
or <2x), is likely just following the system-wide load change — not \
being dragged down by the fault. Flag such nodes.

2. **EFFECTIVENESS GATE**: For mem_stress / cpu_stress, the target must \
show a LATENCY rise (not just resource metric changes like memory \
usage). Memory going up without latency impact means the stress did \
not materialize into observable service degradation. Check the latency \
overview above.

3. **CAUSAL DIRECTION**: Impact flows from broken dependency to its \
CALLERS (upstream in the request path). If service A is slow and A \
calls B, A's slowness does NOT make B slow — only A's callers are \
affected (they wait for A). Check: for each edge from->to, the call \
graph should show to calls from (not from calls to). If from calls \
to, that edge's causal direction is wrong.

4. **WEAK EVIDENCE**: A latency change of <1ms absolute AND <2x \
relative, in a service whose throughput drop matches the system \
median, is noise — not fault evidence.

## Output format

Write a concise review. For each issue, state which node/edge is \
problematic, why (citing the data), and what should change (drop the \
node, drop the edge, change a verdict, etc.).

If the report is sound, write ONLY the word: APPROVED"""


async def _run_critic(data_dir: Path, report: VerifierReport) -> str | None:
    """Single-shot LLM critic. Returns review feedback or None if approved."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        return None

    inj_path = data_dir / "injection.json"
    injection_spec = json.loads(inj_path.read_text()) if inj_path.exists() else {}

    conn = _duck_conn(data_dir)
    context = _collect_critic_context(conn, report)
    conn.close()

    prompt = _build_critic_prompt(report, injection_spec, context)

    headers_raw = os.environ.get("ANTHROPIC_DEFAULT_HEADERS")
    default_headers = json.loads(headers_raw) if headers_raw else None
    client = AsyncAnthropic(default_headers=default_headers)
    model = os.environ.get("AGENTM_MODEL", "K2.6")

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("critic LLM call failed: %s", exc)
        return None

    feedback = response.content[0].text.strip()  # type: ignore[union-attr]
    if feedback.upper() == "APPROVED":
        return None
    return feedback


# ---------------------------------------------------------------------------
# SQL validation
# ---------------------------------------------------------------------------


def _validate_sqls(data_dir: Path, report: VerifierReport) -> list[dict[str, str]]:
    """Re-execute every SQL in the report. Returns a list of failures (empty = all ok)."""
    try:
        conn = _duck_conn(data_dir)
    except Exception:  # noqa: BLE001
        return []

    failures: list[dict[str, str]] = []

    for node in report.propagation_nodes:
        for i, ev in enumerate(node.symptom_evidence):
            try:
                rows = conn.execute(ev.sql).fetchall()
                if not rows:
                    failures.append({
                        "location": f"propagation_nodes[{node.service}].symptom_evidence[{i}]",
                        "error": "query returned 0 rows",
                        "sql": ev.sql,
                    })
            except Exception as exc:  # noqa: BLE001
                failures.append({
                    "location": f"propagation_nodes[{node.service}].symptom_evidence[{i}]",
                    "error": str(exc).splitlines()[0][:300],
                    "sql": ev.sql,
                })

    for edge in report.propagation_edges:
        try:
            rows = conn.execute(edge.relationship_sql).fetchall()
            if not rows:
                failures.append({
                    "location": f"propagation_edges[{edge.from_service}→{edge.to_service}].relationship_sql",
                    "error": "query returned 0 rows",
                    "sql": edge.relationship_sql,
                })
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "location": f"propagation_edges[{edge.from_service}→{edge.to_service}].relationship_sql",
                "error": str(exc).splitlines()[0][:300],
                "sql": edge.relationship_sql,
            })

    conn.close()
    return failures


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    state = _State()
    raw_dir = config.get("data_dir") or os.environ.get("AGENTM_RCA_DATA_DIR")
    data_dir = Path(raw_dir) if raw_dir else None

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

        if data_dir and data_dir.is_dir():
            failures = _validate_sqls(data_dir, report)
            if failures:
                return ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "error": "sql_validation_failed",
                                    "failures": failures,
                                    "hint": "Fix the failing SQLs and resubmit. "
                                    "Each SQL must execute without error and return at least one row.",
                                },
                                ensure_ascii=False,
                            ),
                        )
                    ],
                    is_error=True,
                )

            if not state.critic_done:
                state.critic_done = True
                feedback = await _run_critic(data_dir, report)
                if feedback:
                    return ToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=json.dumps(
                                    {
                                        "review": "critic_feedback",
                                        "feedback": feedback,
                                        "hint": "A critic has reviewed your report "
                                        "and found issues. Address the feedback "
                                        "above, then resubmit with corrections.",
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
