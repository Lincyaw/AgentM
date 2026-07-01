"""Seed verification verdict submission tool.

Same structure as hop_finalize but without the relationship field —
seeds are injection targets, not propagation endpoints.
"""
from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from fpg import Evidence, build_schema, load_profile
from verifier.lib.finalize_feedback import (
    duration_unit_failures,
    sql_statement_shape_failure,
    sql_validation_error_payload,
)


class SeedFinalizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: str | None = None


MANIFEST = ExtensionManifest(
    name="seed_finalize",
    description="Submit tool for seed verification verdict (fpg-native).",
    registers=("tool:submit_seed_verdict",),
    config_schema=SeedFinalizeConfig,
)

_PROFILE_PATH = Path(__file__).parents[1] / "fpg_profile.toml"
_SCHEMA = build_schema(load_profile(_PROFILE_PATH))
NodePredicate = cast(type[Enum], _SCHEMA.NodePredicate)

_STRICT: Final = ConfigDict(extra="forbid")


class SelectivityComparison(BaseModel):
    """One target-vs-control comparison backed by SQL query results."""

    model_config = _STRICT
    metric: str = Field(
        description="What is being compared, e.g. 'span_count', "
        "'p95_latency_ms', 'error_rate', 'log_error_count'."
    )
    target_query: Evidence = Field(
        description="SQL query measuring the metric on the TARGET path "
        "(the injected/propagated service, endpoint, or link)."
    )
    target_normal: float | None = Field(
        default=None,
        description="Target metric value in the normal window.",
    )
    target_abnormal: float | None = Field(
        default=None,
        description="Target metric value in the abnormal window.",
    )
    control_query: Evidence = Field(
        description="SQL query measuring the SAME metric on comparable "
        "non-target paths (sibling endpoints, sibling services at same "
        "topology depth, or same service on a different node)."
    )
    control_normal: float | None = Field(
        default=None,
        description="Control metric value in the normal window.",
    )
    control_abnormal: float | None = Field(
        default=None,
        description="Control metric value in the abnormal window.",
    )
    selective: bool = Field(
        description="Your assessment: did the target change meaningfully "
        "MORE than the control? True only when the target's shift is "
        "clearly disproportionate to the control's shift."
    )
    comparison_rationale: str = Field(
        description="One sentence: why the target change is or is not "
        "selective relative to the control."
    )


class InvestigationCoverage(BaseModel):
    model_config = _STRICT
    schema_discovery: str = Field(
        description="What table/schema/value discovery you performed before "
        "writing detailed SQL, e.g. list_tables plus SELECT DISTINCT/DESCRIBE "
        "for status columns, span-kind values, log levels/templates, metric "
        "names, span names, and target/caller identifiers. If a discovery "
        "step was impossible, say why."
    )
    target_trace: str = Field(
        description="Target trace coverage: target service or rule-bearing "
        "side normal-vs-abnormal span counts, endpoints, latency, status, "
        "HTTP status, and appearing/disappearing span names."
    )
    caller_link_trace: str = Field(
        description="Caller/link coverage: parent_span_id join used to find "
        "normal callers or link endpoints, plus abnormal caller/link checks. "
        "Do not rely only on attr.span_kind values."
    )
    metrics: str = Field(
        description="Metric coverage: target and relevant caller/link "
        "resource/deployment/JVM/container/network signals checked. If "
        "metrics are unavailable or uninformative, state that explicitly."
    )
    logs: str = Field(
        description="Log coverage: target and relevant caller/link log "
        "levels/templates/messages discovered and normal-vs-abnormal findings. "
        "If logs are unavailable or uninformative, state that explicitly."
    )
    fault_specific_reasoning: str = Field(
        description="Why the trace/metric/log/caller-link findings support "
        "this verdict for this specific fault kind."
    )


class SeedVerdict(BaseModel):
    model_config = _STRICT
    verdict: Literal["confirmed", "rejected", "inconclusive"] = Field(
        description="confirmed = injection took effect, service is degraded; "
        "rejected = no degradation found, injection had no observable effect; "
        "inconclusive = ambiguous signal (e.g. zero data in both windows)."
    )
    effect_target: str | None = Field(
        default=None,
        description="For link/path-scoped faults, the service or backing "
        "component that actually shows the observed degradation. Use the "
        "caller/source side when caller-owned spans time out while cross-link "
        "child spans disappear. Leave null for ordinary service-scoped faults.",
    )
    predicate: NodePredicate | None = Field(  # type: ignore[valid-type]
        default=None,
        description="The failure mode the service exhibits, REQUIRED "
        "when confirmed. Pick the most specific value your evidence "
        "supports.",
    )
    rationale: str = Field(description="Why this verdict, citing the data.")
    evidence: list[Evidence] = Field(
        description="REQUIRED for all verdicts. Re-executable DuckDB SQL "
        "statements (query.language='sql') comparing normal vs abnormal "
        "windows, plus an explanation of what the result shows."
    )
    selectivity: list[SelectivityComparison] = Field(
        default_factory=list,
        description="REQUIRED for confirmed verdicts. At least one "
        "target-vs-control comparison showing the observed change is "
        "selective to the injected path and not a proportional "
        "system-wide or workload shift. Each comparison must include "
        "SQL queries with results for both target and control paths.",
    )
    investigation_coverage: InvestigationCoverage = Field(
        description="REQUIRED for all verdicts. Summarize schema discovery "
        "plus target trace, caller/link trace, metric, log, and fault-specific "
        "checks. This is audit metadata; do not replace SQL evidence with "
        "free text."
    )
    claim: str = Field(description="One-line summary.")

    @field_validator("effect_target", mode="before")
    @classmethod
    def _reject_literal_null_effect_target(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            raise ValueError("effect_target must be JSON null, not a string")
        return value

    @model_validator(mode="after")
    def _verdict_contract(self) -> "SeedVerdict":
        if not self.evidence:
            raise ValueError("evidence is required for all verdicts")
        if self.verdict == "confirmed":
            if self.predicate is None:
                raise ValueError("confirmed verdict requires predicate")
            if not self.selectivity:
                raise ValueError(
                    "confirmed verdict requires at least one selectivity "
                    "comparison (target path vs control path with SQL "
                    "queries and results)"
                )
            if not any(s.selective for s in self.selectivity):
                raise ValueError(
                    "confirmed verdict requires at least one selectivity "
                    "comparison marked selective=true; if no comparison is "
                    "selective, the verdict should be rejected or inconclusive"
                )
        elif self.predicate is not None:
            raise ValueError("non-confirmed verdict must omit predicate")
        return self


def _validate_sqls(data_dir: Path, verdict: SeedVerdict) -> list[dict[str, str]]:
    try:
        import duckdb
    except ImportError:
        return []
    conn = duckdb.connect(":memory:")
    cap = os.environ.get("AGENTM_DUCKDB_THREADS")
    if cap:
        try:
            conn.execute(f"SET threads={max(1, int(cap))}")
        except (ValueError, duckdb.Error):
            pass
    for pct in (("p50", "0.5"), ("p90", "0.9"), ("p95", "0.95"), ("p99", "0.99")):
        try:
            conn.execute(f"CREATE OR REPLACE MACRO {pct[0]}(x) AS quantile_cont(x, {pct[1]})")
        except duckdb.Error:
            pass
    view_names: set[str] = set()
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix == ".parquet" and f.name != "conclusion.parquet":
            view_names.add(f.stem)
            path = f.as_posix().replace("'", "''")
            conn.execute(
                f"CREATE OR REPLACE VIEW {f.stem} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )

    failures: list[dict[str, str]] = []
    statements: list[tuple[str, str]] = []

    def _check_evidence(location: str, ev: Evidence) -> None:
        if ev.query.language != "sql":
            failures.append({
                "location": location,
                "error": "only query.language='sql' is executable",
                "sql": ev.query.statement,
            })
            return
        statements.append((location, ev.query.statement))
        shape_failure = sql_statement_shape_failure(location, ev.query.statement)
        if shape_failure:
            failures.append(shape_failure)
            return
        try:
            rows = conn.execute(ev.query.statement).fetchall()
            if not rows:
                failures.append({
                    "location": location,
                    "error": "0 rows", "sql": ev.query.statement,
                })
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "location": location,
                "error": str(exc).splitlines()[0][:300],
                "sql": ev.query.statement,
            })

    for i, ev in enumerate(verdict.evidence):
        _check_evidence(f"evidence[{i}]", ev)
    for i, sel in enumerate(verdict.selectivity):
        _check_evidence(f"selectivity[{i}].target_query", sel.target_query)
        _check_evidence(f"selectivity[{i}].control_query", sel.control_query)

    failures.extend(duration_unit_failures(statements))
    del view_names
    conn.close()
    return failures


def install(api: ExtensionAPI, config: SeedFinalizeConfig) -> None:
    data_dir = Path(config.data_dir) if config.data_dir else None

    async def _submit(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        try:
            verdict = SeedVerdict.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "validation_failed",
                        "detail": exc.errors(include_url=False),
                    }, ensure_ascii=False),
                )],
                is_error=True,
            )

        if data_dir and data_dir.is_dir():
            failures = _validate_sqls(data_dir, verdict)
            if failures:
                return ToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps(
                            sql_validation_error_payload(failures),
                            ensure_ascii=False,
                        ),
                    )],
                    is_error=True,
                )

        return ToolTerminate(
            result=ToolResult(
                content=[TextContent(
                    type="text",
                    text=verdict.model_dump_json(by_alias=True),
                )]
            ),
            reason="seed:verdict-submitted",
        )

    api.register_tool(
        FunctionTool(
            name="submit_seed_verdict",
            description=(
                "Submit your verdict on whether the fault injection took "
                "effect on this service. confirmed = degradation observed "
                "(predicate and evidence required). rejected = no effect. "
                "For link/path faults, use parent_span_id joins to verify "
                "caller/link behavior; do not rely only on attr.span_kind."
            ),
            parameters=SeedVerdict,
            fn=_submit,
        )
    )


__all__: Final = ["MANIFEST", "install"]
