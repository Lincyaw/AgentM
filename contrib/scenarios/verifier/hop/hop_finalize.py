"""Single-hop verdict submission tool for the verifier/hop scenario."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.lib import pydantic_to_tool_schema
from agentm.core.abi import FunctionTool, ToolResult, ToolTerminate
from agentm.core.abi.messages import TextContent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="hop_finalize",
    description="Submit tool for single-hop verdict.",
    registers=("tool:submit_hop_verdict",),
    config_schema={
        "type": "object",
        "properties": {"data_dir": {"type": "string"}},
        "additionalProperties": False,
    },
)

_STRICT = ConfigDict(extra="forbid")


class SqlEvidence(BaseModel):
    model_config = _STRICT
    sql: str = Field(description="DuckDB SELECT comparing normal vs abnormal windows.")
    claim: str = Field(description="<=25-word assertion the rows justify.")


class HopVerdict(BaseModel):
    model_config = _STRICT
    verdict: Literal["confirmed", "rejected"] = Field(
        description="confirmed = service is genuinely degraded by the upstream fault; "
        "rejected = no genuine degradation found."
    )
    rationale: str = Field(description="Why this verdict, citing the data.")
    symptom_evidence: list[SqlEvidence] = Field(
        description="SQL+claim pairs showing the key evidence. Include the "
        "queries you ran even for rejected verdicts (e.g. the throughput/"
        "latency comparison that showed no degradation)."
    )
    relationship_sql: str = Field(
        description="SQL proving the call relationship between the two services "
        "(empty string if rejected)."
    )
    claim: str = Field(description="One-line summary of the hop.")


class HopFinalizeConfig(TypedDict, total=False):
    data_dir: str


class SqlEvidencePayload(TypedDict):
    sql: str
    claim: str


class HopVerdictPayload(TypedDict):
    verdict: Literal["confirmed", "rejected"]
    rationale: str
    symptom_evidence: list[SqlEvidencePayload]
    relationship_sql: str
    claim: str


def _validate_sqls(data_dir: Path, verdict: HopVerdict) -> list[dict[str, str]]:
    try:
        import duckdb
    except ImportError:
        return []
    conn = duckdb.connect(":memory:")
    _cap = os.environ.get("AGENTM_DUCKDB_THREADS")
    if _cap:
        try:
            conn.execute(f"SET threads={max(1, int(_cap))}")
        except (ValueError, duckdb.Error):
            pass
    for pct in (("p50", "0.5"), ("p90", "0.9"), ("p95", "0.95"), ("p99", "0.99")):
        try:
            conn.execute(f"CREATE OR REPLACE MACRO {pct[0]}(x) AS quantile_cont(x, {pct[1]})")
        except duckdb.Error:
            pass
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix == ".parquet" and f.name != "conclusion.parquet":
            path = f.as_posix().replace("'", "''")
            conn.execute(
                f"CREATE OR REPLACE VIEW {f.stem} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )

    failures: list[dict[str, str]] = []

    if verdict.verdict == "confirmed":
        if not verdict.symptom_evidence:
            failures.append({
                "location": "symptom_evidence",
                "error": "confirmed verdict requires non-empty symptom_evidence",
                "sql": "",
            })
        if not verdict.relationship_sql:
            failures.append({
                "location": "relationship_sql",
                "error": "confirmed verdict requires non-empty relationship_sql",
                "sql": "",
            })
        if failures:
            return failures

    for i, ev in enumerate(verdict.symptom_evidence):
        try:
            rows = conn.execute(ev.sql).fetchall()
            if not rows:
                failures.append({
                    "location": f"symptom_evidence[{i}]",
                    "error": "0 rows", "sql": ev.sql,
                })
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "location": f"symptom_evidence[{i}]",
                "error": str(exc).splitlines()[0][:300], "sql": ev.sql,
            })

    if verdict.relationship_sql:
        try:
            rows = conn.execute(verdict.relationship_sql).fetchall()
            if not rows:
                failures.append({
                    "location": "relationship_sql",
                    "error": "0 rows", "sql": verdict.relationship_sql,
                })
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "location": "relationship_sql",
                "error": str(exc).splitlines()[0][:300],
                "sql": verdict.relationship_sql,
            })

    conn.close()
    return failures


def install(api: ExtensionAPI, config: HopFinalizeConfig) -> None:
    raw_dir = config.get("data_dir")
    data_dir = Path(raw_dir) if raw_dir else None

    async def _submit(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        payload = cast(HopVerdictPayload, args)
        try:
            verdict = HopVerdict.model_validate(payload)
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

        if verdict.verdict == "confirmed" and data_dir and data_dir.is_dir():
            failures = _validate_sqls(data_dir, verdict)
            if failures:
                return ToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps({
                            "error": "sql_validation_failed",
                            "failures": failures,
                            "hint": "Fix the failing SQLs and resubmit.",
                        }, ensure_ascii=False),
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
            reason="hop:verdict-submitted",
        )

    api.register_tool(
        FunctionTool(
            name="submit_hop_verdict",
            description=(
                "Submit your verdict on this single propagation hop. "
                "confirmed = the service is genuinely degraded by the "
                "upstream fault. rejected = no genuine degradation. "
                "For confirmed, symptom_evidence and relationship_sql "
                "must be non-empty and re-executable."
            ),
            parameters=pydantic_to_tool_schema(HopVerdict),
            fn=_submit,
        )
    )


__all__ = ["MANIFEST", "install"]
