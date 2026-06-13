"""Single-hop verdict submission tool for the verifier/hop scenario.

The verdict is fpg-native: evidence items are fpg ``Evidence`` objects
(re-executable query + explanation) and the symptom classification is
constrained to the node-predicate vocabulary of the verifier profile
(fpg_profile.toml). Every SQL statement is re-executed against the case
data before the verdict is accepted — a verdict that cites evidence
which does not run, or returns nothing, is rejected at submission time.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from fpg import Evidence, build_schema, load_profile

class HopFinalizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: str | None = None


MANIFEST = ExtensionManifest(
    name="hop_finalize",
    description="Submit tool for single-hop verdict (fpg-native).",
    registers=("tool:submit_hop_verdict",),
    config_schema=HopFinalizeConfig,
)

_PROFILE_PATH = Path(__file__).resolve().parents[1] / "fpg_profile.toml"
_SCHEMA = build_schema(load_profile(_PROFILE_PATH))
NodePredicate = cast(type[Enum], _SCHEMA.NodePredicate)

_STRICT: Final = ConfigDict(extra="forbid")


class HopVerdict(BaseModel):
    model_config = _STRICT
    verdict: Literal["confirmed", "rejected"] = Field(
        description="confirmed = service is genuinely degraded by the upstream fault; "
        "rejected = no genuine degradation found."
    )
    predicate: NodePredicate | None = Field(  # type: ignore[valid-type]
        default=None,
        description="The failure mode the target service exhibits, REQUIRED "
        "when confirmed. Pick the most specific value your evidence "
        "supports; 'service_degraded' if none fits; 'other' only with an "
        "explanation in claim.",
    )
    rationale: str = Field(description="Why this verdict, citing the data.")
    evidence: list[Evidence] = Field(
        description="Re-executable evidence: each item is a DuckDB SQL "
        "statement (query.language='sql') comparing normal vs abnormal "
        "windows, plus an explanation of what the result shows. Include "
        "the queries you ran even for rejected verdicts."
    )
    relationship: Evidence | None = Field(
        default=None,
        description="Proof of the call relationship between the two "
        "services (SQL + explanation). REQUIRED when confirmed.",
    )
    claim: str = Field(description="One-line summary of the hop.")

    @model_validator(mode="after")
    def _confirmed_contract(self) -> "HopVerdict":
        if self.verdict == "confirmed":
            if self.predicate is None:
                raise ValueError("confirmed verdict requires predicate")
            if not self.evidence:
                raise ValueError("confirmed verdict requires non-empty evidence")
            if self.relationship is None:
                raise ValueError("confirmed verdict requires relationship evidence")
        return self


def _validate_sqls(data_dir: Path, verdict: HopVerdict) -> list[dict[str, str]]:
    """Re-execute every evidence statement; collect failures."""
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
    items: list[tuple[str, Evidence]] = [
        (f"evidence[{i}]", ev) for i, ev in enumerate(verdict.evidence)
    ]
    if verdict.relationship is not None:
        items.append(("relationship", verdict.relationship))

    for location, ev in items:
        if ev.query.language != "sql":
            failures.append({
                "location": location,
                "error": "only query.language='sql' is executable against "
                "this case's DuckDB data",
                "sql": ev.query.statement,
            })
            continue
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

    conn.close()
    return failures


def install(api: ExtensionAPI, config: HopFinalizeConfig) -> None:
    data_dir = Path(config.data_dir) if config.data_dir else None

    async def _submit(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        try:
            verdict = HopVerdict.model_validate(args)
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
                "upstream fault (then predicate, evidence and relationship "
                "are required and every SQL must re-execute successfully). "
                "rejected = no genuine degradation; still include the "
                "queries that showed no degradation."
            ),
            parameters=HopVerdict,
            fn=_submit,
        )
    )


__all__: Final = ["MANIFEST", "install"]
