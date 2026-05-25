"""Auditor-scenario termination protocol: ``submit_audit``.

The auditor is handed a *candidate* service-level fault-propagation graph
derived from the dataset's existing labels (the ground-truth causal graph
projected to service granularity), together with the known injection spec
and per-fault mechanism docs — all spliced into its prompt by the driver.
Its job is to AUDIT that candidate against the raw observability parquets:

* 查准 (verify) — for every candidate edge, confirm with SQL whether the
  data supports it (``supported``) or not (``unsupported``). Unsupported
  candidate edges are suspected label false-positives.
* 查漏 (supplement) — find edges the abnormal-window data supports that the
  candidate graph is missing. These are suspected label false-negatives.

It does NOT see ``conclusion.parquet`` / ``causal_graph.json`` directly (the
driver excludes / does not pass them) — the candidate graph is the only
label-derived input, and every verdict must stand on re-executable SQL over
traces / metrics / logs.

It ends by calling ``submit_audit``. ``status`` is a controlled value
(``supported`` / ``unsupported``) because downstream tooling branches on it;
free-text ``note`` / ``summary`` carry the reasoning (see
[[feedback_no_preset_subjective_labels]]).
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
    name="auditor_submit",
    description=(
        "Termination protocol: the auditor must call submit_audit to end "
        "the audit with per-edge verdicts plus any supplemented edges."
    ),
    registers=("tool:submit_audit",),
)


_STRICT = ConfigDict(extra="forbid")


class SqlEvidence(BaseModel):
    model_config = _STRICT
    sql: str = Field(description="DuckDB SQL re-executable against case parquets.")
    claim: str = Field(description="<=25-word assertion the SQL rows back.")


class EdgeAudit(BaseModel):
    """Verdict on ONE candidate (label-derived) edge."""

    model_config = _STRICT
    from_service: str
    to_service: str
    status: Literal["supported", "unsupported"] = Field(
        description="`supported` if the abnormal-window data shows from's "
        "anomaly causes to's (failed-call / timing / call-graph argument); "
        "`unsupported` if the data does not back the causal edge (suspected "
        "label false-positive)."
    )
    evidence: list[SqlEvidence] = Field(
        description="SQL+claim backing the verdict (for both supported and "
        "unsupported — show what you checked)."
    )
    note: str = Field(description="One-sentence rationale.")


class AddedEdge(BaseModel):
    """An edge the data supports that the candidate graph was MISSING."""

    model_config = _STRICT
    from_service: str = Field(
        description="Service whose degradation causes the downstream change. "
        "Fault-impact direction (upstream-failing → downstream-affected), NOT "
        "request-call direction. No synthetic load generators."
    )
    to_service: str
    evidence: list[SqlEvidence] = Field(
        description="At least one SQL+claim showing this hop is real and "
        "directional."
    )


class AuditReport(BaseModel):
    model_config = _STRICT
    edge_audits: list[EdgeAudit] = Field(
        description="One verdict per candidate edge handed to you. Cover "
        "every candidate edge exactly once."
    )
    added_edges: list[AddedEdge] = Field(
        description="Edges the data supports that the candidate graph lacked "
        "(suspected label false-negatives). Empty if none."
    )
    summary: str = Field(
        description="Free-text: overall label quality for this case — which "
        "candidate edges look wrong and what was missing."
    )


@dataclass
class _State:
    submitted: bool = False


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    state = _State()
    del config

    async def _submit(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        try:
            report = AuditReport.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "auditor_contract_validation_failed",
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
            reason="auditor:audit-submitted",
        )

    api.register_tool(
        FunctionTool(
            name="submit_audit",
            description=(
                "Submit the label audit: a supported/unsupported verdict with "
                "SQL evidence for every candidate edge (查准), plus any edges "
                "the data supports that the candidate graph was missing (查漏)."
            ),
            parameters=pydantic_to_openai_tool_schema(AuditReport),
            fn=_submit,
        )
    )


__all__ = ["MANIFEST", "install"]
