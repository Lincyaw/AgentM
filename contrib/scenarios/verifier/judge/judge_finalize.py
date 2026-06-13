"""Whole-graph review submission tool for the verifier/judge scenario.

Promotion is fpg-native: a promoted service must name the confirmed
upstream service it cascades through (``via_service``), so the workflow
can attach it to the graph with a real edge — a promoted node must
never become a spurious root cause. The symptom classification reuses
the node-predicate vocabulary of the verifier profile.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Final, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from fpg import build_schema, load_profile

class JudgeFinalizeConfig(BaseModel):
    """No configuration; present for uniform strong typing."""

    model_config = ConfigDict(extra="forbid")


MANIFEST = ExtensionManifest(
    name="judge_finalize",
    description="Submit tool for whole-graph judge review (fpg-native).",
    registers=("tool:submit_judge_review",),
    config_schema=JudgeFinalizeConfig,
)

_PROFILE_PATH = Path(__file__).resolve().parents[1] / "fpg_profile.toml"
_SCHEMA = build_schema(load_profile(_PROFILE_PATH))
NodePredicate = cast(type[Enum], _SCHEMA.NodePredicate)

_STRICT: Final = ConfigDict(extra="forbid")


class JudgePromotion(BaseModel):
    """One cascade promotion: a rejected service that is in fact down."""

    model_config = _STRICT
    service: str = Field(description="The rejected service to promote.")
    via_service: str = Field(
        description="The CONFIRMED upstream service whose path the cascade "
        "reaches this service through — the promotion is attached to the "
        "graph as an edge via_service -> service. Must be a currently "
        "confirmed service."
    )
    predicate: NodePredicate = Field(  # type: ignore[valid-type]
        description="The failure mode the promoted service exhibits; "
        "cascade promotions are typically service_unavailable or "
        "throughput_collapse."
    )
    rationale: str = Field(
        description="Why this service is genuinely degraded, citing data."
    )


class JudgeReview(BaseModel):
    """Whole-graph review verdict (promotion-only)."""

    model_config = _STRICT
    add: list[JudgePromotion] = Field(
        description="Currently-rejected services to PROMOTE — genuinely "
        "degraded on full-picture review (e.g. system-wide cascade "
        "unavailability). Empty if none."
    )
    suggested_remove: list[str] = Field(
        default_factory=list,
        description="Audit-only: confirmed services whose evidence looks "
        "weak. NOT applied to the graph — per-edge verdicts are "
        "authoritative. Empty if none.",
    )
    rationale: str = Field(
        description="Per-service justification for each add/suggestion, "
        "citing data."
    )


def install(api: ExtensionAPI, config: JudgeFinalizeConfig) -> None:
    async def _submit_judge(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        try:
            review = JudgeReview.model_validate(args)
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
        return ToolTerminate(
            result=ToolResult(
                content=[TextContent(
                    type="text",
                    text=review.model_dump_json(),
                )]
            ),
            reason="judge:review-submitted",
        )

    api.register_tool(
        FunctionTool(
            name="submit_judge_review",
            description=(
                "Submit your whole-graph review. Promotion-only: each "
                "entry in `add` names a rejected service, the confirmed "
                "upstream it cascades through (via_service), and the "
                "failure mode it exhibits. `suggested_remove` is "
                "audit-only and not applied."
            ),
            parameters=JudgeReview,
            fn=_submit_judge,
        )
    )


__all__: Final = ["MANIFEST", "install"]
