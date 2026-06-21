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
    """One cascade promotion: a rejected service with a real target anomaly."""

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
        "cascade promotions are typically process_killed, "
        "flow_interrupted, or resource_pool_exhausted."
    )
    rationale: str = Field(
        description="Why this service is genuinely degraded, citing data. "
        "Do not promote ordinary proportional reduced demand from a slow "
        "caller; that belongs in edge evidence, not as a final node."
    )


class JudgeReEval(BaseModel):
    """Request re-evaluation of an edge with global context."""

    model_config = _STRICT
    service: str = Field(
        description="The inconclusive/rejected service to re-evaluate."
    )
    via_service: str = Field(
        description="The confirmed upstream service this edge comes from."
    )
    context: str = Field(
        description="Global context to provide the hop agent on re-evaluation. "
        "Explain WHY its prior verdict may be wrong given the full graph "
        "(e.g. 'all upstream services on the path to this service are "
        "confirmed dead — zero traffic is expected cascade, not just fewer "
        "calls')."
    )


class JudgeSeedReEval(BaseModel):
    """Request re-evaluation of an injection seed with global context."""

    model_config = _STRICT
    seed: str = Field(
        description="The injection seed id to re-evaluate, exactly as shown "
        "in the Seed verification results section."
    )
    context: str = Field(
        description="Global entry/whole-graph context to provide the seed "
        "agent. Explain why the prior seed verdict may be wrong and what "
        "data shape to verify. Do not request a seed re-check without a "
        "specific entry symptom or causal clue."
    )


class JudgeReview(BaseModel):
    """Whole-graph review verdict."""

    model_config = _STRICT
    entry_explanation: str = Field(
        description="Whether and how the current confirmed graph explains "
        "the entry/frontend observations. Mention the entry endpoints and "
        "symptom shape checked. If there is no meaningful entry anomaly, "
        "say so explicitly."
    )
    unexplained_entry_observations: list[str] = Field(
        default_factory=list,
        description="Entry/frontend symptoms that the current graph does "
        "not explain. Empty means the graph explains the entry symptoms or "
        "there is no meaningful entry anomaly."
    )
    add: list[JudgePromotion] = Field(
        description="Services to PROMOTE directly — you have enough "
        "evidence from the global view to confirm them without "
        "re-investigation (e.g. a user-visible path failure, selective "
        "endpoint disappearance, timeout/error/fail-fast evidence). "
        "Do not use direct promotion for proportional traffic drops alone. "
        "Empty if none."
    )
    re_evaluate: list[JudgeReEval] = Field(
        default_factory=list,
        description="Inconclusive or rejected edges to send BACK to a hop "
        "agent for re-investigation with your global context. Use when "
        "you believe the prior verdict is wrong but want the hop agent "
        "to verify with data queries. Prefer this over direct promotion "
        "when the evidence needs re-examination. Do not request "
        "re-evaluation solely because the callee received proportionally "
        "fewer requests from a slow caller.",
    )
    re_evaluate_seeds: list[JudgeSeedReEval] = Field(
        default_factory=list,
        description="Rejected or inconclusive injection seeds to send BACK "
        "to a seed agent for re-investigation with your global context. "
        "Use this when the current graph is empty or cannot explain a real "
        "entry/frontend symptom and a prior seed verdict may have missed "
        "caller-side, resource, log, or endpoint-level evidence. Empty if "
        "the seed rejection is consistent with the data.",
    )
    suggested_remove: list[str] = Field(
        default_factory=list,
        description="Audit-only: confirmed services whose evidence looks "
        "weak. NOT applied to the graph — per-edge verdicts are "
        "authoritative. Empty if none.",
    )
    rationale: str = Field(
        description="Per-service justification for each add/re_evaluate/"
        "suggestion, citing data."
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
                "Submit your whole-graph review. `add` promotes services "
                "directly. `re_evaluate` sends edges back to hop agents "
                "with your global context for re-investigation (preferred "
                "when evidence needs re-examination). `re_evaluate_seeds` "
                "sends rejected or inconclusive injection seeds back to seed "
                "agents. `suggested_remove` is audit-only and not applied. "
                "Proportional reduced demand alone is not enough to add or "
                "re-evaluate a node."
            ),
            parameters=JudgeReview,
            fn=_submit_judge,
        )
    )


__all__: Final = ["MANIFEST", "install"]
