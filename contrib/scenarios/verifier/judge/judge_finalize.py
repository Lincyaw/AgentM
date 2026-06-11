"""Whole-graph review submission tool for the verifier/judge scenario."""

from __future__ import annotations

import json
from typing import Any, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="judge_finalize",
    description="Submit tool for whole-graph judge review.",
    registers=("tool:submit_judge_review",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
)

_STRICT = ConfigDict(extra="forbid")

class JudgeReview(BaseModel):
    """Whole-graph review verdict."""

    model_config = _STRICT
    remove: list[str] = Field(
        description="Currently-confirmed services to DEMOTE — their evidence is "
        "not genuine degradation of that service (throughput-only drop, "
        "tiny/non-commensurate latency, infra egress double-count). Empty if none."
    )
    add: list[str] = Field(
        description="Currently-rejected services to PROMOTE — genuinely degraded "
        "on full-picture review (e.g. system-wide cascade unavailability). "
        "Empty if none."
    )
    rationale: str = Field(
        description="Per-service justification for each remove/add, citing data."
    )

class JudgeFinalizeConfig(TypedDict):
    pass

class JudgeReviewPayload(TypedDict):
    remove: list[str]
    add: list[str]
    rationale: str

def install(api: ExtensionAPI, config: JudgeFinalizeConfig) -> None:
    async def _submit_judge(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        payload = cast(JudgeReviewPayload, args)
        try:
            review = JudgeReview.model_validate(payload)
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
                "Submit the whole-graph review: `remove` lists confirmed "
                "services to demote, `add` lists rejected services to "
                "promote. Either list may be empty."
            ),
            parameters=JudgeReview,
            fn=_submit_judge,
        )
    )

__all__ = ["MANIFEST", "install"]
