"""Level 2: Claim structure analysis built on top of Level 1 (index).

Runs a tool-using LLM session that reads the index's structured output
via tools and classifies each span as explore/commit/finalize. Produces
a claim ledger consumed by the auditor (Level 3).

Pipeline: Index (lexical) → Claim Analysis (semantic) → Auditor (localization)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import (
    AgentSessionConfig,
    FunctionTool,
    ToolTerminate,
)
from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Claim ledger schema
# ---------------------------------------------------------------------------


@dataclass
class ClaimLedger:
    span_roles: dict[int, str]
    commitment_points: list[dict[str, Any]]
    n_spans: int

    def to_context(self) -> dict[str, Any]:
        """Produce a summary for injection into auditor context_index."""
        return {
            "span_roles": self.span_roles,
            "commitment_points": self.commitment_points,
        }


# ---------------------------------------------------------------------------
# submit_claims tool
# ---------------------------------------------------------------------------


class _CommitmentPoint(BaseModel):
    span: int = Field(description="0-based span index")
    entity: str = Field(default="", description="Entity name committed here")
    grounded: bool = Field(default=False, description="Is this entity grounded by tool output?")
    reason: str = Field(default="", description="Why this is a commitment")


class _SubmitClaimsArgs(BaseModel):
    span_roles: dict[str, str] = Field(
        description='Map of span index (as string) to role: "explore", "commit", "verify", or "finalize"',
    )
    commitment_points: list[_CommitmentPoint] = Field(
        description="Spans where the agent commits to a claim (not exploration)",
    )


_LEDGER_HOLDER: list[ClaimLedger | None] = [None]


async def _submit_handler(args: dict[str, Any]) -> ToolTerminate:
    parsed = _SubmitClaimsArgs.model_validate(args)
    roles = {int(k): v for k, v in parsed.span_roles.items()}
    points = [p.model_dump() for p in parsed.commitment_points]
    _LEDGER_HOLDER[0] = ClaimLedger(
        span_roles=roles,
        commitment_points=points,
        n_spans=max(roles.keys()) + 1 if roles else 0,
    )
    from agentm.core.abi import TextContent, ToolResult
    return ToolTerminate(
        result=ToolResult(content=[TextContent(type="text", text="Claims submitted.")]),
        reason="claim_analysis:submitted",
    )


SUBMIT_CLAIMS_TOOL = FunctionTool(
    name="submit_claims",
    description="Submit the claim structure analysis. Call exactly once as your final action.",
    parameters=_SubmitClaimsArgs,
    fn=_submit_handler,
)


# ---------------------------------------------------------------------------
# Claim analysis prompt
# ---------------------------------------------------------------------------

CLAIM_PROMPT = """\
# Claim Structure Analysis

You are Level 2 in a layered trajectory analysis pipeline. Level 1 (lexical analysis) has already extracted entities and their grounding status. Your job is to classify each span's ROLE in the argument structure.

## Your tools

- `list_entities()` — see all entities from Level 1 with grounding status and reference counts
- `get_entity_timeline(name)` — see where a specific entity appears across spans, with reference kinds (mention vs tool_output)
- `get_turn(index)` — read a specific span's content (use sparingly, only for key spans)
- `list_attention_hints()` — see grounding warnings from Level 1
- `submit_claims(...)` — submit your analysis (call exactly once)

## Workflow

1. Call `list_entities()` to see the entity landscape and grounding status
2. Identify which entities appear in late spans (likely part of the final answer)
3. For entities that are ungrounded (no tool_output references), call `get_entity_timeline` to trace their lifecycle
4. Call `get_turn` on 1-3 key spans to verify commitment vs exploration — focus on the last span (final report) and any span where an ungrounded entity first appears as a commitment
5. Call `submit_claims` with your analysis

## Classification rules

- `explore`: search query, tool call, retrieval, candidate investigation — no conclusion drawn
- `verify`: agent claims to have verified or confirmed something
- `commit`: agent selects an answer, makes a decision, states a conclusion as settled
- `finalize`: final report or answer submission

Most spans are `explore`. Only a few are `commit`/`verify`/`finalize`.

## commitment_points

For each commit/verify/finalize span, report:
- Which entity is being committed to
- Whether that entity is grounded (from the index)
- Why this span is a commitment (not exploration)

Focus on entities that are UNGROUNDED but COMMITTED — these are the most likely error points.
"""


# ---------------------------------------------------------------------------
# Run claim analysis session
# ---------------------------------------------------------------------------


_CLAIM_TOOLS_EXT = "agentm_eval.methods._claim_tools_ext"


async def run_claim_analysis(
    trajectory: list[dict[str, Any]],
    *,
    symbols: list[dict[str, Any]],
    references: list[dict[str, Any]],
    context_index: dict[str, Any],
    model: str | None = None,
) -> ClaimLedger | None:
    """Run a tool-using claim analysis session on index output."""
    from agentm.core.runtime import AgentSession

    _OBS = "agentm.extensions.builtin.observability"
    _OPS = "agentm.extensions.builtin.operations"
    _INDEX_TOOLS = "llmharness.agents.auditor.index_tools"

    ci = context_index if isinstance(context_index, dict) else {}

    config = AgentSessionConfig(
        cwd=".",
        model=model,
        extensions=[
            (_OBS, {}),
            (_OPS, {}),
            (_INDEX_TOOLS, {
                "trajectory": trajectory,
                "symbols": symbols,
                "references": references,
                "context_index": context_index,
            }),
        ],
        purpose="claim_analysis",
        log_trace_command=False,
        extra_tools=[SUBMIT_CLAIMS_TOOL],
    )

    _LEDGER_HOLDER[0] = None

    session = await AgentSession.create(config)
    try:
        await session.prompt(CLAIM_PROMPT)
    except Exception as exc:
        logger.debug("claim analysis failed: {}", exc)
    finally:
        with contextlib.suppress(Exception):
            await session.shutdown()

    return _LEDGER_HOLDER[0]
