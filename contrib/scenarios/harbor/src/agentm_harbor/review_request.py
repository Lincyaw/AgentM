"""A tool for submitting a decision to review, which is not a tool for running agents.

The distinction is the whole point. An agent handed something called
``dispatch_agent`` writes to a subordinate: it appends a brief, decides what the
reviewer should look for, grants it permissions it does not have, and -- most
expensively -- states what the correct output would be, which the reviewer then
verifies the change against. Measured on one run: the submitting agent wrote out
the exact HTML it expected, the reviewer's own stated requirement came back
carrying the same clause, and the review confirmed that the change produced the
string its author had asked for. Author, implementation and acceptance criterion
were the same person, and the review could not have failed.

An agent handed something called ``submit_for_review`` fills in a form. It has no
place to put an instruction and no field for the answer, so the reviewer receives
the reasoning and nothing else. That the form starts a session at all is not
something the submitter needs to know.

The fields are the reasoning and only the reasoning: how the task was read, what
was chosen, what was rejected. Those are what a reviewer has to have -- a change
pointed the wrong way is a reading that went wrong several steps earlier, and the
diff cannot show that. There is deliberately no field for what correct output
looks like. That is the reviewer's to establish.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentm.core.abi import (
    AgentSessionConfig,
    AtomAPI,
    AtomInstallPriority,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.extensions import ExtensionManifest
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ReviewRequestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    #: Composition the review runs under. It is a different agent from the one
    #: submitting: read-only, and told to establish the requirement itself.
    scenario: str = ""


MANIFEST = ExtensionManifest(
    name="review_request",
    description="Submit the decision taken so far for an independent review.",
    registers=("tool:submit_for_review",),
    config_schema=ReviewRequestConfig,
    requires=("atom:system_prompt",),
    priority=AtomInstallPriority.SERVICE,
)


class _SubmitForReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        min_length=1,
        description=(
            "The task as it was given to you, quoted rather than summarised, and where it applies."
        ),
    )
    reading: str = Field(
        min_length=1,
        description=(
            "How you read that task: what you take the requirement to be, and "
            "what in the task led you to read it that way."
        ),
    )
    decision: str = Field(
        min_length=1,
        description="What you chose to do about it, and why you chose that.",
    )
    rejected: str = Field(
        default="",
        description=(
            "What you considered and did not choose, and why each one lost. "
            "Empty if you considered nothing else."
        ),
    )
    paths: str = Field(
        default="",
        description="Paths you have changed so far, one per line.",
    )


def _prompt(request: _SubmitForReview) -> str:
    sections = [
        ("Task", request.task),
        ("How the author read it", request.reading),
        ("What they chose to do, and why", request.decision),
        ("What they rejected, and why", request.rejected),
        ("Paths changed", request.paths),
    ]
    return "\n\n".join(f"## {head}\n\n{body}" for head, body in sections if body.strip())


class _ReviewRuntime:
    def __init__(self, api: AtomAPI, config: ReviewRequestConfig) -> None:
        self._api = api
        self._scenario = config.scenario

    def install(self) -> None:
        # A review of a review is a review of the wrong thing, and the session
        # doing the reviewing is the one that would ask for it.
        if self._api.ctx.parent_session_id is not None:
            return
        self._api.register_tool(
            FunctionTool(
                name="submit_for_review",
                description=(
                    "Submit the decision you have taken so far for an "
                    "independent review, and wait for what it finds. The "
                    "reviewer works in this same workspace and can run things, "
                    "but cannot change them. It reaches its own conclusions "
                    "about what the task requires, so give it your reasoning "
                    "rather than your conclusion."
                ),
                parameters=_SubmitForReview,
                fn=self.submit,
            )
        )

    async def submit(self, args: Mapping[str, object]) -> ToolResult:
        try:
            request = _SubmitForReview.model_validate(args)
        except ValidationError as exc:
            return _failed(str(exc))

        child = await self._api.spawn_child_session(
            AgentSessionConfig(
                cwd=self._api.ctx.cwd,
                scenario=self._scenario or None,
                purpose="review",
            )
        )
        try:
            await child.run(_prompt(request))
            await child.idle()
            outcome = child.final_result()
            findings = outcome.text if outcome else ""
        except Exception as exc:  # noqa: BLE001 - the submitter should hear about it
            logger.warning("review_request: review failed: {}", exc)
            return _failed(f"the review did not finish: {exc or type(exc).__name__}")
        finally:
            try:
                await child.shutdown()
            except Exception as exc:  # noqa: BLE001 - already returning findings
                logger.warning("review_request: review session did not close: {}", exc)

        if not findings.strip():
            return _failed("the review returned nothing")
        return _findings(findings)


def _findings(text: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        extras={"findings": text},
    )


def _failed(reason: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=reason)],
        is_error=True,
        extras={"error": reason},
    )


def install(api: AtomAPI, config: ReviewRequestConfig) -> None:
    _ReviewRuntime(api, config).install()


__all__ = ("MANIFEST", "ReviewRequestConfig", "install")
