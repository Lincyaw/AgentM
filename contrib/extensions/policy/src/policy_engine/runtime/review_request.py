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

The reviewer is the same critic that rules on the finished work at submit: its
identity is the ``critic`` manifest in this package, and the scenario named in
the config supplies only the shell it runs in -- the sandbox binding, the
read-only tool set, the retry policy. One prompt, two checkpoints, so what the
mid-run reader doubts and what the acceptance reader breaks cannot drift apart.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.abi import (
    AgentSessionConfig,
    AtomAPI,
    AtomInstallPriority,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.extensions import ExtensionManifest
from policy_engine.shared.manifest import load_manifest


class ReviewRequestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    #: Composition the review runs under. It is a different agent from the one
    #: submitting: read-only, and told to establish the requirement itself.
    scenario: str = ""

    #: Environment variable naming a file of notes for the repository this run
    #: is against. Read at dispatch rather than at install so one scenario can
    #: serve many repositories: the notes that matter to a database are noise
    #: to a syntax highlighter, and a single blob for both says nothing precise
    #: about either.
    notes_env: str = "AGENTM_REVIEW_NOTES"

    #: What a reviewer has to know about this repository and could not have
    #: reasoned its way to. The reviewer's prompt names no repository on
    #: purpose, so it reproduces a problem the way a stranger would -- and a
    #: stranger models whatever is expensive to stand up. One built a mocked
    #: interleaving to reproduce a race, confirmed the fix against it, and
    #: never saw that the same fix deadlocks against the real database,
    #: because the model it had built contained no locks. Local knowledge is
    #: what closes that, and this is where local knowledge arrives.
    notes: str = ""


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


def _prompt(request: _SubmitForReview, notes: str = "") -> str:
    """The mid-run entry's framing: the work is open, nothing is graded yet,
    and the answer wanted is counterexamples while there is still time to
    change course."""
    header = (
        "The author of the decision below is mid-task and waiting on your "
        "critique before going further. Nothing is finished and nothing is "
        "being graded yet: what they need is every way this decision gives a "
        "wrong answer, while changing course is still cheap. Reply with your "
        "findings, or with what you tried that failed to break it."
    )
    sections = [
        ("Task", request.task),
        ("How the author read it", request.reading),
        ("What they chose to do, and why", request.decision),
        ("What they rejected, and why", request.rejected),
        ("Paths changed", request.paths),
        ("Known about this repository", notes),
    ]
    body = "\n\n".join(
        f"## {head}\n\n{body}" for head, body in sections if body.strip()
    )
    return f"{header}\n\n{body}"


class _ReviewRuntime:
    def __init__(self, api: AtomAPI, config: ReviewRequestConfig) -> None:
        self._api = api
        self._scenario = config.scenario
        self._notes = config.notes
        self._notes_env = config.notes_env

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

    def _repository_notes(self) -> str:
        """Notes for the repository under review, from the file the host named.

        Falls back to the configured text, so a single-repository deployment
        need not set anything up. A named file that is missing is worth a line
        in the log and nothing more: a review without local knowledge is the
        review this started as, not a broken one.
        """
        path = os.environ.get(self._notes_env, "").strip()
        if not path:
            return self._notes
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("review_request: no notes at {}: {}", path, exc)
            return self._notes

    async def submit(self, args: Mapping[str, object]) -> ToolResult:
        try:
            request = _SubmitForReview.model_validate(args)
        except ValidationError as exc:
            return _failed(str(exc))

        try:
            # The critic's prompt comes from this package, the shell from the
            # named scenario. Loaded per dispatch: a failure here should cost
            # this review, not the install.
            manifest = load_manifest("critic")
            child = await self._api.spawn_child_session(
                AgentSessionConfig(
                    cwd=self._api.ctx.cwd,
                    scenario=self._scenario or None,
                    system=manifest.system,
                    purpose="review",
                )
            )
        except Exception as exc:  # noqa: BLE001 - the submitter should hear about it
            logger.warning("review_request: review could not start: {}", exc)
            return _failed(f"the review could not start: {exc or type(exc).__name__}")
        try:
            await child.run(_prompt(request, self._repository_notes()))
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
