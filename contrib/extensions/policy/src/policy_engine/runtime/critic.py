# code-health: ignore-file[AM025] -- tool args and model JSON are untyped
"""One critic, four moments it can be called.

The critic's identity is the ``critic`` manifest beside this file: a second
reader that tries to break what it is handed and endorses only what survives.
This module is everything around that identity — the four entries, what each
hands over, and how each answer comes back:

* **plan** (``PlanCritic``, standing in as ``plan_mode``'s approver) — no code
  exists yet; the verdict decides whether the agent may start writing. The only
  entry with no diff in reach, which is why it is the only one that can be
  asked about direction rather than damage.
* **submit** (``Critic.review`` with ``build_prompt``) — the agent has declared
  the task finished; the verdict decides whether the session may end.
* **rework** (``Critic.review`` with ``build_revision_prompt``) — the agent
  just rebuilt something it had already built; a rejection is injected as a
  mid-task counterexample.
* **on request** (the ``submit_for_review`` tool) — the agent submits its own
  reasoning mid-task and receives findings as tool output.

The reviewer can open the repo and run inside the sandbox, because reading a
trajectory tells you what the agent *said* it did: one run shipped an inverted
nesting order and described it as fixed, and no summary-level reading
distinguishes that from a correct fix.

``submit_for_review`` is a form, not a dispatch tool. An agent handed
``dispatch_agent`` writes to a subordinate: it appends a brief, grants
permissions, and — most expensively — states what the correct output would be,
which the reviewer then verifies the change against. Measured on one run:
author, implementation and acceptance criterion were the same person, and the
review could not have failed. The form has no place to put any of that: the
fields are the reasoning and only the reasoning, and what correct output looks
like is the reviewer's to establish.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.abi import (
    AgentSessionConfig,
    AtomAPI,
    FunctionTool,
    JsonValue,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from policy_engine.shared.manifest import load_manifest

#: Purpose marker on the reviewer's session. The policy atom checks it so a
#: reviewer does not install the policy that spawned it — that child would
#: register its own submit, run its own tagger, and review its own reviewer.
#: The value predates the unified critic and stays for continuity with
#: recorded sessions.
PURPOSE = "acceptance"

#: Room a mid-task finding may take in the agent's context. Enough for the
#: reviewer's own summary of what it ran, not for the output of running it.
_EVIDENCE_LIMIT = 1200


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[…truncated]"


# -- The verdict ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CriticVerdict:
    accepted: bool
    #: What the reviewer actually saw. Required in both directions — three
    #: reviews in a row accepted a fix whose nesting was inverted, and the
    #: transcripts show why: one wrote a probe program and never ran it, one
    #: ran a probe that printed only startup noise and then read a cached test
    #: result. Each believed it had checked. Having to quote the output is the
    #: cheapest thing that distinguishes looking from intending to look.
    evidence: str = ""
    finding: str = ""
    next_step: str = ""
    #: What establishes the standard this verdict was measured against, and
    #: what else the task text would permit. Empty means the reviewer never
    #: examined its own criterion, which four measured reviews did not: each
    #: opened by deriving a rule and then spent every turn applying it.
    criterion: str = ""
    #: The reviewer's own answer to whether the criterion is decidable from
    #: what it can see. False is a finding in its own right and the one output
    #: the old contract had no room for.
    criterion_settled: bool = True

    @property
    def needs_attention(self) -> bool:
        """Worth putting in front of the agent.

        A rejection, or an accepted verdict whose criterion the reviewer could
        not settle. The second is the case the old contract dropped: four
        measured reviews each picked a standard, none could have said it was
        undecidable, and all four picked the one grading rejects. An admission
        of ambiguity reaches the agent while there is still time to resolve it;
        a confident guess reaches it as fact.
        """
        return not self.accepted or not self.criterion_settled

    def as_ambiguity_message(self) -> str:
        """What the agent is told when the reviewer could not settle the standard."""
        return "\n".join(
            [
                (
                    "Before you go further — the requirement reads two ways "
                    "and I could not tell which is meant."
                ),
                "",
                self.criterion.strip() or "(no readings given)",
                "",
                (
                    "Work out which one the task wants before building further "
                    "on either. If nothing in the repository decides it, say "
                    "so in your summary rather than picking silently."
                ),
            ]
        )

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence.strip())

    def as_message(self) -> str:
        """What the agent is told when the submission does not hold up."""
        parts = ["Not quite — one thing is still open before this can go in."]
        if self.finding:
            parts += ["", self.finding]
        if self.next_step:
            parts += ["", f"Next: {self.next_step}"]
        parts += [
            "",
            (
                "Do that, then call `submit` again. If you think this is "
                "already covered, call `submit` and tell me where."
            ),
        ]
        return "\n".join(parts)

    def as_revision_message(self) -> str:
        """What the agent is told mid-task, where there is nothing to resubmit.

        Separate from ``as_message`` because that one ends by naming ``submit``,
        and at a revision point the agent has not submitted and must not be
        pointed at finishing. The close here hands the work back rather than
        asking for a reply -- an injected message that reads as a question gets
        answered, and answering it is how a run ends early.
        """
        parts = ["Hold on — a case that the rework does not survive:"]
        if self.finding:
            parts += ["", self.finding]
        if self.evidence.strip():
            # Capped, unlike at submit, where the verdict ends the turn either
            # way. This one lands mid-task in a context the agent still has to
            # work in, and the reviewer is asked for one line per input it
            # tried -- an unbounded paste of probe output would cost more room
            # than the finding is worth.
            parts += ["", _clip(self.evidence.strip(), _EVIDENCE_LIMIT)]
        if self.next_step:
            parts += ["", f"Next: {self.next_step}"]
        parts += [
            "",
            (
                "Decide what it means for the shape you just settled on, then "
                "carry on with the task. No need to reply to this."
            ),
        ]
        return "\n".join(parts)


@dataclass(slots=True)
class _VerdictSink:
    verdict: CriticVerdict | None = None


#: Environment variable naming a file of notes for the repository under review.
#: One name for every entry: the knowledge is about the repository, not about
#: which checkpoint happens to be asking.
NOTES_ENV = "AGENTM_REVIEW_NOTES"


def repository_notes(env_var: str = NOTES_ENV, fallback: str = "") -> str:
    """What is known about this repository, from the file the host named.

    A named file that is missing is worth a line in the log and nothing more: a
    review without local knowledge is the review this started as, not a broken
    one.
    """
    path = os.environ.get(env_var, "").strip()
    if not path:
        return fallback
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("critic: no notes at {}: {}", path, exc)
        return fallback


def with_notes(prompt: str, notes: str) -> str:
    """Prepend what is known about the repository to any entry's framing.

    Every entry, not just the one the agent calls. Notes reached
    ``submit_for_review`` alone and the three verdict entries -- plan, rework,
    submit -- never saw them, which left local knowledge available only when
    the agent chose to ask for it. Ahead of the framing on purpose: it is a
    premise for reading everything that follows, not an afterthought.
    """
    if not notes.strip():
        return prompt
    return (
        "## Known about this repository\n\n"
        f"{notes.strip()}\n\n"
        "That is established, and holds regardless of what the code or its "
        "tests currently say.\n\n" + prompt
    )


# -- Entry framings ------------------------------------------------------------


def build_prompt(
    *,
    task: str,
    summary: str,
    events: Sequence[str],
    concerns: Sequence[str],
) -> str:
    """The submit entry's framing. The critic's system prompt covers what
    holds at every checkpoint; what is specific to this one — what the log
    contains, what a concern demands, how the answer comes back — arrives with
    the material it describes."""
    parts = [
        (
            "A software-engineering agent has just declared this task "
            "finished. Decide whether that holds up, before the work is "
            "released. The run below lists every step as an event — files "
            "read, files edited with a line count, commands with their exit "
            "status, and what the agent said. File bodies and command output "
            "are not included: use the workspace, not the log, to see what "
            "the code does."
        ),
        f"## Task\n{task.strip()}",
    ]
    if summary.strip():
        parts.append(f"## The agent's closing summary\n{summary.strip()}")
    if concerns:
        joined = "\n\n".join(f"- {c.strip()}" for c in concerns)
        parts.append(
            f"## Process concerns raised during the run\n{joined}\n\n"
            "A concern that names a real gap is closed by an action in the "
            "run, never by reasoning alone; one that genuinely does not "
            "apply here is not a finding."
        )
    parts.append("## The run\n" + "\n".join(events))
    parts.append(
        "When you have decided, call `submit_verdict` exactly once. Accept "
        "only when you tried to break the work across the range and could "
        "not."
    )
    return "\n\n".join(parts)


def build_plan_prompt(*, task: str, plan: str) -> str:
    """The plan checkpoint, and the only one where no code exists yet.

    That absence is the point. Every other entry hands the reviewer a change,
    and a reviewer holding a change reaches for the cheapest reference within
    arm's length: the code before it. Measured on this bench, both reviewers of
    one task opened with ``git diff`` and built a pre-change copy to compare
    against, which answers "did this break anything" and cannot answer "was
    this the right direction" -- and on a task whose whole point is to change
    existing behaviour, the two questions have opposite answers. One review
    derived a rule exactly inverted from the graded one, verified the change
    against it thoroughly, and endorsed it.

    With no diff to anchor on, the only available reference is the task, which
    is where the direction actually lives. So this entry asks about the
    direction and nothing else -- not whether the steps are complete, not
    whether the approach is efficient. A plan can be vague and still point the
    right way; a precise plan pointed the wrong way is the failure this
    checkpoint exists to catch.
    """
    return "\n\n".join(
        [
            (
                "A software-engineering agent has written a plan and cannot "
                "change any file until it is approved. No code exists yet, so "
                "there is nothing to test and nothing to compare against.\n\n"
                "Judge one thing: whether the behaviour this plan aims at is "
                "the behaviour the task asks for. Not whether the steps are "
                "complete, not whether the approach is elegant."
            ),
            f"## Task\n{task.strip()}",
            f"## The plan\n{plan.strip()}",
            (
                "The plan asserts, somewhere, what correct output looks like. "
                "Find that assertion and ask what establishes it.\n\n"
                "The repository's own tests are not automatically the answer. "
                "Where a task asks to change existing behaviour, the tests "
                "that pin that behaviour are the thing being changed, and "
                "reading them as the specification points exactly backwards. "
                "Check what they actually cover: a task describing a case they "
                "never exercise is a task they cannot speak to.\n\n"
                "You may read and run anything that does not change a file."
            ),
            (
                "Call `submit_verdict` exactly once. `accepted: false` when the "
                "plan aims at the wrong behaviour, or when what it treats as "
                "correct rests on nothing -- say which, in `finding`, and in "
                "`next_step` the one thing that would settle it. `accepted: "
                "true` when the direction holds up; put in `evidence` what you "
                "checked it against."
            ),
        ]
    )


def build_revision_prompt(*, task: str, events: Sequence[str]) -> str:
    """The rework checkpoint's framing: work in progress, not finished work.

    The difference from submit is the tense. Nothing has been declared done,
    so there is no claim to test and no verdict to pass on the work as a whole
    -- asking for one here gets a reviewer that either rubber-stamps an
    unfinished change or rejects it for being unfinished. What is worth asking
    is narrower: the agent has just rebuilt something it had already built, so
    the first shape was wrong, and the question is whether the second one is
    wrong too.

    The scope is deliberately the rework rather than the whole change. A
    reviewer given the run entire at the halfway mark reports on whatever is
    least finished, which is everything, and its findings land where the agent
    was going to go anyway.
    """
    return "\n\n".join(
        [
            (
                "A software-engineering agent is part-way through a task and "
                "has just reworked something it had already built. It has not "
                "finished and is not asking for approval, so do not rule on "
                "whether the change is complete.\n\n"
                "Rebuilding means the first shape was wrong. Your question is "
                "whether the second one is wrong too, and the way to answer it "
                "is a case that breaks it -- run in the workspace, not argued. "
                "One counterexample settles it."
            ),
            f"## Task\n{task.strip()}",
            "## The run so far\n" + "\n".join(events),
            (
                "Call `submit_verdict` exactly once. `accepted: false` when you "
                "have a case that breaks the rework -- put the case and its "
                "output in `evidence`. `accepted: true` when you tried and "
                "could not, and say in `evidence` what you tried: this agent "
                "is mid-task and a reviewer that invents work to report costs "
                "it the rest of its budget."
            ),
        ]
    )


def _verdict_tool(sink: _VerdictSink) -> FunctionTool:
    async def submit_verdict(args: dict[str, JsonValue]) -> ToolTerminate:
        accepted = bool(args.get("accepted", True))
        sink.verdict = CriticVerdict(
            accepted=accepted,
            evidence=str(args.get("evidence", "")),
            finding=str(args.get("finding", "")),
            next_step=str(args.get("next_step", "")),
            criterion=str(args.get("criterion", "")),
            criterion_settled=bool(args.get("criterion_settled", True)),
        )
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text="Recorded.")]),
            reason="policy:acceptance-verdict",
        )

    return FunctionTool(
        name="submit_verdict",
        description="Record your decision and finish. Call this exactly once.",
        parameters={  # code-health: ignore[AM011]
            "type": "object",
            "properties": {
                "accepted": {
                    "type": "boolean",
                    "description": "True when the work does what the task asked.",
                },
                "evidence": {
                    "type": "string",
                    "description": (
                        "The rule you derived from the task, then one line per "
                        "input you tried: the input, what the pre-change code "
                        "printed, what the current code printed, and which side "
                        "of the discriminator that puts it on. Required "
                        "whichever way you decide. One line means one input, "
                        "which does not support acceptance."
                    ),
                },
                "finding": {
                    "type": "string",
                    "description": "When rejecting, the specific gap.",
                },
                "next_step": {
                    "type": "string",
                    "description": "When rejecting, the one action that closes it.",
                },
                "criterion": {
                    "type": "string",
                    "description": (
                        "Before the inputs: name the standard you judged "
                        "against, then the other reading of the task that "
                        "would give a different standard, then what in the "
                        "repository distinguishes them. Existing tests are "
                        "evidence of current behaviour, not of required "
                        "behaviour, and a task asking to change behaviour is "
                        "asking to change what they assert."
                    ),
                },
                "criterion_settled": {
                    "type": "boolean",
                    "description": (
                        "False when nothing you can see decides between the "
                        "readings. Answer false rather than picking one: a "
                        "standard you chose is a standard the agent will be "
                        "held to, and saying you could not tell is worth more "
                        "than a confident guess defended at length."
                    ),
                },
            },
            "required": ["accepted", "evidence", "criterion", "criterion_settled"],
        },
        fn=submit_verdict,
    )


# -- Verdict entries: submit and rework ----------------------------------------


async def _shutdown(child: object) -> None:
    """Close a review session, and never let closing it be the failure."""
    closer = getattr(child, "shutdown", None)  # code-health: ignore[AM021]
    if closer is None:
        return
    try:
        await closer()
    except Exception as exc:  # noqa: BLE001 - the verdict already stands
        logger.warning("critic: review session did not close: {}", exc)


@dataclass(slots=True)
class Critic:
    api: AtomAPI
    #: Provider registry name to run the reviewer on. Empty inherits whatever
    #: the child would get by default, which is the session's active provider
    #: — not necessarily the one the reviewed agent ran on. Resolved per review
    #: rather than at construction: the atom installs in the POLICY band,
    #: before providers register.
    provider: str = ""
    #: Composition to run the reviewer under — the same shell the
    #: ``submit_for_review`` entry names in its config, read-only and without
    #: the worker's own system prompt. Empty inherits the parent's full
    #: composition, which hands the reviewer the worker's write tools and
    #: prepends the worker's prompt to the critic's; it exists as the default
    #: only for hosts that register no critic shell.
    scenario: str = ""
    #: Ceiling on one review. Observed reviews run ten to fifteen minutes, so
    #: this only bites on a hang -- and a hang is the case that matters, since
    #: the verdict entries block the agent until they return.
    timeout_sec: float = 1200.0
    #: Fallback notes when no file is named, for a single-repository host.
    notes: str = ""

    async def review(self, prompt: str) -> CriticVerdict:
        """Accepts on any failure of its own.

        A reviewer that breaks must not be able to hold a session open, so
        every path that is not an explicit rejection returns acceptance.
        """
        sink = _VerdictSink()
        prompt = with_notes(prompt, repository_notes(fallback=self.notes))
        try:
            manifest = load_manifest("critic")
            # Only the verdict tool is added — it is this entry's output
            # contract, a closure over the sink, so it is owned here rather
            # than named in the manifest. The child's other tools come from
            # its composition, routed to the same sandbox — passing our own
            # `bash` collided with tool_bash and failed the whole spawn.
            model = None
            stream_fn = None
            if self.provider:
                resolved = self.api.get_provider(self.provider)
                if resolved is None:
                    logger.warning(
                        "critic: provider {!r} not registered; "
                        "reviewing on the session default",
                        self.provider,
                    )
                else:
                    logger.info("critic: reviewing on provider {}", resolved.name)
                    model = resolved.model
                    stream_fn = resolved.stream_fn
            # spawn_child_session rather than spawn: spawn refuses to combine
            # a scenario change with a model override, and this entry wants
            # both — the read-only shell and the reviewer's own provider.
            child = await self.api.spawn_child_session(
                AgentSessionConfig(
                    cwd=self.api.ctx.cwd,
                    scenario=self.scenario or None,
                    system=manifest.system,
                    extra_tools=[_verdict_tool(sink)],
                    purpose=PURPOSE,
                    model=model,
                    stream_fn=stream_fn,
                )
            )
            try:
                await asyncio.wait_for(child.run(prompt), timeout=self.timeout_sec)
            finally:
                # The session outlives the await on every path -- timeout,
                # cancellation, a verdict -- and nothing else closes it. The
                # request entry has always done this; the verdict entry never
                # did, so a review that hung left a child holding a sandbox
                # and burning tokens for the rest of the run.
                await _shutdown(child)
        except TimeoutError:
            logger.warning(
                "critic: review exceeded {}s; accepting and moving on",
                self.timeout_sec,
            )
            return CriticVerdict(accepted=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("critic: review failed: {}", exc)
            return CriticVerdict(accepted=True)

        if sink.verdict is None:
            logger.info("critic: reviewer finished without a verdict; accepting")
            return CriticVerdict(accepted=True)
        return sink.verdict


# -- The on-request entry: submit_for_review -----------------------------------


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
)


class _SubmitForReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        min_length=1,
        description=(
            "The task as it was given to you, quoted rather than summarised, "
            "and where it applies."
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


def _request_prompt(request: _SubmitForReview, notes: str = "") -> str:
    """The on-request entry's framing: the work is open, nothing is graded
    yet, and the answer wanted is counterexamples while there is still time to
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
        f"## {head}\n\n{text}" for head, text in sections if text.strip()
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
            await child.run(_request_prompt(request, self._repository_notes()))
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


#: The service name ``plan_mode`` looks its approver up under. A string rather
#: than an import: atom-to-atom coupling is forbidden, and the name is that
#: atom's published surface.
PLAN_APPROVER_SERVICE = "plan_mode.approver"


@dataclass(frozen=True, slots=True)
class PlanReview:
    """What ``plan_mode`` reads back. Structurally its ``PlanVerdict``.

    Not that class, because importing it would couple two atoms. The protocol
    it satisfies is ``runtime_checkable`` and the value is only read for these
    two fields, which is what makes the decoupling hold rather than merely
    typecheck.
    """

    approved: bool
    feedback: str = ""


@dataclass(slots=True)
class PlanCritic:
    """The critic, standing in as ``plan_mode``'s approver.

    The one checkpoint that can reach the direction, because it runs before
    there is a diff to mistake for a specification. A rejection costs the agent
    a plan revision and no code, which is the cheapest correction available
    anywhere in a run.

    Fails open, like every other entry: a plan held hostage by a broken
    reviewer is worse than an unreviewed plan.
    """

    critic: Critic
    task_for: Callable[[], str]

    async def review(self, plan: str) -> PlanReview:
        verdict = await self.critic.review(
            build_plan_prompt(task=self.task_for(), plan=plan)
        )
        if not verdict.criterion_settled:
            # Checked before `accepted`, and this is the entry where it matters
            # most: a plan built on a standard nobody could settle is a plan
            # whose every later step inherits the ambiguity.
            logger.info("plan critic: criterion undecided, sending the plan back")
            return PlanReview(approved=False, feedback=verdict.as_ambiguity_message())
        if verdict.accepted:
            logger.info("plan critic: direction holds up")
            return PlanReview(approved=True)
        logger.info("plan critic: sent back -- {}", verdict.finding[:160])
        parts = [verdict.finding or "The plan aims at the wrong behaviour."]
        if verdict.evidence.strip():
            parts.append(_clip(verdict.evidence.strip(), _EVIDENCE_LIMIT))
        if verdict.next_step:
            parts.append(f"Next: {verdict.next_step}")
        return PlanReview(approved=False, feedback="\n\n".join(parts))


__all__ = [
    "MANIFEST",
    "PLAN_APPROVER_SERVICE",
    "PURPOSE",
    "Critic",
    "CriticVerdict",
    "PlanCritic",
    "PlanReview",
    "ReviewRequestConfig",
    "build_plan_prompt",
    "build_prompt",
    "build_revision_prompt",
    "install",
]
