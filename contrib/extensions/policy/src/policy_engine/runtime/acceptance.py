# code-health: ignore-file[AM025] -- tool args and model JSON are untyped
"""Acceptance review: a reviewer that can open the repo, run inside the sandbox.

Reading a trajectory tells you what the agent *said* it did. Whether the work
does what the task asked is a question about the code, and the two come apart
exactly where it matters: one run here shipped an inverted nesting order and
described it as fixed, and no summary-level reading distinguishes that from a
correct fix. So the reviewer gets a shell in the same sandbox and is expected
to look — render the named scenario, run the pre-existing suite, read the file.

It runs as a child session with two tools and nothing else. Notably not the
parent's own tools: those include ``submit``, and a reviewer able to end the
session it is reviewing is a loop waiting to happen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from agentm.core.abi import (
    AtomAPI,
    FunctionTool,
    JsonValue,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from policy_engine.shared.manifest import load_manifest

#: Purpose marker on the reviewer's session. The policy atom checks it so a
#: reviewer does not install the policy that spawned it — that child would
#: register its own submit, run its own tagger, and review its own reviewer.
PURPOSE = "acceptance"
_PURPOSE = PURPOSE


@dataclass(frozen=True, slots=True)
class AcceptanceVerdict:
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
            parts += ["", self.evidence.strip()]
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
    verdict: AcceptanceVerdict | None = None


def build_prompt(
    *,
    task: str,
    summary: str,
    events: Sequence[str],
    concerns: Sequence[str],
) -> str:
    """The acceptance entry's framing. The critic's system prompt covers what
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


def build_revision_prompt(*, task: str, events: Sequence[str]) -> str:
    """The revision checkpoint's framing: work in progress, not finished work.

    A third entry to the same critic, and the difference from acceptance is the
    tense. Nothing has been declared done, so there is no claim to test and no
    verdict to pass on the work as a whole -- asking for one here gets a
    reviewer that either rubber-stamps an unfinished change or rejects it for
    being unfinished. What is worth asking is narrower: the agent has just
    rebuilt something it had already built, so the first shape was wrong, and
    the question is whether the second one is wrong too.

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
        sink.verdict = AcceptanceVerdict(
            accepted=accepted,
            evidence=str(args.get("evidence", "")),
            finding=str(args.get("finding", "")),
            next_step=str(args.get("next_step", "")),
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
            },
            "required": ["accepted", "evidence"],
        },
        fn=submit_verdict,
    )


@dataclass(slots=True)
class AcceptanceReviewer:
    api: AtomAPI
    #: Provider registry name to run the reviewer on. Empty inherits whatever
    #: the child would get by default, which is the session's active provider
    #: — not necessarily the one the reviewed agent ran on. Resolved per review
    #: rather than at construction: the atom installs in the POLICY band,
    #: before providers register.
    provider: str = ""

    async def review(self, prompt: str) -> AcceptanceVerdict:
        """Accepts on any failure of its own.

        A reviewer that breaks must not be able to hold a session open, so
        every path that is not an explicit rejection returns acceptance.
        """
        sink = _VerdictSink()
        try:
            manifest = load_manifest("critic")
            # Only the verdict tool is added — it is this entry's output
            # contract, a closure over the sink, so it is owned here rather
            # than named in the manifest. A child inherits the scenario's
            # extensions, so it already has the workspace tools routed to the
            # same sandbox — passing our own `bash` collided with tool_bash and
            # failed the whole spawn.
            model = None
            stream_fn = None
            if self.provider:
                resolved = self.api.get_provider(self.provider)
                if resolved is None:
                    logger.warning(
                        "acceptance: provider {!r} not registered; "
                        "reviewing on the session default",
                        self.provider,
                    )
                else:
                    logger.info("acceptance: reviewing on provider {}", resolved.name)
                    model = resolved.model
                    stream_fn = resolved.stream_fn
            child = await self.api.spawn(
                purpose=_PURPOSE,
                tools=[_verdict_tool(sink)],
                system=manifest.system,
                max_turns=manifest.max_turns,
                model=model,
                stream_fn=stream_fn,
            )
            await child.run(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("acceptance: review failed: {}", exc)
            return AcceptanceVerdict(accepted=True)

        if sink.verdict is None:
            logger.info("acceptance: reviewer finished without a verdict; accepting")
            return AcceptanceVerdict(accepted=True)
        return sink.verdict


__all__ = [
    "AcceptanceReviewer",
    "AcceptanceVerdict",
    "build_prompt",
]
