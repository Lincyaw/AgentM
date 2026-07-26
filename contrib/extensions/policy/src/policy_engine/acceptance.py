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

from .manifest import load_manifest

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
            "Do that, then call `submit` again. If you think this is already "
            "covered, call `submit` and tell me where.",
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
    parts = [f"## Task\n{task.strip()}"]
    if summary.strip():
        parts.append(f"## The agent's closing summary\n{summary.strip()}")
    if concerns:
        joined = "\n\n".join(f"- {c.strip()}" for c in concerns)
        parts.append(f"## Process concerns raised during the run\n{joined}")
    parts.append("## The run\n" + "\n".join(events))
    return "\n\n".join(parts)


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
                        "The output you actually saw, quoted — the command and "
                        "what it printed. Required whichever way you decide."
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

    async def review(self, prompt: str) -> AcceptanceVerdict:
        """Accepts on any failure of its own.

        A reviewer that breaks must not be able to hold a session open, so
        every path that is not an explicit rejection returns acceptance.
        """
        sink = _VerdictSink()
        try:
            manifest = load_manifest("acceptance")
            # Only the verdict tool is added. A child inherits the scenario's
            # extensions, so it already has the workspace tools routed to the
            # same sandbox — passing our own `bash` collided with tool_bash and
            # failed the whole spawn.
            child = await self.api.spawn(
                purpose=_PURPOSE,
                tools=[_verdict_tool(sink)],
                system=manifest.system,
                max_turns=manifest.max_turns,
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
