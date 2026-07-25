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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from loguru import logger

from agentm.core.abi import (
    AtomAPI,
    BashOperations,
    FunctionTool,
    JsonValue,
    TextContent,
    Tool,
    ToolResult,
    ToolTerminate,
)

from .manifest import load_manifest

_OUTPUT_LIMIT = 4000


@dataclass(frozen=True, slots=True)
class AcceptanceVerdict:
    accepted: bool
    finding: str = ""
    next_step: str = ""

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


def _bash_tool(bash: BashOperations, cwd: str) -> FunctionTool:
    async def run(args: dict[str, JsonValue]) -> ToolResult:
        cmd = args.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            return ToolResult(
                content=[TextContent(type="text", text="cmd is required")],
                is_error=True,
            )
        try:
            result = await bash.exec(cmd, cwd=cwd, timeout=180.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("acceptance: bash failed: {}", exc)
            return ToolResult(
                content=[TextContent(type="text", text=f"command failed: {exc}")],
                is_error=True,
            )
        out = (result.stdout + result.stderr).decode("utf-8", "replace")
        text = f"exit {result.exit_code}\n{out[:_OUTPUT_LIMIT]}"
        return ToolResult(
            content=[TextContent(type="text", text=text)],
            is_error=result.exit_code != 0,
        )

    return FunctionTool(
        name="bash",
        description=(
            "Run a shell command in the agent's workspace. Use it to read files, "
            "render a scenario, or run a test suite."
        ),
        parameters={  # code-health: ignore[AM011]
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
        fn=run,
    )


def _verdict_tool(sink: _VerdictSink) -> FunctionTool:
    async def submit_verdict(args: dict[str, JsonValue]) -> ToolTerminate:
        accepted = bool(args.get("accepted", True))
        sink.verdict = AcceptanceVerdict(
            accepted=accepted,
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
                "finding": {
                    "type": "string",
                    "description": "When rejecting, the specific gap.",
                },
                "next_step": {
                    "type": "string",
                    "description": "When rejecting, the one action that closes it.",
                },
            },
            "required": ["accepted"],
        },
        fn=submit_verdict,
    )


@dataclass(slots=True)
class AcceptanceReviewer:
    api: AtomAPI
    bash: BashOperations
    cwd: str

    def _tools(self, names: Sequence[str], sink: _VerdictSink) -> list[Tool]:
        """Build the tools the manifest asked for, and only those."""
        builders: dict[str, Callable[[], Tool]] = {
            "bash": lambda: _bash_tool(self.bash, self.cwd),
            "submit_verdict": lambda: _verdict_tool(sink),
        }
        unknown = [n for n in names if n not in builders]
        if unknown:
            raise ValueError(f"acceptance manifest names unknown tools: {unknown}")
        return [builders[n]() for n in names]

    async def review(self, prompt: str) -> AcceptanceVerdict:
        """Accepts on any failure of its own.

        A reviewer that breaks must not be able to hold a session open, so
        every path that is not an explicit rejection returns acceptance.
        """
        sink = _VerdictSink()
        try:
            manifest = load_manifest("acceptance")
            child = await self.api.spawn(
                purpose="acceptance",
                tools=self._tools(manifest.tools, sink),
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
