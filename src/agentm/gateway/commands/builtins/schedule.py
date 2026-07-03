"""``/schedule`` — manage durable gateway scheduled prompts for this chat."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..protocol import (
    CommandContext,
    CommandInvocation,
    CommandKind,
    CommandResult,
)


def _fmt_time(value: Any) -> str:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return "(unknown)"
    return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _job_line(job: dict[str, Any]) -> str:
    mode = "recurring" if job.get("recurring", True) else "once"
    status = "active" if job.get("enabled", True) else "paused"
    error = f" last_error={job['last_error']!r}" if job.get("last_error") else ""
    prompt = str(job.get("prompt", "")).replace("\n", " ")
    if len(prompt) > 80:
        prompt = prompt[:77] + "..."
    return (
        f"- `{job.get('id')}` {status} {mode} `{job.get('cron')}` "
        f"next={_fmt_time(job.get('next_fire_at'))} fires={job.get('fire_count', 0)}"
        f"{error}\n  {prompt}"
    )


def _parse_add_args(args: str) -> tuple[str, str, bool] | str:
    rest = args.strip()
    recurring = True
    if rest.startswith("--once "):
        recurring = False
        rest = rest[len("--once ") :].lstrip()
    parts = rest.split(maxsplit=5)
    if len(parts) < 6:
        return (
            "Usage: `/schedule add [--once] <minute> <hour> <dom> <month> "
            "<dow> <prompt>`"
        )
    cron = " ".join(parts[:5])
    prompt = parts[5].strip()
    if not prompt:
        return "scheduled prompt cannot be empty"
    return (cron, prompt, recurring)


@dataclass(slots=True)
class ScheduleCommand:
    name: str = "schedule"
    namespace: str | None = None
    summary: str = "Manage durable gateway scheduled prompts"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        subcmd, _, rest = inv.args.strip().partition(" ")
        subcmd = subcmd.lower()
        if subcmd in ("", "help"):
            return CommandResult(outbound=[ctx.notice(_help_text())])
        if subcmd == "list":
            jobs = ctx.list_schedules()
            if not jobs:
                return CommandResult(outbound=[ctx.notice("No scheduled prompts.")])
            body = "**Scheduled Prompts**\n" + "\n".join(_job_line(job) for job in jobs)
            return CommandResult(outbound=[ctx.notice(body)])
        if subcmd == "add":
            parsed = _parse_add_args(rest)
            if isinstance(parsed, str):
                return CommandResult(outbound=[ctx.notice(parsed, title="Schedule error")])
            cron, prompt, recurring = parsed
            job = ctx.create_schedule(cron, prompt, recurring=recurring)
            if "error" in job:
                return CommandResult(
                    outbound=[ctx.notice(str(job["error"]), title="Schedule error")]
                )
            body = (
                "**Scheduled Prompt Created**\n"
                f"- id: `{job['id']}`\n"
                f"- cron: `{job['cron']}`\n"
                f"- next: {_fmt_time(job.get('next_fire_at'))}\n"
                f"- mode: {'recurring' if job.get('recurring', True) else 'once'}"
            )
            return CommandResult(outbound=[ctx.notice(body)])
        if subcmd in ("delete", "del", "rm"):
            job_id = rest.strip()
            if not job_id:
                return CommandResult(
                    outbound=[ctx.notice("Usage: `/schedule delete <job_id>`")]
                )
            if not ctx.delete_schedule(job_id):
                return CommandResult(outbound=[ctx.notice(f"No such job: `{job_id}`")])
            return CommandResult(outbound=[ctx.notice(f"Deleted schedule `{job_id}`.")])
        if subcmd == "run":
            job_id = rest.strip()
            if not job_id:
                return CommandResult(
                    outbound=[ctx.notice("Usage: `/schedule run <job_id>`")]
                )
            ok, message = await ctx.run_schedule(job_id)
            if not ok:
                return CommandResult(
                    outbound=[
                        ctx.notice(
                            f"Could not fire schedule `{job_id}`: {message}",
                            title="Schedule error",
                        )
                    ]
                )
            return CommandResult(outbound=[ctx.notice(f"Fired schedule `{job_id}`.")])
        return CommandResult(outbound=[ctx.notice(_help_text(), title="Schedule")])


def _help_text() -> str:
    return (
        "**Schedule Commands**\n"
        "- `/schedule list`\n"
        "- `/schedule add */15 * * * * check the deployment`\n"
        "- `/schedule add --once 30 14 15 3 * remind me to push the branch`\n"
        "- `/schedule run <job_id>`\n"
        "- `/schedule delete <job_id>`"
    )


HANDLER = ScheduleCommand()
