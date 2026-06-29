from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.extensions.builtin.workflow import AgentResult, WorkflowContext

from .types import PaperReviewArgs

REVIEWER = "paper_review/agents/reviewer"

TOOL_ALLOWLIST = ["read", "write", "edit", "bash", "load_skill"]
REVIEW_BUDGET = [
    (
        "agentm.extensions.builtin.loop_budget",
        {
            "max_turns": 24,
            "max_tool_calls": 40,
        },
    )
]

ROLE_SEQUENCE: tuple[tuple[str, str, str], ...] = (
    (
        "paper-reader",
        "Pass 1 Linear Reading",
        "Run the linear first-read simulation and return Markdown findings.",
    ),
    (
        "paper-prose",
        "Pass 1 Prose-Flow Review",
        "Review prose patterns that interrupt first-read flow and return Markdown findings.",
    ),
    (
        "paper-consistency",
        "Pass 2 Consistency Review",
        "Review whole-paper consistency and return Markdown findings.",
    ),
    (
        "paper-evidence",
        "Pass 2 Evidence Review",
        "Review claim-evidence support and return Markdown findings.",
    ),
    (
        "paper-structure",
        "Pass 3 Global Review",
        "Review the global argument structure and return Markdown findings.",
    ),
    (
        "paper-review",
        "Final Report",
        "Load paper-review and write the complete final Markdown report from the prior artifacts.",
    ),
)


@dataclass(frozen=True, slots=True)
class RoleMarkdown:
    role: str
    pass_name: str
    markdown: str


def _resolve_input_path(raw: str, cwd: str) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    return candidate.resolve(strict=False)


def _workflow_cwd(ctx: WorkflowContext) -> str:
    explicit = ctx.args.get("cwd")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    run = getattr(ctx, "_run", None)
    api = getattr(run, "api", None)
    api_cwd = getattr(api, "cwd", None)
    if isinstance(api_cwd, str) and api_cwd:
        return api_cwd
    return str(Path.cwd())


def _validate_input_path(raw: str, cwd: str) -> Path:
    input_path = _resolve_input_path(raw, cwd)
    if not input_path.exists():
        raise FileNotFoundError(f"paper_review input path does not exist: {raw}")
    if not (input_path.is_dir() or input_path.is_file()):
        raise FileNotFoundError(f"paper_review input is not a file or directory: {raw}")
    return input_path


def _resolve_output_path(
    output_path: str | None, cwd: str, input_path: Path
) -> Path:
    if output_path is None or not output_path.strip():
        base = input_path if input_path.is_dir() else input_path.parent
        return base / "paper-review-report.md"
    out = Path(output_path).expanduser()
    if not out.is_absolute():
        out = Path(cwd) / out
    return out


def _coerce_markdown(result: AgentResult) -> str:
    if isinstance(result, str):
        return result
    return f"```text\n{result}\n```"


def _append_artifact(prior_markdown: str, output: RoleMarkdown) -> str:
    block = f"## {output.role}\n\n{output.markdown.strip()}"
    if not prior_markdown.strip():
        return block
    return f"{prior_markdown.rstrip()}\n\n{block}"


async def _run_role(
    ctx: WorkflowContext,
    *,
    args: PaperReviewArgs,
    role: str,
    pass_name: str,
    task: str,
    input_path: str,
    prior_markdown: str,
) -> RoleMarkdown:
    ctx.log(f"Running {pass_name} with {role}")
    result = await ctx.agent(
        f"Run {pass_name} as {role}. Load skill {role} first.",
        scenario=REVIEWER,
        tool_allowlist=TOOL_ALLOWLIST,
        extra_extensions=REVIEW_BUDGET,
        atom_config={
            "paper_review_context": {
                "role": role,
                "skill_name": role,
                "pass_name": pass_name,
                "task": task,
                "input_path": input_path,
                "prior_markdown": prior_markdown,
            }
        },
        retry=args.agent_retries,
        timeout=args.agent_timeout_seconds,
        trace_label=role,
    )
    return RoleMarkdown(
        role=role, pass_name=pass_name, markdown=_coerce_markdown(result)
    )


async def run(ctx: WorkflowContext) -> dict[str, Any]:
    args = PaperReviewArgs.model_validate(ctx.args)
    cwd = _workflow_cwd(ctx)

    ctx.phase("prepare")
    input_path = _validate_input_path(args.path, cwd)
    ctx.log(f"Using paper input path: {input_path}")

    results: list[RoleMarkdown] = []
    prior_markdown = ""
    final_report = ""

    for role, pass_name, task in ROLE_SEQUENCE:
        phase = (
            "report" if role == "paper-review" else pass_name.lower().replace(" ", "-")
        )
        ctx.phase(phase)
        output = await _run_role(
            ctx,
            args=args,
            role=role,
            pass_name=pass_name,
            task=task,
            input_path=str(input_path),
            prior_markdown=prior_markdown,
        )
        results.append(output)
        if role == "paper-review":
            final_report = output.markdown
        prior_markdown = _append_artifact(prior_markdown, output)

    report = final_report or results[-1].markdown
    output_path = _resolve_output_path(args.output_path, cwd, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    ctx.log(f"Wrote report: {output_path}")

    return {
        "success": True,
        "report_path": str(output_path),
        "input_path": str(input_path),
        "passes": [
            {
                "role": result.role,
                "pass_name": result.pass_name,
                "markdown": result.markdown,
            }
            for result in results
        ],
        "child_sessions": ctx.child_sessions,
    }
