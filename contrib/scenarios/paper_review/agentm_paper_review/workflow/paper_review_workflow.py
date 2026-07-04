from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.core.lib import expand_path_from_cwd
from agentm.extensions.builtin.workflow import WorkflowContext

from .types import PaperReviewArgs

TOOL_ALLOWLIST = ["read", "write", "edit", "bash", "load_skill"]
REVIEW_BUDGET = [
    (
        "agentm.extensions.builtin.loop_budget",
        {
            "max_turns": 30,
            "max_tool_calls": 200,
        },
    )
]

ROLE_SEQUENCE: tuple[tuple[str, str, str], ...] = (
    (
        "paper-reader",
        "Pass 1 Linear Reading",
        "Run the linear first-read simulation and write Markdown findings.",
    ),
    (
        "paper-prose",
        "Pass 1 Prose-Flow Review",
        "Review prose patterns that interrupt first-read flow and write Markdown findings.",
    ),
    (
        "paper-consistency",
        "Pass 2 Consistency Review",
        "Review whole-paper consistency and write Markdown findings.",
    ),
    (
        "paper-evidence",
        "Pass 2 Evidence Review",
        "Review claim-evidence support and write Markdown findings.",
    ),
    (
        "paper-structure",
        "Pass 3 Global Review",
        "Review the global argument structure and write Markdown findings.",
    ),
    (
        "paper-review",
        "Final Report",
        "Load paper-review and write the complete final Markdown report from the prior artifacts.",
    ),
)


def _find_project_root() -> Path:
    d = Path(__file__).parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    return Path(__file__).parents[1]


_PROJECT_ROOT = _find_project_root()
REVIEWER = str(_PROJECT_ROOT / "agents" / "reviewer")


@dataclass(frozen=True, slots=True)
class RoleArtifact:
    role: str
    pass_name: str
    artifact_path: str


def _resolve_input_path(raw: str, cwd: str) -> Path:
    return expand_path_from_cwd(raw, cwd).resolve(strict=False)


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


def _resolve_output_path(output_path: str | None, cwd: str, input_path: Path) -> Path:
    if output_path is None or not output_path.strip():
        base = input_path if input_path.is_dir() else input_path.parent
        return base / "paper-review-report.md"
    return expand_path_from_cwd(output_path, cwd)


def _resolve_artifact_dir(output_path: Path) -> Path:
    name = output_path.name
    if output_path.suffix:
        name = output_path.name[: -len(output_path.suffix)]
    return output_path.with_name(f"{name}.artifacts")


def _pass_artifact_path(
    artifact_dir: Path, output_path: Path, index: int, role: str
) -> Path:
    if role == "paper-review":
        return output_path
    return artifact_dir / f"{index:02d}-{role}.md"


def _require_artifact(path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"paper_review expected {role} to write artifact: {path}"
        )
    if path.stat().st_size <= 0:
        raise RuntimeError(f"paper_review artifact is empty for {role}: {path}")


async def _run_role(
    ctx: WorkflowContext,
    *,
    args: PaperReviewArgs,
    role: str,
    pass_name: str,
    task: str,
    input_path: str,
    output_artifact_path: str,
    prior_artifact_paths: list[str],
) -> RoleArtifact:
    ctx.log(f"Running {pass_name} with {role}")
    await ctx.agent(
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
                "output_artifact_path": output_artifact_path,
                "prior_artifact_paths": prior_artifact_paths,
            }
        },
        retry=args.agent_retries,
        timeout=args.agent_timeout_seconds,
        trace_label=role,
    )
    artifact_path = Path(output_artifact_path)
    _require_artifact(artifact_path, role)
    return RoleArtifact(
        role=role,
        pass_name=pass_name,
        artifact_path=output_artifact_path,
    )


async def run(ctx: WorkflowContext) -> dict[str, Any]:
    args = PaperReviewArgs.model_validate(ctx.args)
    cwd = _workflow_cwd(ctx)

    ctx.phase("prepare")
    input_path = _validate_input_path(args.path, cwd)
    ctx.log(f"Using paper input path: {input_path}")
    output_path = _resolve_output_path(args.output_path, cwd, input_path)
    artifact_dir = _resolve_artifact_dir(output_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.log(f"Using pass artifact directory: {artifact_dir}")

    results: list[RoleArtifact] = []
    prior_artifact_paths: list[str] = []

    for index, (role, pass_name, task) in enumerate(ROLE_SEQUENCE, start=1):
        phase = (
            "report" if role == "paper-review" else pass_name.lower().replace(" ", "-")
        )
        ctx.phase(phase)
        pass_artifact_path = _pass_artifact_path(
            artifact_dir,
            output_path,
            index,
            role,
        )
        output = await _run_role(
            ctx,
            args=args,
            role=role,
            pass_name=pass_name,
            task=task,
            input_path=str(input_path),
            output_artifact_path=str(pass_artifact_path),
            prior_artifact_paths=prior_artifact_paths,
        )
        ctx.log(f"Wrote pass artifact: {pass_artifact_path}")
        results.append(output)
        prior_artifact_paths.append(str(pass_artifact_path))

    ctx.log(f"Wrote report: {output_path}")

    return {
        "success": True,
        "report_path": str(output_path),
        "artifact_dir": str(artifact_dir),
        "input_path": str(input_path),
        "passes": [
            {
                "role": result.role,
                "pass_name": result.pass_name,
                "artifact_path": result.artifact_path,
            }
            for result in results
        ],
        "child_sessions": ctx.child_sessions,
    }
