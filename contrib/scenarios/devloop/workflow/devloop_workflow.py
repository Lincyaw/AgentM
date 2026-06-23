"""DevOps control loop workflow (module mode).

Pure orchestration: requirement → spec → design review → test writing →
development → test verification (with retry) → code review.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentm.extensions.builtin.workflow import WorkflowContext

from .prompts import (
    CODE_REVIEW_TASK,
    code_review_prompt,
    dev_fix_task,
    revise_spec_prompt,
    spec_prompt,
    review_prompt,
    test_run_task,
    test_writing_task,
)
from .types import (
    CodeReview,
    DesignReview,
    DevloopArgs,
    ImplementationSpec,
    TestFailure,
    TestInfo,
    TestResult,
)

CODER = "devloop/agents/coder"
SUPERVISOR = "supervisor"


def _find_project_root() -> Path:
    d = Path(__file__).parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    return Path(__file__).parents[1]


_PROJECT_ROOT = _find_project_root()
_SKILL_CONFIG: dict[str, Any] = {
    "skill_loader": {"skill_paths": [str(_PROJECT_ROOT / "skills")]},
}

STRUCTURED_ONLY_TOOLS = ["submit_result"]
CODING_TOOLS = ["read", "write", "edit", "glob", "grep", "bash"]
TEST_RUN_TOOLS = ["bash", "submit_result"]
REVIEW_TOOLS = ["read", "glob", "grep", "bash", "submit_result"]
STRUCTURED_BUDGET = [
    ("agentm.extensions.builtin.loop_budget", {
        "max_turns": 8,
        "max_tool_calls": 12,
    }),
]
TEST_WRITING_BUDGET = [
    ("agentm.extensions.builtin.loop_budget", {
        "max_turns": 45,
        "max_tool_calls": 90,
    }),
]
DEVELOP_BUDGET = [
    ("agentm.extensions.builtin.loop_budget", {
        "max_turns": 80,
        "max_tool_calls": 160,
    }),
]
TEST_RUN_BUDGET = [
    ("agentm.extensions.builtin.loop_budget", {
        "max_turns": 12,
        "max_tool_calls": 24,
    }),
]
REVIEW_BUDGET = [
    ("agentm.extensions.builtin.loop_budget", {
        "max_turns": 30,
        "max_tool_calls": 80,
    }),
]
SUPERVISE_BUDGET = [
    ("agentm.extensions.builtin.loop_budget", {
        "max_turns": 30,
        "max_tool_calls": 60,
    }),
]

def _snapshot_test_files(test_files: list[str]) -> dict[str, bytes | None]:
    snapshot: dict[str, bytes | None] = {}
    for path in test_files:
        fp = Path(path)
        snapshot[path] = fp.read_bytes() if fp.exists() else None
    return snapshot


def _restore_changed_test_files(snapshot: dict[str, bytes | None]) -> list[str]:
    changed: list[str] = []
    for path, original in snapshot.items():
        fp = Path(path)
        current = fp.read_bytes() if fp.exists() else None
        if current == original:
            continue
        changed.append(path)
        if original is not None:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(original)
    return changed


async def run(ctx: WorkflowContext) -> dict[str, Any]:
    """Execute the full devloop pipeline."""
    args = DevloopArgs.model_validate(ctx.args)

    # ── Stage 1: Spec ─────────────────────────────────────────

    ctx.phase("spec")
    ctx.log("Writing implementation spec from requirement")

    spec: ImplementationSpec = await ctx.agent(
        spec_prompt(args.requirement, args.language, args.test_framework),
        schema=ImplementationSpec,
        scenario=CODER,
        tool_allowlist=STRUCTURED_ONLY_TOOLS,
        extra_extensions=STRUCTURED_BUDGET,
        retry=3, timeout=args.agent_timeout_seconds,
    )
    ctx.log(f"Spec: {spec.title} — {len(spec.acceptance_criteria)} ACs")

    # ── Stage 2: Design Review Gate ───────────────────────────

    ctx.phase("design-review")
    ctx.log("Reviewing spec for completeness")

    spec_json = spec.model_dump_json(indent=2)
    review: DesignReview = await ctx.agent(
        review_prompt(spec_json),
        schema=DesignReview,
        scenario=CODER,
        tool_allowlist=STRUCTURED_ONLY_TOOLS,
        extra_extensions=STRUCTURED_BUDGET,
        retry=3, timeout=args.agent_timeout_seconds,
    )

    if not review.approved:
        ctx.log(f"Spec rejected — revising. Feedback: {review.feedback}")
        spec = await ctx.agent(
            revise_spec_prompt(spec_json, review.feedback, review.issues),
            schema=ImplementationSpec,
            scenario=CODER,
            tool_allowlist=STRUCTURED_ONLY_TOOLS,
            extra_extensions=STRUCTURED_BUDGET,
            retry=3, timeout=args.agent_timeout_seconds,
        )
        ctx.log(f"Revised: {spec.title} — {len(spec.acceptance_criteria)} ACs")

    spec_json = spec.model_dump_json(indent=2)

    # ── Stage 3: Test Writing ─────────────────────────────────

    ctx.phase("test-writing")
    ctx.log("Writing tests from spec (before implementation)")

    test_info: TestInfo = await ctx.agent(
        "Write test files based on the spec. One test per acceptance criterion.",
        schema=TestInfo,
        scenario=CODER,
        tool_allowlist=[*CODING_TOOLS, "submit_result"],
        extra_extensions=TEST_WRITING_BUDGET,
        atom_config={**_SKILL_CONFIG, "devloop_context": {
            "task": test_writing_task(args.test_framework, spec_json),
            "spec": spec_json,
        }},
        retry=3, timeout=args.agent_timeout_seconds,
    )
    test_files = test_info.test_files
    ctx.log(f"Wrote {len(test_files)} test file(s): {', '.join(test_files)}")

    test_snapshot = _snapshot_test_files(test_files)

    # ── Stage 4+5: Development + Test Verification Loop ───────

    test_result = TestResult(all_passed=False, total=0, passed=0, failed=0)
    round_n = 0

    for round_n in range(1, args.max_rounds + 1):
        if test_result.all_passed:
            break

        ctx.phase(f"develop-{round_n}")

        dev_context: dict[str, Any] = {
            "task": "Implement the code to pass all tests.",
            "spec": spec_json,
            "test_files": test_files,
        }
        if test_result.failures:
            dev_context["task"] = dev_fix_task(
                round_n, args.max_rounds,
                test_result.passed, test_result.total,
                test_result.failures,
            )

        ctx.log(f"Development round {round_n}/{args.max_rounds}")
        await ctx.agent(
            "Implement the code." if round_n == 1 else "Fix the failing tests.",
            scenario=CODER,
            tool_allowlist=CODING_TOOLS,
            extra_extensions=DEVELOP_BUDGET,
            atom_config={**_SKILL_CONFIG, "devloop_context": dev_context},
            timeout=args.agent_timeout_seconds,
        )

        mutated = _restore_changed_test_files(test_snapshot)
        if mutated:
            ctx.log(f"Restored modified test files: {', '.join(mutated[:5])}")
            test_result = TestResult(
                all_passed=False,
                total=len(mutated), passed=0, failed=len(mutated),
                failures=[
                    TestFailure(
                        test_name=f"test-file-mutation:{p}",
                        error_message="Test file was modified; fix source files instead.",
                    )
                    for p in mutated
                ],
            )
            continue

        ctx.phase(f"test-{round_n}")
        ctx.log(f"Running tests (round {round_n})")

        test_result = await ctx.agent(
            "Run the test suite and report results.",
            schema=TestResult,
            scenario=CODER,
            tool_allowlist=TEST_RUN_TOOLS,
            extra_extensions=TEST_RUN_BUDGET,
            atom_config={**_SKILL_CONFIG, "devloop_context": {
                "task": test_run_task(args.test_framework, test_files),
            }},
            retry=3, timeout=args.agent_timeout_seconds,
        )

        if test_result.all_passed:
            ctx.log(f"All {test_result.total} tests passed on round {round_n}!")
        else:
            ctx.log(
                f"Round {round_n}: {test_result.passed}/{test_result.total} "
                f"passed, {len(test_result.failures)} failed"
            )
            if round_n >= args.max_rounds:
                ctx.log(f"Max rounds ({args.max_rounds}) reached.")

    # ── Stage 6: Code Review ─────────────────────────────────

    code_review: CodeReview | None = None
    if not args.skip_review and test_result.all_passed:
        ctx.phase("code-review")
        ctx.log("Running code review against spec")

        code_review = await ctx.agent(
            code_review_prompt(spec_json),
            schema=CodeReview,
            scenario=CODER,
            tool_allowlist=REVIEW_TOOLS,
            extra_extensions=REVIEW_BUDGET,
            atom_config={**_SKILL_CONFIG, "devloop_context": {
                "task": CODE_REVIEW_TASK,
                "spec": spec_json,
            }},
            retry=3, timeout=args.agent_timeout_seconds,
        )

        if code_review.approved:
            ctx.log("Code review: APPROVED")
        else:
            ctx.log(f"Code review: {len(code_review.findings)} finding(s)")

    result = {
        "spec": spec.model_dump(),
        "design_review": review.model_dump(),
        "test_files": test_files,
        "test_result": test_result.model_dump(),
        "code_review": code_review.model_dump() if code_review else None,
        "rounds": round_n,
        "success": test_result.all_passed and (
            args.skip_review or code_review is None or code_review.approved
        ),
    }

    # ── Stage 7: Supervisor Reflection ────────────────────────

    ctx.phase("supervise")
    ctx.log("Supervisor reflecting on execution")

    result_summary = {
        "success": result["success"],
        "rounds": result["rounds"],
        "test_passed": test_result.passed,
        "test_total": test_result.total,
        "review_approved": code_review.approved if code_review else None,
        "review_findings": len(code_review.findings) if code_review else 0,
    }

    skills_dir = str(_PROJECT_ROOT / "skills")
    agents_dir = str(_PROJECT_ROOT / "agents")
    await ctx.agent(
        f"A devloop workflow just completed. Analyze the result and "
        f"improve the workflow's capabilities for future runs.\n\n"
        f"## Result\n```json\n{json.dumps(result_summary, indent=2)}\n```\n\n"
        f"## Project layout\n"
        f"- Skills: {skills_dir} (SKILL.md files, loaded by agents)\n"
        f"- Agents: {agents_dir} (manifest.yaml + local atoms)\n"
        f"- Prompts: {_PROJECT_ROOT}/workflow/prompts.py\n\n"
        f"## What you can do\n"
        f"- Edit existing skills to refine guidance based on what you observed\n"
        f"- Create new skills for knowledge gaps that caused failures\n"
        f"- Delete skills that are stale or unhelpful\n"
        f"- Edit prompts if the issue is in task formulation, not knowledge\n"
        f"- Create a new tool atom (single .py file with MANIFEST + install) "
        f"if agents repeatedly do manual work that should be automated\n\n"
        f"Decide what to do based on what you see. Do nothing if "
        f"the existing setup is adequate.",
        scenario=SUPERVISOR,
        tool_allowlist=["read", "write", "edit", "glob", "grep", "bash"],
        extra_extensions=SUPERVISE_BUDGET,
        atom_config=_SKILL_CONFIG,
        timeout=args.agent_timeout_seconds,
    )

    ctx.log("Supervisor reflection complete")
    return result
