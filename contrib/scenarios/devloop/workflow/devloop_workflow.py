"""DevOps control loop workflow (module mode).

Pure orchestration: requirement → spec → design review → test writing →
development → test verification (with retry) → code review.

Types and schemas live in ``types.py``; prompt construction in
``prompts.py``. This file contains only the control flow.
"""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from agentm.extensions.builtin.workflow import AgentResult, WorkflowContext

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
    CODE_REVIEW_SCHEMA,
    REVIEW_SCHEMA,
    SPEC_SCHEMA,
    TEST_INFO_SCHEMA,
    TEST_RESULT_SCHEMA,
    CodeReview,
    DesignReview,
    ImplementationSpec,
    TestInfo,
    TestResult,
)

CODER = "devloop/agents/coder"
_M = TypeVar("_M", bound=BaseModel)


def _parse(model: type[_M], result: AgentResult) -> _M:
    if not isinstance(result, dict):
        raise TypeError(f"expected dict from agent, got {type(result).__name__}")
    return model.model_validate(result)


async def run(ctx: WorkflowContext) -> dict[str, Any]:
    """Execute the full devloop pipeline."""
    args = ctx.args
    requirement: str = args["requirement"]
    language: str = args.get("language", "python")
    test_framework: str = args.get("test_framework", "pytest")
    max_rounds: int = args.get("max_rounds", 3)
    skip_review: bool = args.get("skip_review", False)

    # ── Stage 1: Spec Writing ──────────────────────────────────

    ctx.phase("spec")
    ctx.log("Writing implementation spec from requirement")

    spec: ImplementationSpec = _parse(ImplementationSpec, await ctx.agent(
        spec_prompt(requirement, language, test_framework),
        schema=SPEC_SCHEMA,
    ))
    ctx.log(f"Spec: {spec.title} — {len(spec.acceptance_criteria)} ACs")

    # ── Stage 2: Design Review Gate ────────────────────────────

    ctx.phase("design-review")
    ctx.log("Reviewing spec for completeness")

    spec_json = spec.model_dump_json(indent=2)
    review: DesignReview = _parse(DesignReview, await ctx.agent(
        review_prompt(spec_json),
        schema=REVIEW_SCHEMA,
    ))

    if not review.approved:
        ctx.log(f"Spec rejected — revising. Feedback: {review.feedback}")
        spec = _parse(ImplementationSpec, await ctx.agent(
            revise_spec_prompt(spec_json, review.feedback, review.issues),
            schema=SPEC_SCHEMA,
        ))
        ctx.log(f"Revised: {spec.title} — {len(spec.acceptance_criteria)} ACs")

    spec_json = spec.model_dump_json(indent=2)

    # ── Stage 3: Test Writing ──────────────────────────────────

    ctx.phase("test-writing")
    ctx.log("Writing tests from spec (before implementation)")

    test_info: TestInfo = _parse(TestInfo, await ctx.agent(
        "Write test files based on the spec. One test per acceptance criterion.",
        scenario=CODER,
        atom_config={"devloop_context": {
            "task": test_writing_task(test_framework, spec_json),
            "spec": spec_json,
        }},
        schema=TEST_INFO_SCHEMA,
    ))
    test_files = test_info.test_files
    ctx.log(f"Wrote {len(test_files)} test file(s): {', '.join(test_files)}")

    # ── Stage 4+5: Development + Test Verification Loop ───────

    test_result = TestResult(
        all_passed=False, total=0, passed=0, failed=0,
    )
    round_n = 0

    for round_n in range(1, max_rounds + 1):
        ctx.phase(f"develop-{round_n}")

        dev_context: dict[str, Any] = {
            "task": "Implement the code to pass all tests.",
            "spec": spec_json,
            "test_files": test_files,
        }

        if round_n > 1 and test_result.failures:
            dev_context["task"] = dev_fix_task(
                round_n, max_rounds,
                test_result.passed, test_result.total,
                test_result.failures,
            )

        ctx.log(f"Development round {round_n}/{max_rounds}")
        await ctx.agent(
            "Implement the code." if round_n == 1 else "Fix the failing tests.",
            scenario=CODER,
            atom_config={"devloop_context": dev_context},
        )

        ctx.phase(f"test-{round_n}")
        ctx.log(f"Running tests (round {round_n})")

        test_result = _parse(TestResult, await ctx.agent(
            "Run the test suite and report results.",
            scenario=CODER,
            atom_config={"devloop_context": {
                "task": test_run_task(test_framework, test_files),
            }},
            schema=TEST_RESULT_SCHEMA,
        ))

        if test_result.all_passed:
            ctx.log(f"All {test_result.total} tests passed on round {round_n}!")
            break

        ctx.log(
            f"Round {round_n}: {test_result.passed}/{test_result.total} "
            f"passed, {len(test_result.failures)} failed"
        )
        if round_n >= max_rounds:
            ctx.log(f"Max rounds ({max_rounds}) reached. Stopping.")

    # ── Stage 6: Code Review ──────────────────────────────────

    code_review: CodeReview | None = None
    if not skip_review and test_result.all_passed:
        ctx.phase("code-review")
        ctx.log("Running code review against spec")

        code_review = _parse(CodeReview, await ctx.agent(
            code_review_prompt(spec_json),
            scenario=CODER,
            atom_config={"devloop_context": {
                "task": CODE_REVIEW_TASK,
                "spec": spec_json,
            }},
            schema=CODE_REVIEW_SCHEMA,
        ))

        if code_review.approved:
            ctx.log("Code review: APPROVED")
        else:
            ctx.log(f"Code review: {len(code_review.findings)} finding(s)")

    return {
        "spec": spec.model_dump(),
        "design_review": review.model_dump(),
        "test_files": test_files,
        "test_result": test_result.model_dump(),
        "code_review": code_review.model_dump() if code_review else None,
        "rounds": round_n,
        "success": test_result.all_passed,
    }
