"""DevOps control loop workflow (module mode).

Pure orchestration: requirement → spec → design review → test writing →
development → test verification (with retry) → code review.

Types and schemas live in ``types.py``; prompt construction in
``prompts.py``. This file contains only the control flow.
"""
from __future__ import annotations

import json
from typing import cast

from agentm.extensions.builtin.workflow import (
    AgentResult,
    AtomConfigMap,
    WorkflowContext,
)

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
    DevContext,
    DevloopResult,
    ImplementationSpec,
    TestInfo,
    TestResult,
)

CODER = "devloop/agents/coder"


def _require_dict(result: AgentResult) -> dict:
    if isinstance(result, dict):
        return result
    raise TypeError(f"expected dict from agent, got {type(result).__name__}")


async def run(ctx: WorkflowContext) -> DevloopResult:
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

    spec = cast(ImplementationSpec, _require_dict(await ctx.agent(
        spec_prompt(requirement, language, test_framework),
        schema=SPEC_SCHEMA,
    )))
    ctx.log(f"Spec: {spec['title']} — {len(spec.get('acceptance_criteria', []))} ACs")

    # ── Stage 2: Design Review Gate ────────────────────────────

    ctx.phase("design-review")
    ctx.log("Reviewing spec for completeness")

    spec_json = json.dumps(spec, indent=2)
    review = cast(DesignReview, _require_dict(await ctx.agent(
        review_prompt(spec_json),
        schema=REVIEW_SCHEMA,
    )))

    if not review.get("approved"):
        feedback = review.get("feedback", "")
        ctx.log(f"Spec rejected — revising. Feedback: {feedback}")

        spec = cast(ImplementationSpec, _require_dict(await ctx.agent(
            revise_spec_prompt(spec_json, feedback, review.get("issues", [])),
            schema=SPEC_SCHEMA,
        )))
        ctx.log(f"Revised: {spec['title']} — {len(spec.get('acceptance_criteria', []))} ACs")

    spec_json = json.dumps(spec, indent=2)

    # ── Stage 3: Test Writing ──────────────────────────────────

    ctx.phase("test-writing")
    ctx.log("Writing tests from spec (before implementation)")

    test_info = cast(TestInfo, _require_dict(await ctx.agent(
        "Write test files based on the spec. One test per acceptance criterion.",
        scenario=CODER,
        atom_config={"devloop_context": {
            "task": test_writing_task(test_framework, spec_json),
            "spec": spec_json,
        }},
        schema=TEST_INFO_SCHEMA,
    )))
    test_files: list[str] = test_info.get("test_files", [])
    ctx.log(f"Wrote {len(test_files)} test file(s): {', '.join(test_files)}")

    # ── Stage 4+5: Development + Test Verification Loop ───────

    test_result: TestResult = {
        "all_passed": False, "total": 0, "passed": 0, "failed": 0, "failures": [],
    }
    round_n = 0

    for round_n in range(1, max_rounds + 1):
        ctx.phase(f"develop-{round_n}")

        dev_context: DevContext = {
            "task": "Implement the code to pass all tests.",
            "spec": spec_json,
            "test_files": test_files,
        }

        if round_n > 1 and test_result.get("failures"):
            prev_failures = test_result["failures"]
            dev_context["task"] = dev_fix_task(
                round_n, max_rounds,
                test_result["passed"], test_result["total"],
                prev_failures,
            )
            dev_context["previous_failures"] = prev_failures

        ctx.log(f"Development round {round_n}/{max_rounds}")
        await ctx.agent(
            "Implement the code." if round_n == 1 else "Fix the failing tests.",
            scenario=CODER,
            atom_config=cast(AtomConfigMap, {"devloop_context": dev_context}),
        )

        ctx.phase(f"test-{round_n}")
        ctx.log(f"Running tests (round {round_n})")

        test_result = cast(TestResult, _require_dict(await ctx.agent(
            "Run the test suite and report results.",
            scenario=CODER,
            atom_config={"devloop_context": {
                "task": test_run_task(test_framework, test_files),
            }},
            schema=TEST_RESULT_SCHEMA,
        )))

        if test_result.get("all_passed"):
            ctx.log(f"All {test_result['total']} tests passed on round {round_n}!")
            break

        ctx.log(
            f"Round {round_n}: {test_result.get('passed', 0)}/{test_result.get('total', 0)} "
            f"passed, {len(test_result.get('failures', []))} failed"
        )
        if round_n >= max_rounds:
            ctx.log(f"Max rounds ({max_rounds}) reached. Stopping.")

    # ── Stage 6: Code Review ──────────────────────────────────

    code_review: CodeReview | None = None
    if not skip_review and test_result.get("all_passed"):
        ctx.phase("code-review")
        ctx.log("Running code review against spec")

        code_review = cast(CodeReview, _require_dict(await ctx.agent(
            code_review_prompt(spec_json),
            scenario=CODER,
            atom_config={"devloop_context": {
                "task": CODE_REVIEW_TASK,
                "spec": spec_json,
            }},
            schema=CODE_REVIEW_SCHEMA,
        )))

        if code_review.get("approved"):
            ctx.log("Code review: APPROVED")
        else:
            ctx.log(f"Code review: {len(code_review.get('findings', []))} finding(s)")

    return {
        "spec": spec,
        "design_review": review,
        "test_files": test_files,
        "test_result": test_result,
        "code_review": code_review,
        "rounds": round_n,
        "success": bool(test_result.get("all_passed")),
    }
