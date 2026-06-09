"""DevOps control loop workflow (module mode).

Implements: requirement → spec → design review → test writing →
development → test verification (with retry) → code review.

Args (passed via WorkflowContext.args):
  requirement (str): Natural-language feature requirement.
  language (str): Target language (default: "python").
  test_framework (str): Test framework (default: "pytest").
  max_rounds (int): Max dev+test iterations (default: 3).
  skip_review (bool): Skip final code review (default: false).
"""
from __future__ import annotations

import json
from typing import Required, TypedDict, cast

from agentm.extensions.builtin.workflow import (
    AgentResult,
    AtomConfigMap,
    JsonSchema,
    WorkflowContext,
)

# ── schemas ─────────────────────────────────────────────────────

SPEC_SCHEMA: JsonSchema = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short feature title."},
        "description": {"type": "string", "description": "What the feature does."},
        "interfaces": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "signature": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "signature"],
            },
            "description": "Function/class/method signatures.",
        },
        "acceptance_criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "e.g. AC-1"},
                    "description": {"type": "string"},
                    "test_method": {
                        "type": "string",
                        "description": "How to verify (test name or command).",
                    },
                },
                "required": ["id", "description"],
            },
        },
        "file_structure": {
            "type": "object",
            "properties": {
                "source_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths for implementation files.",
                },
                "test_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths for test files.",
                },
            },
        },
    },
    "required": ["title", "interfaces", "acceptance_criteria"],
}

REVIEW_SCHEMA: JsonSchema = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "feedback": {"type": "string", "description": "Overall feedback."},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ac_id": {"type": "string"},
                    "issue": {"type": "string"},
                },
            },
            "description": "Per-AC issues found.",
        },
    },
    "required": ["approved"],
}

TEST_RESULT_SCHEMA: JsonSchema = {
    "type": "object",
    "properties": {
        "all_passed": {"type": "boolean"},
        "total": {"type": "integer"},
        "passed": {"type": "integer"},
        "failed": {"type": "integer"},
        "failures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "error_message": {"type": "string"},
                },
                "required": ["test_name", "error_message"],
            },
        },
        "stdout": {"type": "string", "description": "Raw pytest output."},
    },
    "required": ["all_passed", "total", "passed", "failed"],
}

CODE_REVIEW_SCHEMA: JsonSchema = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ac_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pass", "fail", "cannot_judge"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["ac_id", "status"],
            },
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
    },
    "required": ["approved"],
}

TEST_INFO_SCHEMA: JsonSchema = {
    "type": "object",
    "properties": {
        "test_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Paths of test files written.",
        },
        "test_count": {"type": "integer"},
    },
    "required": ["test_files"],
}

CODER = "devloop/coder"


class InterfaceSpec(TypedDict, total=False):
    name: Required[str]
    signature: Required[str]
    description: str


class AcceptanceCriterion(TypedDict, total=False):
    id: Required[str]
    description: Required[str]
    test_method: str


class FileStructure(TypedDict, total=False):
    source_files: list[str]
    test_files: list[str]


class ImplementationSpec(TypedDict, total=False):
    title: Required[str]
    description: str
    interfaces: Required[list[InterfaceSpec]]
    acceptance_criteria: Required[list[AcceptanceCriterion]]
    file_structure: FileStructure


class ReviewIssue(TypedDict, total=False):
    ac_id: str
    issue: str


class DesignReview(TypedDict, total=False):
    approved: Required[bool]
    feedback: str
    issues: list[ReviewIssue]


class TestFailure(TypedDict):
    test_name: str
    error_message: str


class TestInfo(TypedDict, total=False):
    test_files: Required[list[str]]
    test_count: int


class TestResult(TypedDict, total=False):
    all_passed: Required[bool]
    total: Required[int]
    passed: Required[int]
    failed: Required[int]
    failures: list[TestFailure]
    stdout: str


class CodeReview(TypedDict, total=False):
    approved: Required[bool]
    verdicts: list[dict[str, str]]
    findings: list[dict[str, str]]


class DevContext(TypedDict, total=False):
    task: Required[str]
    spec: str
    test_files: list[str]
    previous_failures: list[TestFailure]


class DevloopResult(TypedDict):
    spec: ImplementationSpec
    design_review: DesignReview
    test_files: list[str]
    test_result: TestResult
    code_review: CodeReview | None
    rounds: int
    success: bool


# ── helpers ─────────────────────────────────────────────────────

def _require_dict(result: AgentResult) -> AgentResult:
    """Require structured dict output before casting to a schema-specific type."""
    if isinstance(result, dict):
        return result
    raise TypeError(f"expected dict from agent, got {type(result).__name__}")


def _as_spec(result: AgentResult) -> ImplementationSpec:
    return cast(ImplementationSpec, _require_dict(result))


def _as_design_review(result: AgentResult) -> DesignReview:
    return cast(DesignReview, _require_dict(result))


def _as_test_info(result: AgentResult) -> TestInfo:
    return cast(TestInfo, _require_dict(result))


def _as_test_result(result: AgentResult) -> TestResult:
    return cast(TestResult, _require_dict(result))


def _as_code_review(result: AgentResult) -> CodeReview:
    return cast(CodeReview, _require_dict(result))


# ── entry point ─────────────────────────────────────────────────

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

    spec = _as_spec(await ctx.agent(
        f"Write a detailed implementation spec for this requirement.\n\n"
        f"## Requirement\n{requirement}\n\n"
        f"## Constraints\n"
        f"- Language: {language}\n"
        f"- Test framework: {test_framework}\n\n"
        f"Include interface definitions (signatures), testable acceptance "
        f"criteria (each AC maps to a specific test), and a file_structure "
        f"showing where source and test files should go.",
        schema=SPEC_SCHEMA,
    ))
    ctx.log(f"Spec: {spec['title']} — {len(spec.get('acceptance_criteria', []))} ACs")

    # ── Stage 2: Design Review Gate ────────────────────────────

    ctx.phase("design-review")
    ctx.log("Reviewing spec for completeness")

    spec_text = json.dumps(spec, indent=2)
    review = _as_design_review(await ctx.agent(
        f"Review this implementation spec. Check that every acceptance "
        f"criterion is testable (maps to a concrete, automatable test), "
        f"interfaces are specific enough to implement, and there are no "
        f"ambiguous requirements.\n\n{spec_text}",
        schema=REVIEW_SCHEMA,
    ))

    if not review.get("approved"):
        feedback = review.get("feedback", "")
        issues_text = json.dumps(review.get("issues", []), indent=2)
        ctx.log(f"Spec rejected — revising. Feedback: {feedback}")

        spec = _as_spec(await ctx.agent(
            f"Revise the spec based on reviewer feedback.\n\n"
            f"## Original spec\n{spec_text}\n\n"
            f"## Feedback\n{feedback}\n\n"
            f"## Issues\n{issues_text}\n\n"
            f"Address every issue. Keep all existing ACs that were fine; "
            f"fix or add the ones flagged.",
            schema=SPEC_SCHEMA,
        ))
        ctx.log(f"Revised: {spec['title']} — {len(spec.get('acceptance_criteria', []))} ACs")

    spec_text = json.dumps(spec, indent=2)

    # ── Stage 3: Test Writing ──────────────────────────────────

    ctx.phase("test-writing")
    ctx.log("Writing tests from spec (before implementation)")

    test_info = _as_test_info(await ctx.agent(
        "Write test files based on the spec. One test per acceptance criterion.",
        scenario=CODER,
        atom_config={"devloop_context": {
            "task": (
                f"Write {test_framework} test files based on the spec below. "
                f"Create one test function per acceptance criterion. "
                f"Use the file paths from the spec's file_structure.test_files. "
                f"The tests should be runnable (import the modules that will "
                f"be implemented) but are expected to FAIL right now since "
                f"no implementation exists yet."
            ),
            "spec": spec_text,
        }},
        schema=TEST_INFO_SCHEMA,
    ))
    test_files: list[str] = test_info.get("test_files", [])
    ctx.log(f"Wrote {len(test_files)} test file(s): {', '.join(test_files)}")

    # ── Stage 4+5: Development + Test Verification Loop ───────

    test_result: TestResult = {
        "all_passed": False, "total": 0, "passed": 0, "failed": 0, "failures": [],
    }
    round_n = 0

    for round_n in range(1, max_rounds + 1):
        # ── Develop ──
        ctx.phase(f"develop-{round_n}")

        dev_context: DevContext = {
            "task": "Implement the code to pass all tests.",
            "spec": spec_text,
            "test_files": test_files,
        }

        if round_n > 1 and test_result.get("failures"):
            prev_failures: list[TestFailure] = test_result["failures"]
            failure_summary = "\n".join(
                f"- {f['test_name']}: {f['error_message']}"
                for f in prev_failures
            )
            dev_context["task"] = (
                f"Fix the failing tests (round {round_n}/{max_rounds}).\n\n"
                f"Previous round: {test_result['passed']}/{test_result['total']} passed.\n\n"
                f"Failures:\n{failure_summary}\n\n"
                f"Do NOT modify the test files — fix the implementation only."
            )
            dev_context["previous_failures"] = prev_failures

        ctx.log(f"Development round {round_n}/{max_rounds}")
        await ctx.agent(
            "Implement the code." if round_n == 1 else "Fix the failing tests.",
            scenario=CODER,
            atom_config=cast(AtomConfigMap, {"devloop_context": dev_context}),
        )

        # ── Test ──
        ctx.phase(f"test-{round_n}")
        ctx.log(f"Running tests (round {round_n})")

        test_result = _as_test_result(await ctx.agent(
            "Run the test suite and report results.",
            scenario=CODER,
            atom_config={"devloop_context": {
                "task": (
                    f"Run the test suite with {test_framework}. "
                    f"Test files: {', '.join(test_files)}. "
                    f"Run them and report the results. Include the raw "
                    f"output in stdout."
                ),
            }},
            schema=TEST_RESULT_SCHEMA,
        ))

        passed: int = test_result.get("passed", 0)
        total: int = test_result.get("total", 0)
        failures: list[TestFailure] = test_result.get("failures", [])

        if test_result.get("all_passed"):
            ctx.log(f"All {total} tests passed on round {round_n}!")
            break

        ctx.log(f"Round {round_n}: {passed}/{total} passed, {len(failures)} failed")

        if round_n >= max_rounds:
            ctx.log(f"Max rounds ({max_rounds}) reached. Stopping.")

    # ── Stage 6: Code Review ──────────────────────────────────

    code_review: CodeReview | None = None
    if not skip_review and test_result.get("all_passed"):
        ctx.phase("code-review")
        ctx.log("Running code review against spec")

        code_review = _as_code_review(await ctx.agent(
            f"Review the implementation against the spec. Read the source "
            f"files and verify each acceptance criterion is properly "
            f"implemented. Check for correctness, code quality, and "
            f"potential issues.\n\n## Spec\n{spec_text}",
            scenario=CODER,
            atom_config={"devloop_context": {
                "task": (
                    "Review the implementation. Read the source files "
                    "and for each acceptance criterion, verify it's correctly "
                    "implemented. Check code quality and potential issues."
                ),
                "spec": spec_text,
            }},
            schema=CODE_REVIEW_SCHEMA,
        ))

        if code_review.get("approved"):
            ctx.log("Code review: APPROVED")
        else:
            findings = code_review.get("findings", [])
            ctx.log(f"Code review: {len(findings)} finding(s)")

    # ── Result ─────────────────────────────────────────────────

    return {
        "spec": spec,
        "design_review": review,
        "test_files": test_files,
        "test_result": test_result,
        "code_review": code_review,
        "rounds": round_n,
        "success": bool(test_result.get("all_passed")),
    }
