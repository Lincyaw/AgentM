"""Prompt construction functions for the devloop workflow.

Each function takes structured data and returns a prompt string.
The workflow script calls these instead of building prompts inline.
"""
from __future__ import annotations

import json

from .types import ReviewIssue, TestFailure


def spec_prompt(
    requirement: str,
    language: str,
    test_framework: str,
) -> str:
    return (
        f"Write a detailed implementation spec for this requirement.\n\n"
        f"## Requirement\n{requirement}\n\n"
        f"## Constraints\n"
        f"- Language: {language}\n"
        f"- Test framework: {test_framework}\n\n"
        f"Operational rules:\n"
        f"- Do not read files or write artifacts; this is a planning step.\n"
        f"- Return only the structured result via submit_result.\n"
        f"- Keep the spec compact: 10-18 acceptance criteria.\n\n"
        f"Include interface definitions (signatures), testable acceptance "
        f"criteria (each AC maps to a specific test), and a file_structure "
        f"showing where source and test files should go."
    )


def review_prompt(spec_json: str) -> str:
    return (
        f"Review this implementation spec. Check that every acceptance "
        f"criterion is testable (maps to a concrete, automatable test), "
        f"interfaces are specific enough to implement, and there are no "
        f"ambiguous requirements.\n\n"
        f"Operational rules: do not use tools except submit_result; return only "
        f"the structured review.\n\n{spec_json}"
    )


def revise_spec_prompt(
    original_spec_json: str,
    feedback: str,
    issues: list[ReviewIssue],
) -> str:
    issues_text = json.dumps(
        [i.model_dump() for i in issues], indent=2, ensure_ascii=False,
    )
    return (
        f"Revise the spec based on reviewer feedback.\n\n"
        f"## Original spec\n{original_spec_json}\n\n"
        f"## Feedback\n{feedback}\n\n"
        f"## Issues\n{issues_text}\n\n"
        f"Operational rules: do not use tools except submit_result; return only "
        f"the revised structured spec.\n\n"
        f"Address every issue. Keep all existing ACs that were fine; "
        f"fix or add the ones flagged."
    )


def test_writing_task(
    test_framework: str,
    spec_json: str,
) -> str:
    return (
        f"Write {test_framework} test files based on the spec below. "
        f"Create one test function per acceptance criterion. "
        f"Use the file paths from the spec's file_structure.test_files. "
        f"The tests should be runnable (import the modules that will "
        f"be implemented) but are expected to FAIL right now since "
        f"no implementation exists yet.\n\n"
        f"Testing rules:\n"
        f"- Never assert exact float equality on time-dependent values "
        f"(tokens, elapsed time). Use pytest.approx() with abs=1e-4.\n"
        f"- For throughput/rate tests, use a generous relative tolerance "
        f"(e.g., rel=0.15) and run for at least 5 seconds.\n"
        f"- When draining a bucket to 0, accept that tokens may be a tiny "
        f"positive epsilon — the refill clock never stops.\n\n"
        f"## Spec\n{spec_json}"
    )


def dev_fix_task(
    round_n: int,
    max_rounds: int,
    passed: int,
    total: int,
    failures: list[TestFailure],
) -> str:
    failure_summary = "\n".join(
        f"- {f.test_name}: {f.error_message}"
        for f in failures
    )
    if any(f.test_name.startswith("code-review:") for f in failures):
        return (
            f"Fix the code review findings (round {round_n}/{max_rounds}).\n\n"
            f"Treat every finding below as required implementation work.\n\n"
            f"Findings:\n{failure_summary}\n\n"
            f"- Do NOT modify test files.\n"
            f"- Implement the missing functionality identified in the findings.\n"
            f"- Run a targeted smoke check, then return control.\n"
        )
    if any(f.test_name.startswith("test-file-mutation:") for f in failures):
        return (
            f"Recover from invalid test modification (round {round_n}/{max_rounds}).\n\n"
            f"The previous round modified test files. The workflow restored "
            f"the originals and rejected that round.\n\n"
            f"Mutations:\n{failure_summary}\n\n"
            f"- Do NOT modify test files.\n"
            f"- Fix only source and configuration files.\n"
            f"- Run a targeted smoke check, then return control.\n"
        )
    return (
        f"Fix the failing tests (round {round_n}/{max_rounds}).\n\n"
        f"Previous round: {passed}/{total} passed.\n\n"
        f"Failures:\n{failure_summary}\n\n"
        f"- Do NOT modify test files to hide failures.\n"
        f"- Make a focused source or configuration change.\n"
        f"- Run a targeted smoke check, then return control; the outer "
        f"workflow owns full verification.\n"
    )


def test_run_task(
    test_framework: str,
    test_files: list[str],
) -> str:
    return (
        f"Run the test suite with {test_framework}. "
        f"Test files: {', '.join(test_files)}. "
        f"Do not reuse old logs or unrelated test output; "
        f"the total/pass/fail counts must come only from the listed files "
        f"and the commands you run in this repository. "
        f"Report the results and include raw output in stdout."
    )


def code_review_prompt(spec_json: str, review_round: int = 0) -> str:
    return (
        f"Review the implementation against the spec. Read the source "
        f"files and verify each acceptance criterion is properly "
        f"implemented. Check for correctness, code quality, and "
        f"potential issues.\n\n"
        f"Review round nonce: {review_round}. This is a fresh review of the "
        f"current filesystem state. Do not reuse findings from an earlier "
        f"review if the files have changed; re-check whether each finding is "
        f"still true.\n\n"
        f"Operational rules:\n"
        f"- Inspect implementation/source files first; do not read every test file.\n"
        f"- Use grep/glob/bash to summarize when possible instead of reading large files.\n"
        f"- If evidence is insufficient for an AC, mark that AC as cannot_judge.\n"
        f"- Always finish by calling submit_result exactly once with the structured "
        f"CodeReview object; do not return prose or Markdown.\n\n"
        f"## Spec\n{spec_json}"
    )


CODE_REVIEW_TASK = (
    "Review the implementation. Read the source files "
    "and for each acceptance criterion, verify it's correctly "
    "implemented. Check code quality and potential issues. "
    "Do not exhaust the tool budget reading every test file; submit a "
    "structured CodeReview result when you have enough evidence."
)
