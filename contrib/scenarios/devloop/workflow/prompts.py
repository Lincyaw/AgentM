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
        f"Include interface definitions (signatures), testable acceptance "
        f"criteria (each AC maps to a specific test), and a file_structure "
        f"showing where source and test files should go."
    )


def review_prompt(spec_json: str) -> str:
    return (
        f"Review this implementation spec. Check that every acceptance "
        f"criterion is testable (maps to a concrete, automatable test), "
        f"interfaces are specific enough to implement, and there are no "
        f"ambiguous requirements.\n\n{spec_json}"
    )


def revise_spec_prompt(
    original_spec_json: str,
    feedback: str,
    issues: list[ReviewIssue],
) -> str:
    issues_text = json.dumps(issues, indent=2, ensure_ascii=False)
    return (
        f"Revise the spec based on reviewer feedback.\n\n"
        f"## Original spec\n{original_spec_json}\n\n"
        f"## Feedback\n{feedback}\n\n"
        f"## Issues\n{issues_text}\n\n"
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
        f"no implementation exists yet."
    )


def dev_fix_task(
    round_n: int,
    max_rounds: int,
    passed: int,
    total: int,
    failures: list[TestFailure],
) -> str:
    failure_summary = "\n".join(
        f"- {f['test_name']}: {f['error_message']}"
        for f in failures
    )
    return (
        f"Fix the failing tests (round {round_n}/{max_rounds}).\n\n"
        f"Previous round: {passed}/{total} passed.\n\n"
        f"Failures:\n{failure_summary}\n\n"
        f"Do NOT modify the test files — fix the implementation only."
    )


def test_run_task(
    test_framework: str,
    test_files: list[str],
) -> str:
    return (
        f"Run the test suite with {test_framework}. "
        f"Test files: {', '.join(test_files)}. "
        f"Run them and report the results. Include the raw "
        f"output in stdout."
    )


def code_review_prompt(spec_json: str) -> str:
    return (
        f"Review the implementation against the spec. Read the source "
        f"files and verify each acceptance criterion is properly "
        f"implemented. Check for correctness, code quality, and "
        f"potential issues.\n\n## Spec\n{spec_json}"
    )


CODE_REVIEW_TASK = (
    "Review the implementation. Read the source files "
    "and for each acceptance criterion, verify it's correctly "
    "implemented. Check code quality and potential issues."
)
