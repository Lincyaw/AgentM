"""Type definitions and JSON Schemas for the devloop workflow."""
from __future__ import annotations

from typing import Required, TypedDict

from agentm.extensions.builtin.workflow import JsonSchema

# ── JSON Schemas (passed to agent(schema=...)) ─────────────────

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

# ── TypedDict definitions ──────────────────────────────────────


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
