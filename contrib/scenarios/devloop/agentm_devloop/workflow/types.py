"""Pydantic models for the devloop workflow.

Each model serves as both the runtime type AND the JSON Schema source
(via ``pydantic_to_tool_schema``). No hand-written schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

_STRICT = ConfigDict(extra="forbid")


# ── Workflow Args ─────────────────────────────────────────────


class DevloopArgs(BaseModel):
    model_config = ConfigDict(extra="allow")
    requirement: str = Field(description="Natural-language feature requirement.")
    language: str = Field(default="python", description="Target language.")
    test_framework: str = Field(default="pytest", description="Test framework.")
    max_rounds: int = Field(default=3, description="Max develop+test iterations.")
    skip_review: bool = Field(default=False, description="Skip final code review.")
    agent_timeout_seconds: float = Field(
        default=900.0, description="Per-agent wall-clock timeout."
    )


# ── Spec ───────────────────────────────────────────────────────


class InterfaceSpec(BaseModel):
    model_config = _STRICT
    name: str = Field(description="Function/class/method name.")
    signature: str = Field(description="Full signature with types.")
    description: str = Field(default="", description="What it does.")


class AcceptanceCriterion(BaseModel):
    model_config = _STRICT
    id: str = Field(description="e.g. AC-1")
    description: str = Field(description="What to verify.")
    test_method: str = Field(
        default="", description="How to verify (test name or command)."
    )


class FileStructure(BaseModel):
    model_config = _STRICT
    source_files: list[str] = Field(
        default_factory=list, description="Paths for implementation files."
    )
    test_files: list[str] = Field(
        default_factory=list, description="Paths for test files."
    )


class ImplementationSpec(BaseModel):
    model_config = _STRICT
    title: str = Field(description="Short feature title.")
    description: str = Field(default="", description="What the feature does.")
    interfaces: list[InterfaceSpec] = Field(
        description="Function/class/method signatures."
    )
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        description="Testable acceptance criteria."
    )
    file_structure: FileStructure = Field(
        default_factory=FileStructure, description="Where source and test files go."
    )


# ── Design Review ──────────────────────────────────────────────


class ReviewIssue(BaseModel):
    model_config = _STRICT
    ac_id: str = Field(default="", description="Which AC is affected.")
    issue: str = Field(default="", description="What's wrong.")

    @field_validator("ac_id", "issue", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        return "" if value is None else value


class DesignReview(BaseModel):
    model_config = _STRICT
    approved: bool = Field(description="Whether the spec passes review.")
    feedback: str = Field(default="", description="Overall feedback.")
    issues: list[ReviewIssue] = Field(
        default_factory=list, description="Per-AC issues found."
    )


# ── Test ───────────────────────────────────────────────────────


class TestFailure(BaseModel):
    model_config = _STRICT
    test_name: str = Field(description="Failing test name.")
    error_message: str = Field(description="Error message or traceback summary.")


class TestInfo(BaseModel):
    model_config = _STRICT
    test_files: list[str] = Field(description="Paths of test files written.")
    test_count: int = Field(default=0, description="Number of test functions.")


class TestResult(BaseModel):
    model_config = _STRICT
    all_passed: bool = Field(description="Whether every test passed.")
    total: int = Field(description="Total test count.")
    passed: int = Field(description="Passed count.")
    failed: int = Field(description="Failed count.")
    failures: list[TestFailure] = Field(
        default_factory=list, description="Details of each failure."
    )
    stdout: str = Field(default="", description="Raw pytest output.")


# ── Code Review ────────────────────────────────────────────────


class ACVerdict(BaseModel):
    model_config = _STRICT
    ac_id: str = Field(description="Acceptance criterion ID.")
    status: str = Field(description="pass, fail, or cannot_judge.")
    reason: str = Field(default="", description="Why this verdict.")


class Finding(BaseModel):
    model_config = _STRICT
    severity: str = Field(description="critical, major, or minor.")
    description: str = Field(description="What was found.")


class CodeReview(BaseModel):
    model_config = _STRICT
    approved: bool = Field(description="Whether the implementation passes review.")
    verdicts: list[ACVerdict] = Field(
        default_factory=list, description="Per-AC verdicts."
    )
    findings: list[Finding] = Field(
        default_factory=list, description="Code quality findings."
    )
