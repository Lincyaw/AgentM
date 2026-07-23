"""Structured-output schemas for the pattern miner (host-side)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int = Field(description="Turn index in the trajectory")
    observation: str = Field(
        description="What this turn shows, in one or two sentences"
    )


class PatternFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(
        description=(
            "The reusable failure pattern, phrased as a tendency on a class "
            "of task — not a retelling of this case"
        )
    )
    instance: str = Field(description="How the pattern shows up in this case")
    evidence: list[EvidenceRef] = Field(description="Turn-grounded evidence")
    online_signature: str = Field(
        description=(
            "What a live monitor could observe in the trajectory alone, "
            "without ground truth, to justify triggering this check; "
            "'none' if nothing qualifies"
        )
    )
    gt_dependence: str = Field(
        description="Which parts of this finding depended on ground truth"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="0..1")


class AngleReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str = Field(
        description="The dimension under review, exactly as named in the task"
    )
    findings: list[PatternFinding] = Field(
        description="Empty if the dimension does not fire on this case"
    )
    outside_model: str = Field(
        default="",
        description="Observations that fit no dimension; empty if none",
    )


class ShadowLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shadow_dimension: str = Field(description="The dimension that fired downstream")
    root_dimension: str = Field(description="The root it shadows")
    why_observable: str = Field(
        description="Why the shadow is the observable online trigger for the root"
    )


class RootAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dimension: str = Field(
        description=(
            "Most upstream causally-decisive dimension; 'none' when no chain "
            "mismatch exists or the label is invalid"
        )
    )
    counterfactual: str = Field(
        description="Why correcting the root would have changed the outcome"
    )
    shadows: list[ShadowLink] = Field(description="Downstream firings, paired to roots")
    rationale: str = Field(description="The attribution argument, turn-grounded")


class ChecklistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str = Field(description="Dimension this item belongs to")
    check: str = Field(
        description=(
            "The check question, phrased in solving-chain language so a "
            "critic can execute it on any future case"
        )
    )
    when: str = Field(
        description=("Applicability trigger computable from a live task and trajectory")
    )
    how: str = Field(
        description=(
            "Cheapest sufficient sensor: structural rule, LLM read, or "
            "active probe — and what it does"
        )
    )
    evidence: list[str] = Field(description="Trial names supporting this item")
    notes: str = Field(
        default="",
        description="Caveats, e.g. findings too case-bound to generalize",
    )


class ChecklistDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ChecklistItem] = Field(
        description="Merged checklist for the dimension under distillation"
    )
    dropped: str = Field(
        default="",
        description="Findings deliberately not turned into items, and why",
    )
