"""Pydantic models for the paper_review workflow."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PaperReviewArgs(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str = Field(description="Paper file or directory.")
    output_path: str | None = None
    agent_timeout_seconds: float = 1200.0
    agent_retries: int = 2
