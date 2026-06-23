"""Extraction result schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    name: str = Field(description="Canonical entity name")
    kind: str = Field(description="One of: variable, object, concept, tool, file, api, state_field")
    summary: str | None = Field(default=None, description="One-sentence description")
    aliases: list[str] = Field(default_factory=list, description="Alternative surface forms")


class ExtractedMention(BaseModel):
    entity_name: str = Field(description="Must match an entity name exactly")
    step_index: int = Field(description="Step index where this mention appears")
    text: str = Field(description="Short phrase (< 50 chars)")
    mention_type: str = Field(
        default="use",
        description="One of: define, use, read, write, mutate, question, answer, tool_input, tool_output",
    )


class ExtractedRelation(BaseModel):
    from_entity: str = Field(description="Source entity name")
    to_entity: str = Field(description="Target entity name")
    relation_type: str = Field(
        description="One of: uses, defines, updates, derived_from, input_to, output_of, mentions, explains",
    )
    step_index: int = Field(description="Step where this relation was observed")


class ReportEntitiesParams(BaseModel):
    entities: list[ExtractedEntity] = Field(description="All extracted entities")
    mentions: list[ExtractedMention] = Field(default_factory=list, description="Entity mentions")
    relations: list[ExtractedRelation] = Field(default_factory=list, description="Relations between entities")
