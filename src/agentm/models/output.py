"""Structured output schemas for Orchestrator final responses.

These Pydantic models are used with create_react_agent's response_format
parameter. When set, the framework appends a generate_structured_response
node that invokes the LLM with structured output after the ReAct loop.

Registry lookup: OUTPUT_SCHEMAS maps schema_name -> model class.

Domain-specific schemas live in their canonical locations under ``scenarios/``.
"""

from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Schema registry (lazy-loaded)
# ---------------------------------------------------------------------------

OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {}
_defaults_loaded = False


def _ensure_defaults() -> None:
    """Lazily import and register default output schemas from scenarios."""
    global _defaults_loaded
    if _defaults_loaded:
        return
    _defaults_loaded = True

    from agentm.scenarios.rca.output import CausalGraph
    from agentm.scenarios.memory_extraction.output import KnowledgeSummary

    OUTPUT_SCHEMAS.setdefault("CausalGraph", CausalGraph)
    OUTPUT_SCHEMAS.setdefault("KnowledgeSummary", KnowledgeSummary)


def get_output_schema(schema_name: str) -> type[BaseModel]:
    """Look up an output schema by name.

    Raises ValueError if the schema is not registered.
    """
    _ensure_defaults()
    if schema_name not in OUTPUT_SCHEMAS:
        available = list(OUTPUT_SCHEMAS.keys())
        raise ValueError(
            f"Unknown output schema: {schema_name!r}. Available: {available}"
        )
    return OUTPUT_SCHEMAS[schema_name]
