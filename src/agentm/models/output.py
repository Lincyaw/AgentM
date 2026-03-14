"""Structured output schemas for Orchestrator final responses.

These Pydantic models are used with create_react_agent's response_format
parameter. When set, the framework appends a generate_structured_response
node that invokes the LLM with structured output after the ReAct loop.

Registry lookup: OUTPUT_SCHEMAS maps schema_name -> model class.

Registries are populated by scenario ``register()`` functions called
via ``agentm.scenarios.discover()``.  The SDK core never imports from
``scenarios/`` directly.
"""

from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Schema registry — populated by scenario register() functions
# ---------------------------------------------------------------------------

OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {}


def get_output_schema(schema_name: str) -> type[BaseModel]:
    """Look up an output schema by name.

    Raises ValueError if the schema is not registered.
    """
    if schema_name not in OUTPUT_SCHEMAS:
        available = list(OUTPUT_SCHEMAS.keys())
        raise ValueError(
            f"Unknown output schema: {schema_name!r}. Available: {available}"
        )
    return OUTPUT_SCHEMAS[schema_name]
