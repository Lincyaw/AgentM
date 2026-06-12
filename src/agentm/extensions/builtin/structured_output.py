"""``structured_output`` atom — register a terminal ``submit_result`` tool.

The caller supplies a JSON Schema via atom config; the atom builds a
``submit_result`` tool whose single ``result`` parameter conforms to that
schema. When the worker calls the tool, the session terminates and the
structured data is returned to the orchestrator.

Designed for use with the ``workflow`` atom's ``agent(prompt, schema=...)``
convenience — the workflow wires the atom config and extracts the result
automatically. Can also be loaded standalone in any scenario that needs a
schema-constrained terminal tool.

single-file contract: stdlib + ``agentm.core.abi.*`` +
``agentm.extensions.*``. No atom-to-atom imports.
"""

from __future__ import annotations

import json
from loguru import logger
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate
from agentm.extensions import ExtensionManifest


class StructuredOutputConfig(BaseModel):
    result_schema: dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        # Accept "schema" from external callers (manifest config dicts use
        # the original key name).
        if "schema" in data and "result_schema" not in data:
            data["result_schema"] = data.pop("schema")
        super().__init__(**data)


MANIFEST = ExtensionManifest(
    name="structured_output",
    description=(
        "Register a terminal submit_result tool whose parameter schema "
        "is supplied via atom config. The tool validates the input and "
        "terminates the session."
    ),
    registers=("tool:submit_result",),
    config_schema=StructuredOutputConfig,
)


def install(api: Any, config: StructuredOutputConfig) -> None:
    schema: dict[str, Any] = config.result_schema

    # Soft-dep: validate against the schema if jsonschema is available.
    _validate_fn: Any = None
    try:
        from jsonschema import validate as _jschema_validate, ValidationError  # type: ignore[import-untyped]

        _validate_fn = _jschema_validate
        _validation_error_cls = ValidationError
    except ImportError:
        logger.debug(
            "jsonschema not installed; submit_result will skip validation "
            "(LLM schema enforcement at the provider level is the primary gate)"
        )
        _validation_error_cls = None

    tool_params: dict[str, Any] = {
        "type": "object",
        "properties": {
            "result": schema,
        },
        "required": ["result"],
    }

    async def _submit_result(args: dict[str, Any]) -> ToolTerminate:
        result = args.get("result")

        if _validate_fn is not None and _validation_error_cls is not None:
            try:
                _validate_fn(result, schema)
            except _validation_error_cls as exc:
                return ToolTerminate(
                    result=ToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=json.dumps(
                                    {
                                        "error": "schema_validation_failed",
                                        "detail": str(exc.message),
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                        ],
                        is_error=True,
                    ),
                    reason="workflow:structured_output",
                )

        return ToolTerminate(
            result=ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(result, ensure_ascii=False),
                    )
                ],
            ),
            reason="workflow:structured_output",
        )

    api.register_tool(
        FunctionTool(
            name="submit_result",
            description=(
                "Submit your structured result and end this session. "
                "The result must conform to the schema provided in the "
                "tool parameters."
            ),
            parameters=tool_params,
            fn=_submit_result,
        )
    )
