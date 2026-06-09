"""``structured_output`` atom — register a terminal ``submit_result`` tool.

The caller supplies a JSON Schema via atom config; the atom builds a
``submit_result`` tool whose single ``result`` parameter conforms to that
schema. When the worker calls the tool, the session terminates and the
structured data is returned to the orchestrator.

Designed for use with the ``workflow`` atom's ``agent(prompt, schema=...)``
convenience — the workflow wires the atom config and extracts the result
automatically. Can also be loaded standalone in any scenario that needs a
schema-constrained terminal tool.

§11 single-file contract: stdlib + ``agentm.core.abi.*`` +
``agentm.extensions.*``. No atom-to-atom imports.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate
from agentm.extensions import ExtensionManifest

_log = logging.getLogger(__name__)

MANIFEST = ExtensionManifest(
    name="structured_output",
    description=(
        "Register a terminal submit_result tool whose parameter schema "
        "is supplied via atom config. The tool validates the input and "
        "terminates the session."
    ),
    registers=("tool:submit_result",),
    config_schema={
        "type": "object",
        "properties": {
            "schema": {
                "type": "object",
                "description": "JSON Schema for the result parameter.",
            },
        },
        "required": ["schema"],
        "additionalProperties": False,
    },
)


def install(api: Any, config: dict[str, Any]) -> None:
    schema: dict[str, Any] = config["schema"]

    # Soft-dep: validate against the schema if jsonschema is available.
    _validate_fn: Any = None
    try:
        from jsonschema import validate as _jschema_validate, ValidationError  # type: ignore[import-untyped]

        _validate_fn = _jschema_validate
        _validation_error_cls = ValidationError
    except ImportError:
        _log.debug(
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
        "additionalProperties": False,
    }

    async def _submit_result(args: dict[str, Any]) -> ToolTerminate:
        result = args.get("result")

        if _validate_fn is not None and _validation_error_cls is not None:
            try:
                _validate_fn(result, schema)
            except _validation_error_cls as exc:
                return ToolTerminate(
                    result=ToolResult(
                        content=[TextContent(
                            type="text",
                            text=json.dumps({
                                "error": "schema_validation_failed",
                                "detail": str(exc.message),
                            }, ensure_ascii=False),
                        )],
                        is_error=True,
                    ),
                    reason="workflow:structured_output",
                )

        return ToolTerminate(
            result=ToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False),
                )],
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
