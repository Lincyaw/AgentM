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

from pydantic import BaseModel, Field

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest


class StructuredOutputConfig(BaseModel):
    result_schema: dict[str, Any] = Field(default_factory=dict)

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


class _SchemaValidator:
    def __init__(self) -> None:
        self._validate_fn: Any = None
        self._validation_error_cls: Any = None
        self._load()

    def _load(self) -> None:
        # Soft-dep: validate against the schema if jsonschema is available.
        try:
            from jsonschema import validate as _jschema_validate, ValidationError  # type: ignore[import-untyped]
        except ImportError:
            logger.debug(
                "jsonschema not installed; submit_result will skip validation "
                "(LLM schema enforcement at the provider level is the primary gate)"
            )
            return

        self._validate_fn = _jschema_validate
        self._validation_error_cls = ValidationError

    def validate_error(self, result: Any, schema: dict[str, Any]) -> str | None:
        if self._validate_fn is None or self._validation_error_cls is None:
            return None
        try:
            self._validate_fn(result, schema)
        except self._validation_error_cls as exc:
            return str(exc.message)
        return None


class _StructuredOutputRuntime:
    def __init__(self, api: ExtensionAPI, schema: dict[str, Any]) -> None:
        self._api = api
        self._schema = schema
        self._validator = _SchemaValidator()

    def install(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="submit_result",
                description=(
                    "Submit your structured result and end this session. "
                    "The result must conform to the schema provided in the "
                    "tool parameters."
                ),
                parameters=self._tool_params(),
                fn=self.submit_result,
            )
        )

    def _tool_params(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "result": self._schema,
            },
            "required": ["result"],
        }

    async def submit_result(self, args: dict[str, Any]) -> ToolTerminate:
        result = args.get("result")
        validation_error = self._validator.validate_error(result, self._schema)
        if validation_error is not None:
            return _terminate_result(
                {
                    "error": "schema_validation_failed",
                    "detail": validation_error,
                },
                is_error=True,
            )

        return _terminate_result(result)


def _terminate_result(payload: Any, *, is_error: bool = False) -> ToolTerminate:
    return ToolTerminate(
        result=ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(payload, ensure_ascii=False),
                )
            ],
            is_error=is_error,
        ),
        reason="workflow:structured_output",
    )


def install(api: ExtensionAPI, config: StructuredOutputConfig) -> None:
    _StructuredOutputRuntime(api, config.result_schema).install()
