"""``structured_output`` atom — register a terminal ``submit_result`` tool.

The caller supplies a JSON Schema via atom config; the atom builds a
``submit_result`` tool whose single ``result`` parameter conforms to that
schema. When the worker submits a schema-valid result, the session
terminates and the structured data is returned to the orchestrator; a
validation failure returns a retryable tool error instead of terminating.

Hosts and orchestration atoms can configure this tool when a child session
must return a machine-validated payload. It can also be loaded standalone in
any scenario that needs a schema-constrained terminal tool.

single-file contract: stdlib + ``agentm.core.abi.*`` +
``agentm.extensions.*``. No atom-to-atom imports.
"""

from __future__ import annotations

import json
from loguru import logger
from typing import Any

from pydantic import BaseModel, Field

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
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
        "is supplied via atom config. A schema-valid submission terminates "
        "the session; a validation failure returns a retryable tool error "
        "so the model can correct and resubmit in the same session."
    ),
    registers=("tool:submit_result",),
    config_schema=StructuredOutputConfig,
    priority=AtomInstallPriority.TOOL,
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
    def __init__(self, api: AtomAPI, schema: dict[str, Any]) -> None:
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
                    "tool parameters. A valid submission ends the session "
                    "immediately; if schema validation fails, the session "
                    "continues and the error details tell you what to fix — "
                    "correct the result and call submit_result again."
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

    async def submit_result(
        self, args: dict[str, Any]
    ) -> ToolTerminate | ToolResult:
        result = args.get("result")
        validation_error = self._validator.validate_error(result, self._schema)
        if validation_error is not None and isinstance(result, str):
            # Recovery for double-encoded submissions: models sometimes pass
            # the result as a JSON string instead of an object. Unwrap only
            # when the raw string fails validation and the decoded value
            # passes — a legitimately-string result is never reinterpreted.
            # This is the single unwrap point; downstream consumers must not
            # second-guess it.
            try:
                decoded = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if (
                isinstance(decoded, (dict, list))
                and self._validator.validate_error(decoded, self._schema) is None
            ):
                logger.debug(
                    "submit_result: unwrapped double-encoded JSON string result"
                )
                result = decoded
                validation_error = None
        if validation_error is not None:
            # Retryable: keep the session alive so the model can correct the
            # payload and resubmit, instead of terminating with a dead error
            # result the orchestrator can only retry from scratch.
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "schema_validation_failed",
                                "detail": validation_error,
                                "hint": (
                                    "Fix the result to match the schema and "
                                    "call submit_result again."
                                ),
                            },
                            ensure_ascii=False,
                        ),
                    )
                ],
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


def install(api: AtomAPI, config: StructuredOutputConfig) -> None:
    _StructuredOutputRuntime(api, config.result_schema).install()
