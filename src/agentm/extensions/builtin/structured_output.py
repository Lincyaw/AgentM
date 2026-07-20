"""``structured_output`` atom -- register a terminal ``submit_result`` tool.

The caller supplies a JSON Schema via atom config; the atom builds a
``submit_result`` tool whose single ``result`` parameter conforms to that
schema.  A schema-valid submission terminates the session; a validation
failure returns a retryable tool error instead.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel, Field

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    FunctionTool,
    JsonValue,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest


class StructuredOutputConfig(BaseModel):
    result_schema: dict[str, object] = Field(default_factory=dict)

    def __init__(self, **data: object) -> None:
        if "schema" in data and "result_schema" not in data:
            data["result_schema"] = data.pop("schema")
        super().__init__(**data)


MANIFEST = ExtensionManifest(
    name="structured_output",
    description=(
        "Register a terminal submit_result tool whose parameter schema "
        "is supplied via atom config."
    ),
    registers=("tool:submit_result",),
    config_schema=StructuredOutputConfig,
    priority=AtomInstallPriority.TOOL,
)


class _SchemaValidator:
    def __init__(self) -> None:
        self._validate_fn: Callable[[JsonValue, dict[str, object]], None] | None = None
        self._error_cls: type[Exception] | None = None
        try:
            from jsonschema import validate, ValidationError  # type: ignore[import-untyped]

            self._validate_fn = validate
            self._error_cls = ValidationError
        except ImportError:
            pass

    def check(self, result: JsonValue, schema: dict[str, object]) -> str | None:
        if self._validate_fn is None or self._error_cls is None:
            return None
        try:
            self._validate_fn(result, schema)
        except self._error_cls as exc:
            return str(exc)
        return None


class _StructuredOutputRuntime:
    def __init__(self, api: AtomAPI, schema: dict[str, object]) -> None:
        self._api = api
        self._schema = schema
        self._validator = _SchemaValidator()

    def install(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="submit_result",
                description=(
                    "Submit your structured result and end this session. "
                    "The result must conform to the schema in the tool "
                    "parameters. A valid submission ends the session; on "
                    "validation failure, correct and resubmit."
                ),
                parameters={  # code-health: ignore[AM011]
                    "type": "object",
                    "properties": {"result": self._schema},
                    "required": ["result"],
                },
                fn=self.submit_result,
            )
        )

    async def submit_result(
        self, args: dict[str, JsonValue]
    ) -> ToolTerminate | ToolResult:
        result = args.get("result")
        error = self._validator.check(result, self._schema)
        if error is not None and isinstance(result, str):
            decoded: JsonValue
            try:
                decoded = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if (
                isinstance(decoded, (dict, list))
                and self._validator.check(decoded, self._schema) is None
            ):
                result = decoded
                error = None
        if error is not None:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "schema_validation_failed",
                                "detail": error,
                                "hint": "Fix the result and call submit_result again.",
                            },
                            ensure_ascii=False,
                        ),
                    )
                ],
                is_error=True,
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
            reason="structured_output:submitted",
        )


def install(api: AtomAPI, config: StructuredOutputConfig) -> None:
    _StructuredOutputRuntime(api, config.result_schema).install()
