"""Cap oversized tool results with middle-out truncation."""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    AtomInstallPriority,
    BusPriority,
    ImageContent,
    TextContent,
    ToolResult,
    ToolResultEvent,
)
from agentm.core.lib import (
    count_text_tokens,
    truncate_text_tokens_middle,
)
from agentm.extensions import ExtensionManifest

_DEFAULT_MAX_TOKENS: Final[int] = 50_000


class ToolResultCapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(gt=0, default=_DEFAULT_MAX_TOKENS)
    error_floor_tokens: int = Field(ge=0, default=0)


MANIFEST = ExtensionManifest(
    name="tool_result_cap",
    description=(
        "Cap oversized tool-result text with middle-out truncation while "
        "preserving image blocks."
    ),
    registers=("event:tool_result",),
    config_schema=ToolResultCapConfig,
    requires=(),
    priority=AtomInstallPriority.TOOL,
)


def install(session: Any, config: ToolResultCapConfig) -> None:
    _ToolResultCapRuntime(session, config).install()


class _ToolResultCapRuntime:
    def __init__(self, session: Any, config: ToolResultCapConfig) -> None:
        self._session = session
        self._max_tokens = config.max_tokens
        self._error_floor_tokens = config.error_floor_tokens

    def install(self) -> None:
        self._session.on(
            ToolResultEvent.CHANNEL,
            self.on_tool_result,
            priority=BusPriority.POST,
        )

    async def on_tool_result(self, event: ToolResultEvent) -> ToolResult | None:
        if event.tool_name == "read":
            return None
        result = event.result
        if result is None:
            return None

        model_name = self._model_name()
        full_text = self._text_payload(result)
        total_tokens = count_text_tokens(full_text, model=model_name)
        if total_tokens <= self._max_tokens:
            return None

        # Honour error_floor_tokens: if the result is an error and
        # within the floor, don't truncate.
        effective_max = self._max_tokens
        if result.is_error and self._error_floor_tokens > 0:
            effective_max = max(
                self._max_tokens, min(self._error_floor_tokens, total_tokens)
            )
            if total_tokens <= effective_max:
                return None

        return self._truncated_result(
            event,
            result=result,
            total_tokens=total_tokens,
            effective_max=effective_max,
            model_name=model_name,
        )

    def _model_name(self) -> str | None:
        return self._session.model.id if self._session.model is not None else None

    @staticmethod
    def _text_payload(result: ToolResult) -> str:
        text_blocks: list[str] = []
        for block in result.content:
            if isinstance(block, TextContent):
                text_blocks.append(block.text)
        return "\n".join(text_blocks)

    def _truncated_result(
        self,
        event: ToolResultEvent,
        *,
        result: ToolResult,
        total_tokens: int,
        effective_max: int,
        model_name: str | None,
    ) -> ToolResult:
        remaining_tokens = effective_max
        kept_tokens = 0
        new_content_trunc: list[TextContent | ImageContent] = []
        for block in result.content:
            if isinstance(block, ImageContent):
                new_content_trunc.append(block)
                continue
            if remaining_tokens <= 0:
                continue
            truncated = truncate_text_tokens_middle(
                block.text, remaining_tokens, model=model_name
            )
            new_content_trunc.append(TextContent(type="text", text=truncated.text))
            kept_tokens += truncated.kept_tokens
            remaining_tokens -= truncated.kept_tokens

        truncated_tokens = total_tokens - kept_tokens
        total_lines = sum(
            block.text.count("\n") + 1
            for block in result.content
            if isinstance(block, TextContent)
        )
        new_content_trunc.append(
            TextContent(
                type="text",
                text=(
                    f"\n[tool_result_cap: truncated {truncated_tokens} tokens; "
                    f"original {total_tokens} tokens / {total_lines} lines "
                    f"from {event.tool_name}]"
                ),
            )
        )
        return ToolResult(
            content=new_content_trunc,
            is_error=result.is_error,
            extras=result.extras,
        )
