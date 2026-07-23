# code-health: ignore-file[AM025] -- atom tools validate untyped tool, config, and service payloads
"""Cap oversized tool results and spill the full text to the workspace."""

from __future__ import annotations

import re
from typing import Final

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    RESOURCE_WRITER,
    AtomAPI,
    AtomInstallPriority,
    BusPriority,
    ImageContent,
    ResourceWriter,
    TextContent,
    ToolResult,
    ToolResultEvent,
)
from agentm.core.lib.tokens import (
    count_text_tokens,
    truncate_text_tokens_middle,
)
from agentm.extensions import ExtensionManifest

_DEFAULT_MAX_TOKENS: Final[int] = 50_000
_SPILL_READ_EXAMPLE_LIMIT: Final[int] = 200
_SAFE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_path_name(value: str, *, fallback: str) -> str:
    sanitized = _SAFE_NAME_PATTERN.sub("_", value).strip("._")
    return sanitized[:120] if sanitized else fallback


class ToolResultCapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(gt=0, default=_DEFAULT_MAX_TOKENS)
    error_floor_tokens: int = Field(ge=0, default=0)


MANIFEST = ExtensionManifest(
    name="tool_result_cap",
    description=(
        "Cap oversized tool-result text, spill the full text through the "
        "workspace ResourceWriter, and preserve image blocks."
    ),
    registers=("event:tool_result",),
    config_schema=ToolResultCapConfig,
    requires=("service:resource_writer",),
    priority=AtomInstallPriority.TOOL,
)


def install(session: AtomAPI, config: ToolResultCapConfig) -> None:
    _ToolResultCapRuntime(session, config).install()


class _ToolResultCapRuntime:
    def __init__(self, session: AtomAPI, config: ToolResultCapConfig) -> None:
        self._session = session
        self._max_tokens = config.max_tokens
        self._error_floor_tokens = config.error_floor_tokens
        writer = session.services.get_role(RESOURCE_WRITER)
        if writer is None:
            raise RuntimeError("tool_result_cap requires a ResourceWriter service")
        self._writer: ResourceWriter = writer
        session_name = _safe_path_name(
            session.ctx.session_id,
            fallback="unknown-session",
        )
        self._output_dir = f".agentm/tool_outputs/{session_name}"

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

        spill_path = await self._spill(event.tool_call_id, full_text)
        return self._truncated_result(
            event,
            result=result,
            total_tokens=total_tokens,
            effective_max=effective_max,
            model_name=model_name,
            spill_path=spill_path,
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

    async def _spill(self, tool_call_id: str, full_text: str) -> str | None:
        filename = f"{_safe_path_name(tool_call_id, fallback='tool-call')}.txt"
        candidate = f"{self._output_dir}/{filename}"
        try:
            result = await self._writer.write(
                candidate,
                full_text.encode("utf-8"),
                rationale="tool_result_cap full output spill",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "tool_result_cap failed to spill full output to {}: {}",
                candidate,
                exc,
            )
            return None
        if result.error is not None:
            logger.warning(
                "tool_result_cap spill was rejected for {}: {}",
                candidate,
                result.error,
            )
            return None
        return candidate

    def _truncated_result(
        self,
        event: ToolResultEvent,
        *,
        result: ToolResult,
        total_tokens: int,
        effective_max: int,
        model_name: str | None,
        spill_path: str | None,
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
        if spill_path is None:
            retrieval_notice = "full output could not be saved"
        else:
            retrieval_notice = (
                f"full output saved to {spill_path}; inspect it with paged reads, "
                f'for example read(path="{spill_path}", offset=1, '
                f"limit={_SPILL_READ_EXAMPLE_LIMIT})"
            )
        new_content_trunc.append(
            TextContent(
                type="text",
                text=(
                    f"\n[truncated {truncated_tokens} tokens; "
                    f"original {total_tokens} tokens / {total_lines} lines "
                    f"from {event.tool_name}; {retrieval_notice}]"
                ),
            )
        )
        return ToolResult(
            content=new_content_trunc,
            is_error=result.is_error,
            extras=result.extras,
        )
