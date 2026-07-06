"""Cap tool-result size: spill large outputs to a file, truncate the rest.

Unified atom combining two strategies for oversized tool results:

1. **Spill** — when ``spill_to_disk`` is enabled (default) and a tool result
   exceeds ``max_tokens``, the full text is written via the session
   ``ResourceWriter`` to ``.agentm/tool_outputs/<session_id>/<tool_call_id>.txt``
   (workspace-relative, so the ``read`` tool can open it under BOTH the local
   and remote operations backends) and the in-context payload is replaced
   with a ``preview_tokens``-bounded preview plus the file path.

2. **Middle-out truncation** — fallback when spill fails (write error)
   or when ``spill_to_disk`` is disabled.  Preserves the first and last halves
   of the token limit so the model sees both initial context (headers, imports)
   and trailing state (errors, final output).  Error results are guaranteed at
   least ``error_floor_tokens`` of payload.
"""

from __future__ import annotations

import re
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    ExtensionAPI,
    ImageContent,
    TextContent,
    ToolResult,
    ToolResultEvent,
)
from agentm.core.lib import (
    count_text_tokens,
    truncate_text_tokens,
    truncate_text_tokens_middle,
)
from agentm.extensions import ExtensionManifest

_SPILL_READ_EXAMPLE_LIMIT: Final[int] = 200
_SAFE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_path_name(value: str, *, fallback: str) -> str:
    sanitized = _SAFE_NAME_PATTERN.sub("_", value).strip("._")
    return sanitized[:120] if sanitized else fallback


class ToolResultCapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Defaults let the session factory floor-mount this atom with empty
    # config; scenarios override per their context budget.
    max_tokens: int = Field(gt=0, default=50_000)
    preview_tokens: int = Field(ge=0, default=50_000)
    error_floor_tokens: int = Field(ge=0, default=0)
    spill_to_disk: bool = True


MANIFEST = ExtensionManifest(
    name="tool_result_cap",
    description=(
        "Cap tool-result size: spill large outputs to a workspace file "
        "(.agentm/tool_outputs/, via ResourceWriter so it works on local "
        "and remote backends), fall back to middle-out token truncation."
    ),
    registers=("event:tool_result",),
    config_schema=ToolResultCapConfig,
    requires=(),
)


def install(api: ExtensionAPI, config: ToolResultCapConfig) -> None:
    _ToolResultCapRuntime(api, config).install()


class _ToolResultCapRuntime:
    def __init__(self, api: ExtensionAPI, config: ToolResultCapConfig) -> None:
        self._api = api
        self._max_tokens = config.max_tokens
        self._preview_tokens = config.preview_tokens
        self._error_floor_tokens = config.error_floor_tokens
        self._spill_to_disk = config.spill_to_disk
        # Workspace-relative so the read tool resolves it under both local
        # and remote operations backends.
        self._output_dir = f".agentm/tool_outputs/{self._session_dir_name()}"
        # Lazy-resolved ResourceWriter (resolving at install time is a §11
        # anti-pattern; the writer atom may install after this one).
        self._writer_cache: list[Any] = []

    def install(self) -> None:
        self._api.on(ToolResultEvent.CHANNEL, self.on_tool_result)

    def _get_writer(self) -> Any:
        if not self._writer_cache:
            self._writer_cache.append(self._api.get_resource_writer())
        return self._writer_cache[0]

    async def on_tool_result(self, event: ToolResultEvent) -> ToolResult | None:
        if event.tool_name == "read":
            return None

        model_name = self._model_name()
        full_text = self._text_payload(event.result)
        total_tokens = count_text_tokens(full_text, model=model_name)
        if total_tokens <= self._max_tokens:
            return None

        # Honour error_floor_tokens: if the result is an error and
        # within the floor, don't truncate.
        effective_max = self._max_tokens
        if event.result.is_error and self._error_floor_tokens > 0:
            effective_max = max(
                self._max_tokens, min(self._error_floor_tokens, total_tokens)
            )
            if total_tokens <= effective_max:
                return None

        # Strategy 1: spill to a workspace file.
        spill_path = await self._spill(event.tool_call_id, full_text)

        if spill_path is not None:
            return self._spilled_result(
                event,
                full_text=full_text,
                total_tokens=total_tokens,
                spill_path=spill_path,
                model_name=model_name,
            )

        # Strategy 2: middle-out truncation (fallback).
        return self._truncated_result(
            event,
            total_tokens=total_tokens,
            effective_max=effective_max,
            model_name=model_name,
        )

    def _session_dir_name(self) -> str:
        session_id = getattr(self._api, "session_id", None)
        if isinstance(session_id, str) and session_id:
            return _safe_path_name(session_id, fallback="unknown-session")
        return "unknown-session"

    def _model_name(self) -> str | None:
        return self._api.model.id if self._api.model is not None else None

    @staticmethod
    def _text_payload(result: ToolResult) -> str:
        text_blocks: list[str] = []
        for block in result.content:
            if isinstance(block, TextContent):
                text_blocks.append(block.text)
        return "\n".join(text_blocks)

    async def _spill(self, tool_call_id: str, full_text: str) -> str | None:
        if not self._spill_to_disk:
            return None
        filename = f"{_safe_path_name(tool_call_id, fallback='tool-call')}.txt"
        candidate = f"{self._output_dir}/{filename}"
        try:
            result = await self._get_writer().write(
                candidate,
                full_text.encode("utf-8"),
                rationale="tool_result_cap spill",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("tool_result_cap: spill write failed for {}: {}", candidate, exc)
            return None
        if result.error is not None:
            logger.debug(
                "tool_result_cap: spill rejected for {}: {}", candidate, result.error
            )
            return None
        return candidate

    def _spilled_result(
        self,
        event: ToolResultEvent,
        *,
        full_text: str,
        total_tokens: int,
        spill_path: str,
        model_name: str | None,
    ) -> ToolResult:
        preview = truncate_text_tokens(
            full_text, self._preview_tokens, model=model_name
        ).text
        notice = (
            f"\n\n[Output truncated: {total_tokens} tokens total. "
            f"Full output saved to {spill_path} (workspace-relative). "
            "Inspect it with paged reads, for example: "
            f'read(path="{spill_path}", offset=1, limit={_SPILL_READ_EXAMPLE_LIMIT}).]'
        )
        new_content: list[TextContent | ImageContent] = [
            block for block in event.result.content if isinstance(block, ImageContent)
        ]
        new_content.append(TextContent(type="text", text=preview + notice))
        return ToolResult(
            content=new_content,
            is_error=event.result.is_error,
            extras=event.result.extras,
        )

    def _truncated_result(
        self,
        event: ToolResultEvent,
        *,
        total_tokens: int,
        effective_max: int,
        model_name: str | None,
    ) -> ToolResult:
        remaining_tokens = effective_max
        kept_tokens = 0
        new_content_trunc: list[TextContent | ImageContent] = []
        for block in event.result.content:
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
            for block in event.result.content
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
            is_error=event.result.is_error,
            extras=event.result.extras,
        )
