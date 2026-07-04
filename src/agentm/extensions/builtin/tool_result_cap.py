"""Cap tool-result size: spill large outputs to disk, truncate the rest.

Unified atom combining two strategies for oversized tool results:

1. **Spill** — when ``spill_to_disk`` is enabled (default) and a tool result
   exceeds ``max_tokens``, the full text is written to
   ``.agentm/tool_outputs/<tool_call_id>.txt`` and the in-context payload is
   replaced with a ``preview_tokens``-bounded preview plus the file path.

2. **Middle-out truncation** — fallback when spill fails (disk write error)
   or when ``spill_to_disk`` is disabled.  Preserves the first and last halves
   of the token limit so the model sees both initial context (headers, imports)
   and trailing state (errors, final output).  Error results are guaranteed at
   least ``error_floor_tokens`` of payload.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

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
    expand_path,
    truncate_text_tokens,
    truncate_text_tokens_middle,
)
from agentm.extensions import ExtensionManifest

_SPILL_READ_EXAMPLE_LIMIT: Final[int] = 200


class ToolResultCapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(gt=0)
    preview_tokens: int = Field(ge=0)
    error_floor_tokens: int = Field(ge=0, default=0)
    spill_to_disk: bool = True

MANIFEST = ExtensionManifest(
    name="tool_result_cap",
    description=(
        "Cap tool-result size: spill large outputs to disk when possible, "
        "fall back to middle-out token truncation."
    ),
    registers=("event:tool_result",),
    config_schema=ToolResultCapConfig,
    requires=(),
)


def install(api: ExtensionAPI, config: ToolResultCapConfig) -> None:
    max_tokens = config.max_tokens
    preview_tokens = config.preview_tokens
    error_floor_tokens = config.error_floor_tokens
    spill_to_disk = config.spill_to_disk
    output_dir = (expand_path(api.cwd) / ".agentm" / "tool_outputs").resolve()

    def _on_tool_result(event: ToolResultEvent) -> ToolResult | None:
        if event.tool_name == "read":
            return None

        model_name = api.model.id if api.model is not None else None

        text_blocks: list[str] = []
        for block in event.result.content:
            if isinstance(block, TextContent):
                text_blocks.append(block.text)

        full_text = "\n".join(text_blocks)
        total_tokens = count_text_tokens(full_text, model=model_name)
        if total_tokens <= max_tokens:
            return None

        # Honour error_floor_tokens: if the result is an error and
        # within the floor, don't truncate.
        effective_max = max_tokens
        if event.result.is_error and error_floor_tokens > 0:
            effective_max = max(max_tokens, min(error_floor_tokens, total_tokens))
            if total_tokens <= effective_max:
                return None

        # Strategy 1: spill to disk.
        spill_path: Path | None = None
        if spill_to_disk:
            candidate = output_dir / f"{event.tool_call_id}.txt"
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                tmp = candidate.with_name(f"{candidate.name}.tmp")
                tmp.write_text(full_text, encoding="utf-8")
                os.replace(tmp, candidate)
                spill_path = candidate
            except OSError:
                pass

        if spill_path is not None:
            preview = truncate_text_tokens(
                full_text, preview_tokens, model=model_name
            ).text
            notice = (
                f"\n\n[Output truncated: {total_tokens} tokens total. "
                f"Full output saved to {spill_path}. "
                "Inspect it with paged reads, for example: "
                f'read(path="{spill_path}", offset=1, limit={_SPILL_READ_EXAMPLE_LIMIT}).]'
            )
            new_content: list[TextContent | ImageContent] = [
                block for block in event.result.content
                if isinstance(block, ImageContent)
            ]
            new_content.append(TextContent(type="text", text=preview + notice))
            return ToolResult(
                content=new_content,
                is_error=event.result.is_error,
                extras=event.result.extras,
            )

        # Strategy 2: middle-out truncation (fallback).
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

    api.on(ToolResultEvent.CHANNEL, _on_tool_result)
