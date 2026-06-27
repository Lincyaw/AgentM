"""Cap tool-result size; spill large outputs to disk.

When a tool result exceeds the scenario-configured ``max_tokens``, the full
text is persisted to ``.agentm/tool_outputs/<tool_call_id>.txt`` and the
in-context payload is replaced with a token-bounded preview plus the file path so
the model can ``read`` the spill file on demand. This keeps conversation
history lean and avoids blowing provider context windows on unexpectedly
large outputs (e.g. a download script dumping megabytes of stdout).
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
from agentm.core.lib import count_text_tokens, truncate_text_tokens
from agentm.extensions import ExtensionManifest

_SPILL_READ_EXAMPLE_LIMIT: Final[int] = 200


class ToolResultCapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(gt=0)
    preview_tokens: int = Field(ge=0)

MANIFEST = ExtensionManifest(
    name="tool_result_cap",
    description=(
        "Cap tool-result size; spill large outputs to disk so the model "
        "can read them on demand."
    ),
    registers=("event:tool_result",),
    config_schema=ToolResultCapConfig,
    requires=(),
)

def install(api: ExtensionAPI, config: ToolResultCapConfig) -> None:
    max_tokens = config.max_tokens
    preview_tokens = config.preview_tokens
    output_dir = Path(api.cwd) / ".agentm" / "tool_outputs"

    def _on_tool_result(event: ToolResultEvent) -> ToolResult | None:
        if event.tool_name == "read":
            return None

        text_blocks: list[str] = []
        for block in event.result.content:
            if isinstance(block, TextContent):
                text_blocks.append(block.text)

        full_text = "\n".join(text_blocks)
        model_name = api.model.id if api.model is not None else None
        total_tokens = count_text_tokens(full_text, model=model_name)
        if total_tokens <= max_tokens:
            return None

        # Spill to disk.
        spill_path = output_dir / f"{event.tool_call_id}.txt"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = spill_path.with_name(f"{spill_path.name}.tmp")
            tmp_path.write_text(full_text, encoding="utf-8")
            os.replace(tmp_path, spill_path)
        except OSError:
            # If we can't write the spill file, don't replace the result —
            # let tool_result_budget handle the truncation instead.
            return None

        # Build replacement: preview + notice.
        preview = truncate_text_tokens(
            full_text,
            preview_tokens,
            model=model_name,
        ).text
        notice = (
            f"\n\n[Output truncated: {total_tokens} tokens total. "
            f"Full output saved to {spill_path}. "
            "Inspect it with paged reads, for example: "
            f'read(path="{spill_path}", offset=1, limit={_SPILL_READ_EXAMPLE_LIMIT}).]'
        )

        new_content: list[TextContent | ImageContent] = []
        # Preserve any image blocks from the original result.
        for block in event.result.content:
            if isinstance(block, ImageContent):
                new_content.append(block)
        new_content.append(TextContent(type="text", text=preview + notice))

        return ToolResult(
            content=new_content,
            is_error=event.result.is_error,
            extras=event.result.extras,
        )

    api.on(ToolResultEvent.CHANNEL, _on_tool_result)
