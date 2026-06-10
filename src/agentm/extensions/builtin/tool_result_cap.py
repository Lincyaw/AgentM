"""Cap tool-result size; spill large outputs to disk.

When a tool result exceeds ``max_chars`` (default 8000), the full text is
persisted to ``.agentm/tool_outputs/<tool_call_id>.txt`` and the in-context
payload is replaced with a truncated preview plus the file path so the model
can ``read`` the spill file on demand. This keeps conversation history lean
and avoids blowing provider context windows on unexpectedly large outputs
(e.g. a download script dumping megabytes of stdout).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import ImageContent, TextContent, ToolResult, ToolResultEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest


class ToolResultCapConfig(BaseModel):
    model_config = {"extra": "allow"}

    max_chars: int = 8000
    preview_chars: int = 2000


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
    max_chars = max(0, config.max_chars)
    preview_chars = max(0, config.preview_chars)
    output_dir = Path(api.cwd) / ".agentm" / "tool_outputs"

    def _on_tool_result(event: ToolResultEvent) -> ToolResult | None:
        # Measure total text length across all TextContent blocks.
        text_blocks: list[str] = []
        total_chars = 0
        for block in event.result.content:
            if isinstance(block, TextContent):
                text_blocks.append(block.text)
                total_chars += len(block.text)

        if total_chars <= max_chars:
            return None

        # Spill to disk.
        full_text = "\n".join(text_blocks)
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
        preview = full_text[:preview_chars]
        notice = (
            f"\n\n[Output truncated: {total_chars} chars total. "
            f"Full output saved to {spill_path}. "
            f"Use the read tool to inspect it.]"
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
