"""``/context`` — show token usage stats for the current session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..protocol import (
    CommandContext,
    CommandInvocation,
    CommandKind,
    CommandResult,
)


@dataclass(slots=True)
class ContextCommand:
    name: str = "context"
    namespace: str | None = None
    summary: str = "Show token usage for this session"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del inv
        stats = ctx.get_route_stats()
        session_id = stats.get("session_id")
        if not session_id:
            return CommandResult(outbound=[ctx.reply("No active session.")])

        from agentm.core.runtime.otel_export import resolve_observability_dir

        trace_path = resolve_observability_dir(ctx.cwd) / f"{session_id}.jsonl"
        if not trace_path.exists():
            return CommandResult(outbound=[ctx.reply("No trace data yet.")])

        from agentm.cli_trace import TraceReader

        records = TraceReader(trace_path).load_turn_summaries()
        if not records:
            return CommandResult(outbound=[ctx.reply("No turns recorded yet.")])

        total_input = sum(r.get("input_tokens", 0) for r in records)
        total_output = sum(r.get("output_tokens", 0) for r in records)
        cache_read = sum(r.get("cache_read", 0) for r in records)
        cache_write = sum(r.get("cache_write", 0) for r in records)
        non_cached = total_input - cache_read
        hit_pct = (cache_read / total_input * 100) if total_input else 0.0
        total = total_input + total_output

        body = (
            "\U0001f4ca Session context usage\n\n"
            f"turns:            {len(records)}\n"
            f"input tokens:     {total_input:>12,}\n"
            f"  cache read:     {cache_read:>12,}  ({hit_pct:.1f}%)\n"
            f"  cache write:    {cache_write:>12,}\n"
            f"  non-cached:     {non_cached:>12,}\n"
            f"output tokens:    {total_output:>12,}\n"
            f"total tokens:     {total:>12,}\n"
        )
        return CommandResult(outbound=[ctx.reply(body)])


HANDLER = ContextCommand()
