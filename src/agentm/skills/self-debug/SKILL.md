---
name: self-debug
description: Inspect your own running session to debug yourself — see the terminal UI the user sees (tui_snapshot), read your own live trace, and mine historical traces. Use when the user reports a bug in how you behaved, a rendering/interaction glitch in the terminal client, or asks you to figure out what went wrong in this or a past session.
---

# self-debug

You are running inside the gateway. Your behaviour, your tool calls, and
the UI the user sees are all observable from where you sit. When something
looks wrong, look at the evidence before guessing.

## Principle

Three independent surfaces, each for a different question:

| Question | Surface |
|----------|---------|
| "What does the user actually see on screen?" | `tui_snapshot` tool |
| "What did *I* just do this session?" | `agentm trace … --latest` via bash |
| "What happened in some past session?" | `agentm trace index` → drill in |

Prefer evidence over speculation. The session trace is the source of
truth for your own actions; the conversation is a lossy view of it.

## See the UI (tui_snapshot)

> Scope: `tui_snapshot` exists **only for the terminal client**. A
> chat-channel deployment (e.g. Feishu/Lark over the gateway) has no TUI
> frame — skip this section there and rely on traces. See the
> `deployment-awareness` skill for what your deployment exposes.

The terminal client renders a frame the agent never sees directly. When
the user runs **`/dump`** in the client, that frame is written to a file
(`$AGENTM_TUI_DUMP`, default `/tmp/agentm-tui-dump.txt`). Read it with the
`tui_snapshot` tool.

- The dump is **user-triggered**. If `tui_snapshot` reports no file or a
  stale frame, ask the user to press `/dump` again, then retry.
- Pass `raw: true` to keep ANSI colour codes for debugging styling bugs.
- Pass `tail: N` for just the last N lines of a long transcript.

## Trace and log locations

| Data | Location | Query with |
|------|----------|-----------|
| Session traces (spans + tool calls + messages) | `$AGENTM_HOME/observability/<session_id>.jsonl` | `agentm trace` subcommands |
| Harness operational logs (loguru) | Same JSONL file (interleaved as OTLP log records) when `OTEL_EXPORTER_OTLP_ENDPOINT` is set; otherwise **stderr only** | `agentm trace logs --latest` |
| ClickHouse (when configured) | `AGENTM_CLICKHOUSE_URL` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `agentm trace --session <id>` |

For the full command reference and composition patterns, load the
`trace-analysis` skill.

## Diagnostic workflow

1. **Reproduce-from-evidence**: `agentm trace tools --latest` to see
   exactly what you did, `tui_snapshot` to see what the user saw.
2. **Check harness logs**: `agentm trace logs --latest --format ndjson | jq 'select(.severityText=="WARN" or .severityText=="ERROR")'`
3. **Form a hypothesis** grounded in that evidence.
4. If the bug is in a past session: `agentm trace index` → find the
   session → drill in with `--session <id>`.
5. Only then propose or apply a fix.
