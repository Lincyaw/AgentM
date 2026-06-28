---
name: trace-analysis
description: Analyze agentm session traces using `agentm trace` atomic commands. Use when inspecting trajectories, token economics, tool calls, or aggregating stats across a trace tree (parent + child sessions). Compose atomic commands via shell pipes — never parse OTLP JSONL directly.
---

# trace-analysis

## Data sources

Session traces live in two backends. `agentm trace` abstracts both:

| Backend | When active | Session selector |
|---------|------------|-----------------|
| **Local JSONL** (default) | Always; files at `$AGENTM_HOME/observability/<session_id>.jsonl` | `--file <path>` or `--session <id>` (resolves to the JSONL) |
| **ClickHouse** | When `AGENTM_CLICKHOUSE_URL` or `OTEL_EXPORTER_OTLP_ENDPOINT` is set | `--session <id>` (queries ClickHouse automatically) |

`--latest` picks the most recent session file in the observability dir.
When remote trace storage is configured AND a local file exists, `--session`
prefers ClickHouse. The same subcommands and `--format ndjson` work on both.

Harness operational logs (loguru) are forwarded into the OTLP logs signal
when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. They land in the same JSONL
file interleaved with spans. Query them with `agentm trace logs`.

## Atomic commands

| Command | Returns |
|---------|---------|
| `index` | one identity row per session: `session_id, trace_id, parent_session_id, purpose, scenario` |
| `info` | session header + atom fingerprint + task_meta |
| `usage` | aggregate token economics: `turns, input_tokens, cache_read, output_tokens, total_tokens, cache_hit_rate` |
| `turns` | per-turn: `turn_index, duration_ns, tool_calls, tool_call_count, input_tokens, output_tokens, cache_read, stop_reason` |
| `tools` | per-tool-call: `tool, args, result, span_id, attributes` |
| `messages` | full conversation trajectory in message order |
| `chats` | per-LLM-call with duration |
| `spans` | generic span query (custom `--name` / `--where` / `--since`) |
| `logs` | generic log query (harness operational logs) |
| `stats` | histogram of event/span names (orientation) |

All accept `--session <id>` / `--file <path>` / `--latest` and `--format ndjson`.

## Composition patterns

### Single session
```bash
agentm trace messages --latest                        # your own trajectory
agentm trace tools --latest --format ndjson | jq '.tool'  # tool names
agentm trace tools --latest --tool bash --format ndjson   # specific tool
agentm trace usage --latest                           # token economics
agentm trace logs --latest --format ndjson | jq '.body'   # harness logs
```

### Trace tree (parent + children)
```bash
TID=<trace_id>

# List all sessions in the trace
agentm trace index --format ndjson | jq --arg t "$TID" 'select(.trace_id==$t)'

# Aggregate usage across trace
agentm trace index --format ndjson \
  | jq -r --arg t "$TID" 'select(.trace_id==$t) | .session_id' \
  | while read sid; do agentm trace usage --session "$sid" --format ndjson; done \
  | jq -s '{sessions: length, input: (map(.input_tokens)|add), output: (map(.output_tokens)|add)}'
```

### Tool-specific extraction
```bash
# All verdicts in a verifier trace
agentm trace index --format ndjson \
  | jq -r --arg t "$TID" 'select(.trace_id==$t and .purpose=="hop_worker") | .session_id' \
  | while read sid; do agentm trace tools --session "$sid" --tool submit_hop_verdict --format ndjson; done \
  | jq -c '.args | {verdict, predicate, rationale}'
```

## Analysis order

1. **Find the session** — `--latest`, workflow artifact `child_sessions[].session_id`, or `index`
2. **Cost?** `usage --session <sid>`
3. **What happened per turn?** `turns --session <sid>`
4. **Specific tool calls?** `tools --session <sid> --tool <name>`
5. **Harness logs?** `logs --session <sid>` — look for warnings/errors
6. **Raw trajectory?** `messages --session <sid>` — only when the above don't suffice
