---
name: trace-analysis
description: Analyze agentm session traces using `agentm trace` atomic commands. Use when inspecting trajectories, token economics, tool calls, or aggregating stats across a trace tree (parent + child sessions). Compose atomic commands via shell pipes — never parse OTLP JSONL directly.
---

# trace-analysis

## Principle

`agentm trace` queries session traces from **ClickHouse** (default
backend). Each subcommand returns structured data (`--format ndjson`)
suitable for piping through `jq`. **Never parse OTLP JSONL or artifact
files directly** — always use `agentm trace`.

A logical trace spans multiple sessions (one root + N spawned children).
`index` lists all sessions; per-session commands query by `--session <id>`.

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
| `logs` | generic log query |
| `stats` | histogram of event/span names (orientation) |

All accept `--session <id>` and `--format ndjson`.

## Composition patterns

### Single session
```bash
# Full trajectory
agentm trace messages --session <sid>

# Tool calls
agentm trace tools --session <sid> --format ndjson | jq '.tool'

# Specific tool
agentm trace tools --session <sid> --tool submit_hop_verdict --format ndjson | jq '.args'

# Token economics
agentm trace usage --session <sid>
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

### Tool-specific extraction across a trace
```bash
# All hop verdicts in a trace
agentm trace index --format ndjson \
  | jq -r --arg t "$TID" 'select(.trace_id==$t and .scenario=="verifier/hop") | .session_id' \
  | while read sid; do agentm trace tools --session "$sid" --tool submit_hop_verdict --format ndjson; done \
  | jq -c '.args | {verdict, predicate, rationale}'
```

## Analysis order

1. **Find the session** — from workflow delivery artifact (`child_sessions[].session_id`) or `index`
2. **How much did it cost?** `usage --session <sid>`
3. **What happened per turn?** `turns --session <sid>`
4. **What specific calls?** `tools --session <sid> --tool <name>`
5. **Raw trajectory?** `messages --session <sid>` only when the above don't suffice
