---
name: trace-analysis
description: Analyze agentm session traces using `agentm trace` atomic commands. Use when inspecting trajectories, token economics, tool calls, or aggregating stats across a trace tree (parent + child sessions). Compose atomic commands via shell pipes — never parse OTLP JSONL directly.
---

# trace-analysis

## Principle

`agentm trace` provides atomic, single-file query commands. Each returns
structured data (`--format ndjson`) suitable for piping through `jq` and
shell loops. **Never parse OTLP JSONL directly** — the trace CLI handles
format evolution; raw parsing breaks when the schema changes.

A logical trace spans many JSONL files (one root session + N spawned
children). `index` is the **selection layer** (directory-granular); every
other command stays single-file. Composition: `index` selects files →
per-file command extracts data → `jq` / shell aggregates.

## Atomic commands

| Command | Returns |
|---------|---------|
| `index` | one identity row per session: `path, trace_id, session_id, parent_session_id, purpose, scenario, records` |
| `info` | session header + atom fingerprint + task_meta |
| `usage` | aggregate token economics: `turns, input_tokens, cache_read, output_tokens, total_tokens, cache_hit_rate` |
| `turns` | per-turn: `turn_index, duration_ns, tool_calls, tool_call_count, input_tokens, output_tokens, cache_read, stop_reason` |
| `tools` | per-tool-call: `tool, args, result, span_id, attributes` |
| `messages` | full conversation trajectory in message order |
| `chats` | per-LLM-call with duration |
| `stats` | histogram of event/span names (orientation) |

All accept `--file <path>` / `--session <id>` / `--latest` and `--format ndjson`.

## Composition patterns

### Single session
```bash
# Token economics
agentm trace usage --file <f>

# Per-turn timeline
agentm trace turns --file <f> --format ndjson | jq '{turn: .turn_index, in: .input_tokens, out: .output_tokens, dur_s: (.duration_ns/1e9)}'

# Find a specific tool call
agentm trace tools --file <f> --tool submit_final_report --format ndjson | jq '.args'
```

### Trace tree (parent + children)
```bash
TID=<trace_id>

# List all sessions in the trace
agentm trace index --format ndjson | jq -r --arg t "$TID" 'select(.trace_id==$t) | .path'

# Aggregate usage across trace
agentm trace index --format ndjson \
  | jq -r --arg t "$TID" 'select(.trace_id==$t) | .path' \
  | while read f; do agentm trace usage --file "$f" --format ndjson; done \
  | jq -s '{sessions: length, input: (map(.input_tokens)|add), output: (map(.output_tokens)|add)}'

# Breakdown by purpose
agentm trace index --format ndjson \
  | jq -r --arg t "$TID" 'select(.trace_id==$t) | "\(.purpose)\t\(.path)"' \
  | while IFS=$'\t' read purpose path; do
      echo "{\"purpose\":\"$purpose\",$(agentm trace usage --file "$path" --format ndjson | sed 's/^{//')}"
    done \
  | jq -s 'group_by(.purpose) | map({purpose: .[0].purpose, sessions: length, input: (map(.input_tokens)|add), output: (map(.output_tokens)|add)})'
```

### Tool-specific extraction across a trace
```bash
# All auditor verdicts in a trace
agentm trace index --format ndjson \
  | jq -r --arg t "$TID" 'select(.trace_id==$t and .purpose=="cognitive_audit_auditor_offline") | .path' \
  | while read f; do agentm trace tools --file "$f" --tool submit_verdict --format ndjson; done \
  | jq -c '.args | {fired: .surface_reminder, reminder: .reminder_text[:80]}'
```

## Analysis order

When analyzing an experiment:

1. **Was it recorded?** `index` → filter by trace_id/purpose
2. **How much did it cost?** `usage` per file, aggregate via pipe
3. **What happened per turn?** `turns` for the timeline
4. **What specific calls?** `tools --tool <name>` for tool-level detail
5. **Raw trajectory?** `messages` only when the above don't suffice

## Key invariant

Child sessions (extractor, auditor) share the parent's `trace_id`, so
`index | select(.trace_id==...)` recovers the full trace tree. If children
appear orphaned, check that `StandaloneChildRunner` received the parent's
session_id as `trace_id` (fixed in `658cd134`).
