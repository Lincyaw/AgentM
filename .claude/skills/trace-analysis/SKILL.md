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
When remote trace storage is configured, `--session` tries ClickHouse first.
If ClickHouse has no matching session header and a local JSONL file exists,
it falls back to that file. The same subcommands and `--format ndjson` work
on both.

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
| `stats` | full session profile (see below) — the first stop for trajectory analysis |
| `doctor` | data-quality invariants for one session (duplication, lifecycle, turn contiguity, span parentage); exit 1 on violations |
| `scan` | cohort-baseline outlier list (turns/tokens/duration/peak-context/error-rate vs (scenario, task_class) p50/p95) — generates attribution entry points |

All accept `--session <id>` / `--file <path>` / `--latest` and `--format ndjson`.

## Session profile: `stats`

`stats` is no longer just a name histogram — it returns one JSON document
profiling the whole session:

| Field | Contents | Use for |
|---|---|---|
| `logs` / `spans` | event/span name histograms | orientation, sanity checks |
| `tools.<name>` | calls, errors, result-size (avg/max/total chars), duration (avg/p95/max ms) | which tool dominates latency or output volume |
| `turns` | total turns/tool_calls/errors, input/output/cache_read tokens, avg/max/min input | token economics, context growth |
| `session` | start/end/duration, `stop_reasons` histogram | did the agent stop naturally (`stop`) or run out of budget |
| `context_snapshots` | peak-context turn with per-source char attribution (`tool_result_by_name`, assistant, user, system) | *what fills the context window* — e.g. `read` results dominating 60% of peak context |

```bash
agentm trace stats --session <sid> | jq '{turns: .turns, stop: .session.stop_reasons}'
agentm trace stats --session <sid> | jq '.context_snapshots[0].tool_result_by_name'
```

## Data-quality checks (do these before trusting aggregates)

Run `agentm trace doctor --session <sid>` before quoting numbers from an
unfamiliar session — it audits the raw tables for record duplication,
lifecycle counts, turn_index contiguity, and span parentage, exiting 1 on
violations. `bench.py batch` runs it automatically over each run's sessions.

Readers (`stats`, `turns`, `usage`, …) deduplicate at query time, so
historical sessions from the double-export era (pre-89f20ea7, 2026-07-07)
read clean even though doctor still reports their raw duplication. When a
number drives a decision, cross-check one independent source (e.g. bench.py's
`tools=N` console line, or the score.json `session_id` join).

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

1. **Find the session** — `--latest`, workflow artifact `child_sessions[].session_id`, `index`, or a `scan` finding
2. **Profile first** — `stats --session <sid>` (tools, tokens, stop_reasons, peak context), then `doctor --session <sid>`
3. **What happened per turn?** `turns --session <sid>`
4. **Specific tool calls?** `tools --session <sid> --tool <name>`
5. **Harness logs?** `logs --session <sid>` — look for warnings/errors
6. **Raw trajectory?** `messages --session <sid>` — only when the above don't suffice

## Attribution playbook

The purpose of trace analysis is to attribute an observed effect (a failed
task, an outlier session, a suspicious number) to an intervenable cause.
Rules that hold regardless of the question:

- **Aggregates are indexes, not conclusions.** A number only tells you
  *that* something differs; the *why* lives at turn level. Never state a
  cause without citing turn/span evidence.
- **Decompose before narrating.** Pick the anomalous dimension and split it
  mechanically first; only then read the trajectory.
- **Judge against a cohort.** A lone trajectory is uninterpretable; compare
  to sibling sessions (same task_class/scenario — `scan` output gives the
  cohort context for free).

### The three decomposition knives

| Dimension | Split into | Command |
|---|---|---|
| **Wall time** | LLM latency (`chat` spans) vs tool wall (`execute_tool` spans) vs residual gap (harness overhead) | `chats` / `tools` durations vs session duration |
| **Tokens** | context composition by source: tool_result by tool, assistant, user/injections | `stats` → `context_snapshots.tool_result_by_name` |
| **Turns** | phase structure + loop detection: repeated similar tool args, edit→revert churn, same error recurring | `tools --format ndjson \| jq '.args'`, `turns` |

### Cause-class signatures

Anchor classes (conclusions stay free-text; these guide where to look):

- **Framework/tooling**: retries of the *same args after the same error*
  (the error message taught the model nothing — tool_error_messages'
  job); truncated results followed by re-reads (result-cap tuning); large
  residual gap in the time split; harness ERROR/diagnostic logs;
  compaction thrash.
- **Model**: retries that *change approach but gain no information*;
  ignoring an explicit error message; edit→revert oscillation; high output
  tokens with low action density.
- **Task inherently expensive**: wall time concentrated *inside* tool
  spans (builds, test suites); monotonic progress across turns — each
  turn advances, there is just a lot to do.

Loops are the strongest discriminator: same-args-same-error → framework;
new-approach-no-progress → model; steady progress → task.
