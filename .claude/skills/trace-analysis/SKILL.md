---
name: trace-analysis
description: >
  Analyze agentm session traces using `agentm trace` commands. Use when
  inspecting trajectories, token economics, tool calls, or reviewing
  session behavior. Compose commands via shell pipes — never parse
  trajectory JSONL directly.
---

# trace-analysis

## Data sources

Session traces are stored in trajectory stores. `agentm trace` abstracts
the backend:

| Backend | When active | Session selector |
|---------|------------|-----------------|
| **Local JSONL** (default) | When `.agentm/trajectory/` exists or `AGENTM_TRAJECTORY_DIR` is set | `--session <id>` or `--latest` |
| **Postgres** | When `AGENTM_TRAJECTORY_DSN` is set | `--session <id>` or `--latest` |

`--latest` picks the most recent session by creation time.

## Commands

| Command | Returns |
|---------|---------|
| `sessions` | List sessions: `id, parent_session_id, root_session_id, purpose, cwd` |
| `turns` | Per-turn summary: `turn_index, trigger_source, rounds, tool_calls, tool_call_count, tool_error_count, input_tokens, output_tokens, cache_read, model, cause` |
| `messages` | Full conversation in message order (user, assistant, tool_result, error) |
| `usage` | Aggregate token economics: `turns, input_tokens, cache_read, cache_write, non_cached_input, output_tokens, total_tokens, cache_hit_rate` |
| `tools` | Per-tool-call: `turn_index, round_index, tool, args, is_error, result` |
| `view` | Interactive terminal viewer with turn navigation and expand/collapse |

All accept `--session <id>` or `--latest` and `--format ndjson|text`.
Text is default for TTY, ndjson for pipes.

## Composition patterns

### Single session
```bash
agentm trace messages --latest                            # your own trajectory
agentm trace tools --latest --format ndjson | jq '.tool'  # tool names
agentm trace tools --latest --tool bash --format ndjson   # specific tool
agentm trace usage --latest                               # token economics
```

### Session tree (parent + children)
```bash
# List all sessions
agentm trace sessions --format ndjson

# Filter by parent
agentm trace sessions --parent <parent_id> --format ndjson

# Aggregate usage across children
agentm trace sessions --parent <parent_id> --format ndjson \
  | jq -r '.id' \
  | while read sid; do agentm trace usage --session "$sid" --format ndjson; done \
  | jq -s '{sessions: length, input: (map(.input_tokens)|add), output: (map(.output_tokens)|add)}'
```

### Tool-specific extraction
```bash
# All bash commands in a session
agentm trace tools --session <sid> --tool bash --format ndjson \
  | jq -r '.args.cmd'

# Tool error rate
agentm trace tools --session <sid> --format ndjson \
  | jq -s '{total: length, errors: [.[] | select(.is_error)] | length}'
```

## Analysis order

1. **Find the session** — `--latest`, or `sessions --parent <id>`
2. **Token economics** — `usage --session <sid>` (cache hit rate, total spend)
3. **Turn structure** — `turns --session <sid>` (how many rounds, which tools, error count)
4. **Specific tools** — `tools --session <sid> --tool <name>`
5. **Raw trajectory** — `messages --session <sid>` — only when the above don't suffice
6. **Interactive view** — `view --session <sid>` when you need to navigate turn by turn

## Attribution playbook

The purpose of trace analysis is to attribute an observed effect (a failed
task, an outlier session, a suspicious number) to an intervenable cause.

- **Aggregates are indexes, not conclusions.** A number only tells you
  *that* something differs; the *why* lives at turn level.
- **Decompose before narrating.** Pick the anomalous dimension and split
  it mechanically first; only then read the trajectory.

### Loop detection

Group tool calls by args, sort by repeat count:

```bash
agentm trace tools --session <sid> --format ndjson \
  | jq -s 'group_by([.tool, .args]) | map({tool: .[0].tool, args: .[0].args, n: length}) | sort_by(-.n) | .[:5]'
```

A high repeat count on identical args is the strongest single signal:
- Same args, same error → framework/tooling issue (error message taught the model nothing)
- New approach, no progress → model issue (spinning without learning)
- Steady progress → task is inherently expensive
