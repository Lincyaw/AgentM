# AgentM

Hypothesis-driven multi-agent orchestration framework for Root Cause Analysis (RCA).

## Quick Start

```bash
# Install dependencies
uv sync

# Set required environment variables
export AGENTM_API_KEY="your-api-key"
export AGENTM_API_BASE_URL="https://your-api-endpoint"  # optional, OpenAI-compatible
```

## CLI Commands

### `analyze` — Trajectory Analysis

Analyze completed RCA trajectories and extract reusable knowledge.

```bash
# Successful trajectory — extract what went right
agentm analyze trajectories/rca-20260311-162834.jsonl \
    --feedback "success: correctly identified mysql connection pool as root cause"

# Failed trajectory — analyze what went wrong
agentm analyze trajectories/rca-20260312-091500.jsonl \
    --feedback "failure: missed ts-order-service, anchored on ts-preserve-service"

# Multiple files with custom task
agentm analyze trajectories/rca-*.jsonl \
    --task "Focus on database failure patterns" \
    --feedback "2/3 succeeded, 1 failed on cascade identification"

# With live dashboard
agentm analyze trajectories/rca-20260311-162834.jsonl --dashboard --port 8765
```

| Option | Default | Description |
|--------|---------|-------------|
| `--feedback` | _(empty)_ | Evaluation feedback (success/failure + details) |
| `--task` | _(auto-generated)_ | Custom analysis task description |
| `--scenario` | `config/scenarios/trajectory_analysis` | Scenario directory |
| `--config` | `config/system.yaml` | System config YAML |
| `--max-steps` | 60 | Maximum orchestrator steps |
| `--debug` | false | Enable rich debug terminal UI |
| `--verbose` | false | Extra detail in output |
| `--dashboard` | false | Start web dashboard for real-time monitoring |
| `--port` | 8765 | Dashboard server port |
| `--dashboard-host` | `127.0.0.1` | Dashboard server bind address |

### `resume` — Resume Investigation

Resume an interrupted RCA investigation from a trajectory checkpoint.

```bash
# Interactive checkpoint selection
agentm resume trajectories/rca-20260311-162834.jsonl \
    --data-dir /path/to/observability-data

# List available checkpoints
agentm resume trajectories/rca-20260311-162834.jsonl --list

# Resume from a specific checkpoint
agentm resume trajectories/rca-20260311-162834.jsonl \
    --data-dir /path/to/observability-data \
    --checkpoint <checkpoint-id>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | _(empty)_ | Observability data directory |
| `--scenario` | `config/scenarios/rca_hypothesis` | Scenario directory |
| `--config` | `config/system.yaml` | System config YAML |
| `--checkpoint` | _(interactive)_ | Checkpoint ID to restore directly |
| `--list` | false | List checkpoints without executing |
| `--dashboard` | false | Start web dashboard after resuming |
| `--port` | 8765 | Dashboard server port |
| `--dashboard-host` | `127.0.0.1` | Dashboard server bind address |
| `--verbose` | false | Extra detail in output |

### `debug` — Trajectory Inspection

Inspect and analyze trajectory JSONL files offline.

```bash
# Show summary statistics
agentm debug trajectories/rca-20260311-162834.jsonl --summary

# Show tool call timeline
agentm debug trajectories/rca-20260311-162834.jsonl --timeline

# Filter by agent or event type
agentm debug trajectories/rca-20260311-162834.jsonl --filter-agent orchestrator
agentm debug trajectories/rca-20260311-162834.jsonl --filter-type tool_call
```

| Option | Default | Description |
|--------|---------|-------------|
| `--summary` | false | Print summary statistics |
| `--timeline` | false | Show tool call timeline |
| `--filter-agent` | _(none)_ | Filter by agent path prefix |
| `--filter-type` | _(none)_ | Filter by event_type |

### `export-result` — Export Single Trajectory

Export case_dir + ground_truth + final outputs from one trajectory.

```bash
agentm export-result trajectories/rca-20260311-162834.jsonl
agentm export-result trajectories/rca-20260311-162834.jsonl -o result.json
```

### `export-batch` — Batch Export

Batch export from a directory of trajectories.

```bash
agentm export-batch trajectories/
agentm export-batch trajectories/ -p "rca-*.jsonl" -o exported/
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENTM_API_KEY` | Yes | LLM API key |
| `AGENTM_API_BASE_URL` | No | Custom API base URL (OpenAI-compatible) |
| `AGENTM_ORCHESTRATOR_MODEL` | No | Override orchestrator model name |
| `AGENTM_WORKER_MODEL` | No | Override worker model name |
| `AGENTM_LOG_LEVEL` | No | Log level for agentm loggers (default: `INFO`) |

## Configuration

AgentM uses two config files:

- **`config/system.yaml`** — model registry, storage backend, debug settings
- **`config/scenarios/<name>/scenario.yaml`** — scenario-specific orchestrator/agent config

Available scenarios:

| Scenario | Description |
|----------|-------------|
| `rca_hypothesis` | RCA investigation with hypothesis-driven reasoning |
| `trajectory_analysis` | Extract knowledge from completed trajectories |
| `general_purpose` | General-purpose agent orchestration |
