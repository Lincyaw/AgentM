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

### `analyze` — Single Trajectory Analysis

Analyze completed RCA trajectories and extract reusable knowledge.
Accepts both JSONL (event format) and JSON (eval DB export with `_eval_meta`).

```bash
# Failed trajectory — analyze what went wrong
agentm analyze trajectories/rca-20260312-091500.jsonl \
    --task "failure: missed ts-order-service, anchored on ts-preserve-service"

# Eval DB export (JSON with _eval_meta)
agentm analyze eval-trajectories/agentm-v11_7901_incorrect.json \
    --task "failure: ground truth is ts-basic-service,ts-price-service"

# Multiple files
agentm analyze trajectories/rca-*.jsonl \
    --task "2/3 succeeded, 1 failed on cascade identification"

# With live dashboard
agentm analyze trajectories/rca-20260311-162834.jsonl \
    --task "success: root cause identified" --dashboard --port 8765
```

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | _(required)_ | Analysis task with evaluation feedback |
| `--scenario` | `config/scenarios/trajectory_analysis` | Scenario directory |
| `--config` | `config/system.yaml` | System config YAML |
| `--max-steps` | 60 | Maximum orchestrator steps |
| `--debug` | false | Enable rich debug terminal UI |
| `--verbose` | false | Extra detail in output |
| `--dashboard` | false | Start web dashboard for real-time monitoring |
| `--port` | 8765 | Dashboard server port |
| `--dashboard-host` | `127.0.0.1` | Dashboard server bind address |

### `analyze-batch` — Batch Trajectory Analysis

Batch analyze evaluation trajectories driven by a YAML config file.
Groups N trajectories per analysis run so the agent can identify cross-case patterns.

**Data flow:**

```
PostgreSQL (evaluation_data)               Local directory
    |                                          |
    |  source.type: "database"                 |  source.type: "directory"
    v                                          v
  export to JSON ──> eval-trajectories/ <──  read directly
                           |
                     group into batches (batch.size)
                           |
                     trajectory_analysis agent
                     (one run per batch)
```

```bash
# Error analysis on failed cases (default config)
agentm analyze-batch config/batch/default.yaml

# Feature extraction — include correct cases too
agentm analyze-batch config/batch/default.yaml --filter all

# From database — change source.type to "database" in config, then:
agentm analyze-batch config/batch/default.yaml --exp-id agentm-v12 --limit 30

# Ad-hoc overrides
agentm analyze-batch config/batch/default.yaml --batch-size 5 --verbose
```

| Option | Default | Description |
|--------|---------|-------------|
| `CONFIG_FILE` | _(required)_ | Path to batch config YAML |
| `--limit` | _(from config)_ | Override max total cases |
| `--batch-size` | _(from config)_ | Override trajectories per batch |
| `--concurrency` | _(from config)_ | Override parallel batch runs |
| `--exp-id` | _(from config)_ | Override experiment ID filter |
| `--filter` | _(from config)_ | Override correctness filter (incorrect/correct/all) |
| `--verbose` | false | Extra detail in output |
| `--dashboard` | false | Start web dashboard |
| `--port` | 8765 | Dashboard server port |

**Batch config structure** (`config/batch/*.yaml`):

```yaml
source:
  type: "directory"              # "directory" or "database"
  directory: "./eval-trajectories"
  filter: "incorrect"            # incorrect | correct | all
  exp_id: "agentm-v11"          # optional experiment filter
  limit: 50                     # optional case cap

batch:
  size: 10                       # trajectories per analysis run
  concurrency: 1                 # parallel batch runs
  max_steps: 60                  # max orchestrator steps per batch

task:
  goals:                         # custom analysis goals (optional)
    - "Classify each case using Outcome Classification taxonomy"
    - "Identify cross-case error patterns"

scenario: "config/scenarios/trajectory_analysis"
system_config: "config/system.yaml"

output:
  export_dir: "./eval-trajectories"
  verbose: false
  dashboard: false
```

Default config: `config/batch/default.yaml`. Copy and modify for different experiments or analysis goals.

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

## Programmatic API (Eval Integration)

For batch evaluation via rcabench-platform, use `AgentMAgent`:

```python
from agentm.agents.eval_agent import AgentMAgent

agent = AgentMAgent(
    scenario_dir="config/scenarios/rca_hypothesis",
    config_path="config/system.yaml",
    max_steps=100,
    timeout=600,
)
ok, fail = await benchmark.rollout(agent)
```

## Data Pipeline

```
1. Run eval         rcabench rollout --agent agentm ...
                         |
                         v
2. Judge             rcabench judge --exp-id agentm-v11
                         |
                         v
3. Export             python scripts/export_eval_trajectories.py \
                         --exp-id agentm-v11 --correct false
                         |
                         v
4. Batch analyze     agentm analyze-batch config/batch/default.yaml
                         |
                         v
5. Knowledge         knowledge/vault/ (accumulated analysis entries)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENTM_API_KEY` | Yes | LLM API key |
| `AGENTM_API_BASE_URL` | No | Custom API base URL (OpenAI-compatible) |
| `AGENTM_ORCHESTRATOR_MODEL` | No | Override orchestrator model name |
| `AGENTM_WORKER_MODEL` | No | Override worker model name |
| `AGENTM_LOG_LEVEL` | No | Log level for agentm loggers (default: `INFO`) |
| `LLM_EVAL_DB_URL` | For DB mode | PostgreSQL connection URL for eval database |

## Configuration

AgentM uses layered config files:

- **`config/system.yaml`** — model registry, storage backend, debug settings
- **`config/scenarios/<name>/scenario.yaml`** — scenario-specific orchestrator/agent config
- **`config/batch/<name>.yaml`** — batch analysis config (data source, grouping, goals)

Available scenarios:

| Scenario | Description |
|----------|-------------|
| `rca_hypothesis` | RCA investigation with hypothesis-driven reasoning |
| `trajectory_analysis` | Extract knowledge from completed trajectories |
| `memory_extraction` | Extract and organize knowledge from trajectory data |
| `general_purpose` | General-purpose agent orchestration |
