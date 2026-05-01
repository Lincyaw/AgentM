# AgentM

A generic multi-agent orchestration SDK built on LangGraph. Supports custom state types, pluggable reasoning strategies, and middleware pipelines for diverse agent workflows.

## Quick Start

```bash
# Install dependencies
uv sync

# Authenticate with a registry-backed provider
uv run agentm auth providers
uv run agentm auth login anthropic

# Or use provider-specific environment variables instead of OAuth
export ANTHROPIC_API_KEY="your-anthropic-key"
# export OPENAI_API_KEY="your-openai-key"
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
| `--dashboard-host` | `0.0.0.0` | Dashboard server bind address (binds all interfaces by default; restrict to `127.0.0.1` on untrusted networks) |

### `judge` — Trajectory Judging

Judge evaluation trajectories using the trajectory_judger scenario. Driven by a YAML config file.

```bash
# Judge cases from default config
agentm judge config/batch/default.yaml

# Override filters
agentm judge config/batch/default.yaml --filter all --limit 30

# Filter by experiment
agentm judge config/batch/default.yaml --exp-id agentm-v12
```

**Judge config structure** (`config/batch/*.yaml`):

```yaml
source:
  type: "directory"              # "directory" or "database"
  directory: "./eval-trajectories"
  filter: "incorrect"            # incorrect | correct | all
  # exp_id: "agentm-v11"        # optional experiment filter
  # limit: 50                   # optional case cap
  data_base_dir: "${RCABENCH_DATA_DIR}"

concurrency: 10                  # parallel judge workers
scenario: "config/scenarios/trajectory_analysis"
system_config: "config/system.yaml"
```

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
| `ANTHROPIC_API_KEY` | No | Anthropic API key fallback when OAuth is not configured |
| `ANTHROPIC_OAUTH_TOKEN` | No | Explicit Anthropic OAuth access token override |
| `OPENAI_API_KEY` | No | OpenAI API key for the `openai` registry provider |
| `OPENAI_BASE_URL` | No | Custom OpenAI-compatible base URL for the `openai` provider |
| `AGENTM_AUTH_PATH` | No | Override persistent auth-storage path (default: `~/.agentm/auth.json`) |
| `AGENTM_ORCHESTRATOR_MODEL` | No | Override orchestrator model name (default: from config) |
| `AGENTM_WORKER_MODEL` | No | Override worker model name (default: from config) |
| `AGENTM_LOG_LEVEL` | No | Log level for agentm loggers (default: `INFO`) |
| `LLM_EVAL_DB_URL` | For DB mode | PostgreSQL connection URL for eval database |

## Configuration

AgentM uses layered config files:

- **`config/system.yaml`** — system-wide storage/debug defaults plus legacy examples
- **`config/scenarios/<name>/scenario.yaml`** — scenario-specific orchestrator/agent config, including provider registry keys
- **`config/batch/<name>.yaml`** — batch analysis config (data source, grouping, goals)

Available scenarios:

| Scenario | Description |
|----------|-------------|
| `rca_hypothesis` | RCA investigation with hypothesis-driven reasoning |
| `trajectory_analysis` | Extract knowledge from completed trajectories |
| `memory_extraction` | Extract and organize knowledge from trajectory data |
| `general_purpose` | General-purpose agent orchestration |
