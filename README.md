# AgentM

Hypothesis-driven multi-agent orchestration framework for Root Cause Analysis (RCA).

## Quick Start

### CLI

```bash
uv sync
uv run agentm run \
  --data-dir /path/to/observability-data \
  --incident "Service X latency spike at 2026-03-10T14:00:00"
```

### Library (for evaluation)

```python
import asyncio
import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from agentm.builder import AgentSystemBuilder
from agentm.config.loader import load_system_config, load_scenario_config
from agentm.tools.observability import set_data_directory
from agentm.tools.duckdb_sql import register_tables


async def run_rca(
    data_dir: str,
    incident: str,
    scenario_dir: str = "config/scenarios/rca_hypothesis",
    config_path: str = "config/system.yaml",
    max_steps: int = 100,
) -> dict:
    """Run an RCA investigation and return the CausalGraph result.

    Args:
        data_dir: Path to directory containing observability parquet files.
        incident: Natural language incident description.
        scenario_dir: Path to the scenario config directory.
        config_path: Path to system.yaml.
        max_steps: Maximum orchestrator steps.

    Returns:
        dict with keys: nodes, edges, root_causes, component_to_service.
    """
    project_root = Path(config_path).resolve().parent.parent
    scenario_path = Path(scenario_dir)
    if not scenario_path.is_absolute():
        scenario_path = project_root / scenario_path

    # 1. Load configs
    system_config = load_system_config(config_path)
    scenario_config = load_scenario_config(scenario_path / "scenario.yaml")

    # 2. Initialize data sources
    init_result = json.loads(set_data_directory(data_dir))
    if "error" in init_result:
        raise RuntimeError(init_result["error"])

    register_tables({
        Path(f).stem: str(Path(data_dir) / f)
        for f in init_result["files"]
        if Path(f).parent == Path(".")
    })

    # 3. Resolve prompt paths (relative to scenario dir)
    if scenario_config.orchestrator.prompts:
        scenario_config.orchestrator.prompts = {
            k: str(scenario_path / v)
            for k, v in scenario_config.orchestrator.prompts.items()
        }
    if scenario_config.orchestrator.output is not None:
        scenario_config.orchestrator.output.prompt = str(
            scenario_path / scenario_config.orchestrator.output.prompt
        )
    for agent_config in scenario_config.agents.values():
        if agent_config.task_type_prompts:
            agent_config.task_type_prompts = {
                k: str(scenario_path / v)
                for k, v in agent_config.task_type_prompts.items()
            }

    # 4. Build agent system
    system = AgentSystemBuilder.build(
        system_type="hypothesis_driven",
        scenario_config=scenario_config,
        system_config=system_config,
    )

    # 5. Execute
    initial_state = {"messages": [HumanMessage(content=incident)]}
    result = None

    async with system:
        step = 0
        async for event in system.stream(initial_state):
            step += 1
            # Extract structured output (CausalGraph)
            for node_name, node_data in event.items():
                if node_name in ("generate_structured_response", "synthesize"):
                    sr = node_data.get("structured_response") if isinstance(node_data, dict) else None
                    if sr is not None:
                        result = sr.model_dump() if hasattr(sr, "model_dump") else sr
            if step >= max_steps:
                break

    return result or {}


# --- Usage ---
if __name__ == "__main__":
    causal_graph = asyncio.run(run_rca(
        data_dir="/path/to/parquet-data",
        incident="Service payment-gateway P99 latency exceeded 5s SLO",
    ))
    print(json.dumps(causal_graph, indent=2, ensure_ascii=False))
```

### Return Value

`run_rca()` returns a `CausalGraph` dict:

```json
{
  "nodes": [
    {"component": "mysql-primary", "state": ["slow_queries"], "timestamp": "1741600800000000000"},
    {"component": "order-service", "state": ["high_latency"], "timestamp": "1741600830000000000"}
  ],
  "edges": [
    {"source": "mysql-primary", "target": "order-service"}
  ],
  "root_causes": [
    {"component": "mysql-primary", "state": ["slow_queries"], "timestamp": "1741600800000000000"}
  ],
  "component_to_service": [
    {"component_name": "mysql-pod-0", "service_name": "mysql-primary"}
  ]
}
```

### Prerequisites: Populate the Dataset DB

`agentm eval` reads from the rcabench-platform DB (`data` table). The table must contain `DatasetSample` rows with `dataset="RCABench"` before you can run eval. Each row's `meta` field must include `source_data_dir` pointing to the local datapack directory (the folder with parquet files + `causal_graph.json`).

Use the rcabench-platform tooling to import your dataset:

```bash
# In the rcabench-platform repo
LLM_EVAL_DB_URL=sqlite:///path/to/eval.db \
  python cli/dataset_transform/make_rcabench.py run
```

Or insert samples directly in Python:

```python
import os
os.environ["LLM_EVAL_DB_URL"] = "sqlite:///eval.db"

from rcabench_platform.v3.sdk.llm_eval.db import DatasetSample
from rcabench_platform.v3.sdk.llm_eval.utils import SQLModelUtils

samples = [
    DatasetSample(
        dataset="RCABench",
        index=1,
        source="my-datapack-001",
        question="",          # filled by preprocess
        answer="mysql-primary",
        meta={"source_data_dir": "/path/to/my-datapack-001"},
    ),
    # ... more samples
]

with SQLModelUtils.create_session() as session:
    session.add_all(samples)
    session.commit()
```

### Run Full Pipeline

```bash
uv run agentm eval config/eval/example.yaml \
  --scenario config/scenarios/rca_hypothesis \
  --system-config config/system.yaml
```

### Run Specific Phases

```bash
# Re-judge already-rolled-out results (skip rollout)
uv run agentm eval config/eval/test.yaml --judge-only

# Show stats from judged results only
uv run agentm eval config/eval/test.yaml --stat-only
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--scenario` | `config/scenarios/rca_hypothesis` | Scenario directory |
| `--system-config` | `config/system.yaml` | System config YAML |
| `--exp-id` | _(from config)_ | Override experiment ID |
| `--judge-only` | false | Skip rollout, run judge + stat |
| `--stat-only` | false | Run stat only |
| `--max-steps` | 100 | Max orchestrator steps per sample |
| `--timeout` | 600.0 | Per-sample timeout in seconds (0 = none) |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENTM_API_KEY` | Yes | LLM API key |
| `AGENTM_API_BASE_URL` | No | Custom API base URL (OpenAI-compatible) |
| `AGENTM_ORCHESTRATOR_MODEL` | No | Override orchestrator model name |
| `AGENTM_WORKER_MODEL` | No | Override worker model name |
| `LLM_EVAL_DB_URL` | No | Override eval DB URL (auto-set from config) |
