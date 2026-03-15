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

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENTM_API_KEY` | Yes | LLM API key |
| `AGENTM_API_BASE_URL` | No | Custom API base URL (OpenAI-compatible) |
| `AGENTM_ORCHESTRATOR_MODEL` | No | Override orchestrator model name |
| `AGENTM_WORKER_MODEL` | No | Override worker model name |

### Data Directory Structure

The `data_dir` should contain observability parquet files:

```
data_dir/
  abnormal_metrics.parquet
  normal_metrics.parquet
  abnormal_traces.parquet
  normal_traces.parquet
  abnormal_logs.parquet
  normal_logs.parquet
  ...
```

## Batch Evaluation

```python
import asyncio
import json
from pathlib import Path


async def evaluate_batch(cases_dir: str, output_path: str):
    """Run RCA on multiple incident cases and collect results."""
    results = []
    for case_dir in sorted(Path(cases_dir).iterdir()):
        if not case_dir.is_dir():
            continue

        # Each case has: data/ (parquets) + incident.txt
        incident_file = case_dir / "incident.txt"
        data_dir = case_dir / "data"
        if not incident_file.exists() or not data_dir.exists():
            continue

        incident = incident_file.read_text().strip()
        print(f"Running: {case_dir.name}")

        try:
            from readme_example import run_rca  # or inline the function
            causal_graph = await run_rca(
                data_dir=str(data_dir),
                incident=incident,
            )
            results.append({
                "case": case_dir.name,
                "status": "success",
                "causal_graph": causal_graph,
            })
        except Exception as e:
            results.append({
                "case": case_dir.name,
                "status": "error",
                "error": str(e),
            })

    Path(output_path).write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(evaluate_batch("./eval_cases", "./eval_results.json"))
```

## Architecture

```
Orchestrator (LLM + tools)
  ├── dispatch_agent → Sub-Agent (scout/verify/deep_analyze)
  │     └── query_sql / describe_tables (DuckDB over parquet)
  ├── update_hypothesis / remove_hypothesis
  ├── knowledge_search / knowledge_read
  ├── check_tasks / inject_instruction / abort_task
  └── recall_history (post-compression retrieval)
```

- **Orchestrator**: Manages hypotheses, dispatches workers, synthesizes CausalGraph
- **Sub-Agents**: Query observability data via SQL, report structured findings
- **Knowledge Store**: File-system backed store with inverted index + hybrid search
