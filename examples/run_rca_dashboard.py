"""RCA investigation with live Dashboard.

Starts both the FastAPI dashboard server and the RCA investigation
concurrently. Open http://localhost:8765 to view the dashboard.

Usage:
    uv run python examples/run_rca_dashboard.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import uvicorn
from langchain_core.messages import HumanMessage

from agentm.builder import AgentSystemBuilder
from agentm.config.loader import load_scenario_config, load_system_config
from agentm.server.app import broadcast_event, create_dashboard_app
import agentm.tools.observability as obs_tools

# ---------------------------------------------------------------------------
# Configuration — override via env vars
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get(
    "AGENTM_DATA_DIR",
    "/home/nn/workspace/RCAgentEval/worktrees/refactor/eval-data/gpt-test-3.7/data_1069895e",
)

INCIDENT_DESCRIPTION = os.environ.get(
    "AGENTM_INCIDENT",
    (
        "The following API endpoints are experiencing possible SLO violations "
        "and need investigation:\n"
        "- HTTP POST http://ts-ui-dashboard:8080/api/v1/travelplanservice/travelPlan/minStation\n\n"
        "Please investigate the root cause of these SLO violations."
    ),
)

DASHBOARD_PORT = int(os.environ.get("AGENTM_DASHBOARD_PORT", "8765"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = PROJECT_ROOT / "config" / "scenarios" / "rca_hypothesis"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def run_with_dashboard() -> None:
    """Run the RCA investigation with a live dashboard."""

    # Apply env-var overrides for API key / base_url / model
    if api_key := os.environ.get("AGENTM_API_KEY"):
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url := os.environ.get("AGENTM_API_BASE_URL"):
        os.environ.setdefault("OPENAI_BASE_URL", base_url)

    # Load configs
    system_config = load_system_config(PROJECT_ROOT / "config" / "system.yaml")
    scenario_config = load_scenario_config(SCENARIO_DIR / "scenario.yaml")

    # Override models from env if set
    if model := os.environ.get("AGENTM_ORCHESTRATOR_MODEL"):
        scenario_config.orchestrator.model = model
    if model := os.environ.get("AGENTM_WORKER_MODEL"):
        scenario_config.agents["worker"].model = model

    # Initialize observability data
    print(f"\n{'='*70}")
    print("AgentM — RCA Investigation + Dashboard")
    print(f"{'='*70}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Orchestrator:   {scenario_config.orchestrator.model}")
    print(f"Worker:         {scenario_config.agents['worker'].model}")
    print(f"Dashboard:      http://localhost:{DASHBOARD_PORT}")
    print(f"{'='*70}\n")

    result = obs_tools.set_data_directory(DATA_DIR)
    init_info = json.loads(result)
    if "error" in init_info:
        print(f"ERROR: {init_info['error']}")
        sys.exit(1)
    print(f"Data initialized: {len(init_info['files'])} parquet files\n")

    # Build agent system
    system = AgentSystemBuilder.build(
        system_type="hypothesis_driven",
        scenario_config=scenario_config,
        system_config=system_config,
        scenario_dir=SCENARIO_DIR,
    )

    # Create dashboard app
    app = create_dashboard_app(
        graph=system.graph,
        scenario_config=system.scenario_config,
        task_manager=system.task_manager,
    )

    # Wire WebSocket broadcast to TaskManager
    if system.task_manager is not None:
        system.task_manager._broadcast_callback = broadcast_event

    # Start uvicorn server in background
    config = uvicorn.Config(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait a moment for the server to start
    await asyncio.sleep(0.5)
    print(f"Dashboard ready at http://localhost:{DASHBOARD_PORT}\n")
    print(f"{'─'*70}\n")

    # Stream investigation
    initial_state = {"messages": [HumanMessage(content=INCIDENT_DESCRIPTION)]}
    task_id = f"rca-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Starting investigation: {task_id}")

    step = 0
    try:
        async for event in system.stream(initial_state, on_event=broadcast_event):
            step += 1

            for node_name, node_data in event.items():
                if node_name == "__interrupt__":
                    continue

                messages = node_data.get("messages", [])
                for msg in messages:
                    role = getattr(msg, "type", "unknown")
                    content = getattr(msg, "content", "")

                    if role == "ai" and content:
                        print(f"\n[Orchestrator step {step}]")
                        print(content[:2000])

                        tool_calls = getattr(msg, "tool_calls", [])
                        for tc in tool_calls:
                            args_str = json.dumps(tc["args"], ensure_ascii=False)
                            print(f"\n  -> {tc['name']}({args_str[:300]})")

                    elif role == "tool":
                        tool_name = getattr(msg, "name", "?")
                        if content:
                            print(f"\n  <- {tool_name}: {content[:500]}")

            if step > 100:
                print("\n[!] Reached step limit (100), stopping.")
                break

    except KeyboardInterrupt:
        print("\n\n[!] Investigation interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Investigation completed after {step} streaming steps.")
    print(f"Dashboard still running at http://localhost:{DASHBOARD_PORT}")
    print("Press Ctrl+C to stop.")
    print(f"{'='*70}")

    # Keep server alive for post-hoc inspection
    try:
        await server_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(run_with_dashboard())
