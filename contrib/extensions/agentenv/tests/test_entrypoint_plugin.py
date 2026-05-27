"""The agentm-agentenv plugin is discovered BY NAME through the entry-point
groups — no source checkout, no AGENTM_PROJECT_ROOT, no path. This is the
contract the workbuddy K8s image relies on (`pip install agentm[agent-env]`
then `--scenario agent_env_repo`).

Runs in the member's own venv (the workspace install registers the entry
points); the root suite ignores this directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def test_atom_registered_via_entrypoint() -> None:
    from agentm.extensions.discover import discover_entrypoint_atoms

    atoms = discover_entrypoint_atoms()
    assert "operations_agent_env" in atoms, atoms
    entry = atoms["operations_agent_env"]
    # Real, installed dotted module — not a synthetic _agentm_contrib__ name.
    assert entry.module_path == "agentm_agentenv.operations_agent_env"
    assert entry.manifest.name == "operations_agent_env"


def test_scenario_resolves_by_name_from_unrelated_cwd(tmp_path: Path) -> None:
    from agentm.extensions.loader import load_scenario

    # cwd deliberately unrelated to any AgentM checkout: name resolution must
    # come purely from the installed entry point, not a path walk.
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        exts = load_scenario("agent_env_repo")
    finally:
        os.chdir(cwd)

    modules = [m for m, _ in exts]
    assert "agentm_agentenv.operations_agent_env" in modules
    # tool atoms still come from the builtin set
    assert "agentm.extensions.builtin.tool_bash" in modules


def test_operations_atom_carries_sync_cwd_default() -> None:
    # The agent_env_repo scenario must enable sync_cwd; the plain agent_env
    # scenario (RL/llmharness) leaves it off. Assert the scenario config.
    from agentm.extensions.loader import load_scenario

    cfg = {m: c for m, c in load_scenario("agent_env_repo")}
    assert cfg["agentm_agentenv.operations_agent_env"].get("sync_cwd") is True
