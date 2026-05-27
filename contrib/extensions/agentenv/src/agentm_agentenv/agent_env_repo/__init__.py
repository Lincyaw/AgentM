"""Resource package for the ``agent_env_repo`` scenario.

Holds ``manifest.yaml`` so AgentM's ``agentm.scenarios`` entry point can
resolve it via ``importlib.resources.files("agentm_agentenv.agent_env_repo")``.
The package directory name (``agent_env_repo``) is the scenario name, which
the loader's dir-name == scenario-name contract requires.
"""
