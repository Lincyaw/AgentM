"""``agentm-agentenv`` — AgentM ⟷ ARL agent-env sandbox integration, published
as an installable plugin (entry points), not a source-checkout contrib file.

Provides:

- the ``operations_agent_env`` atom (Operations bundle backed by an ARL
  agent-env sandbox pod; supports ``sync_cwd`` host⟷pod workspace sync),
  registered via the ``agentm.atoms`` entry-point group; and
- the ``agent_env_repo`` scenario (agent_env + ``sync_cwd``), registered via
  the ``agentm.scenarios`` entry-point group.

Install with ``pip install agentm[agent-env]`` (or ``uv sync --extra
agent-env`` in the workspace). Once installed, AgentM resolves both BY NAME
from any cwd — no source checkout, no ``AGENTM_PROJECT_ROOT``, no path.
"""

from __future__ import annotations

from .operations_agent_env import MANIFEST, install

__all__ = ["MANIFEST", "install"]
