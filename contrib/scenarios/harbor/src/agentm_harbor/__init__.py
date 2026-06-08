"""Harbor agent adapter for AgentM.

Usage::

    harbor trial start -p <task> \\
        --agent-import-path agentm_harbor:AgentMAgent \\
        -m doubao \\
        --ae AGENTM_API_KEY=... \\
        --ae AGENTM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

Install into Harbor's tool environment::

    uv tool install harbor --with agentm-harbor
"""

from agentm_harbor.agent import AgentMAgent

__all__ = ["AgentMAgent"]
