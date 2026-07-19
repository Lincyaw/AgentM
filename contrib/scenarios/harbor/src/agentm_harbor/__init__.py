"""Harbor external agent adapter for AgentM.

AgentM runs on the host; tool calls (bash, file I/O) route through
Harbor's ``BaseEnvironment`` into the sandbox container.  Trajectory
is managed locally by AgentM's own trace layer.

::

    harbor trial start -p <task> \\
        --agent-import-path agentm_harbor:ExternalAgentMAgent ...
"""

from agentm_harbor.external_agent import ExternalAgentMAgent

__all__ = ["ExternalAgentMAgent"]
