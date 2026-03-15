"""AgentM exception hierarchy.

All public exceptions inherit from AgentMError so library consumers can
catch a single base class or specific subclasses as needed.
"""


class AgentMError(Exception):
    """Base exception for all AgentM errors."""


class ConfigError(AgentMError):
    """Configuration loading or validation error."""


class DataInitError(AgentMError):
    """Data directory initialization failed."""


class ToolError(AgentMError):
    """Tool execution error."""


class AgentError(AgentMError):
    """Agent execution error."""


class CheckpointError(AgentMError):
    """Checkpoint read/write error."""


class StoreNotInitializedError(AgentMError):
    """A store (knowledge/memory) was used before initialization."""
