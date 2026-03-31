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
    """Tool execution or registration error."""


class AgentError(AgentMError):
    """Agent runtime error."""


class CheckpointError(AgentMError):
    """Checkpoint read/write error."""


class StoreNotInitializedError(AgentMError):
    """Attempted to use a store before initialization."""
