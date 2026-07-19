"""Runtime substrate — sessions, extensions, catalog, resources.

Holds the stateful side of the SDK: the extension loader,
catalog/freeze machinery, default session/resource backends.
Modules here may touch the filesystem at runtime but perform no side
effects at module import time (``agentm.core`` stays Jupyter-clean to
import). Atoms reach this layer only through ``agentm.core.abi.*``
Protocols and ``api.*`` hooks; direct atom imports of
``agentm.core.runtime.*`` are validator-rejected.
"""

from __future__ import annotations

# v2 session model
from agentm.core.runtime.session import Session
from agentm.core.runtime.execution import Execution
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.runtime.trigger_queue import TriggerQueue, QueueClosed
from agentm.core.runtime.tree import InMemorySessionGraph
from agentm.core.runtime.stores.memory import InMemoryTrajectoryStore
from agentm.core.runtime.session_factory import create_session
from agentm.core.runtime.extension import load_extension, ExtensionLoadError

__all__ = [
    "Execution",
    "ExtensionLoadError",
    "InMemorySessionGraph",
    "InMemoryTrajectoryStore",
    "QueueClosed",
    "Session",
    "Trajectory",
    "TriggerQueue",
    "create_session",
    "load_extension",
]
