"""Runtime substrate for sessions, extension loading, and execution.

Holds the stateful side of the SDK: the extension loader,
session factory, driver, and live session state.
Modules here may touch the filesystem at runtime but perform no side
effects at module import time (``agentm.core`` stays Jupyter-clean to
import). Atoms reach this layer only through ``agentm.core.abi.*``
Protocols and ``api.*`` hooks.
"""

from __future__ import annotations

from agentm.core.runtime.catalog import (
    InMemoryAtomCatalog,
    InMemoryVersionedResourceStore,
)
from agentm.core.runtime.session import Session, SessionRuntimeConfig
from agentm.core.runtime.execution import Execution
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.runtime.trigger_queue import TriggerQueue, QueueClosed
from agentm.core.runtime.tree import InMemorySessionGraph
from agentm.core.lib.trajectory_query import TrajectoryStoreQueryAdapter
from agentm.core.runtime.session_factory import SessionBuildConfig, create_session
from agentm.core.runtime.extension import load_extension, ExtensionLoadError

__all__ = [
    "Execution",
    "ExtensionLoadError",
    "InMemoryAtomCatalog",
    "InMemorySessionGraph",
    "InMemoryVersionedResourceStore",
    "QueueClosed",
    "Session",
    "SessionBuildConfig",
    "SessionRuntimeConfig",
    "Trajectory",
    "TrajectoryStoreQueryAdapter",
    "TriggerQueue",
    "create_session",
    "load_extension",
]
