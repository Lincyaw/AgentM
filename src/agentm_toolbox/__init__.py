"""agentm-toolbox — standalone file-tool runtime.

Runs both in-process (``import agentm_toolbox``) and as a CLI inside
sandbox containers (``python3 -m agentm_toolbox read '{...}'``).
Zero external dependencies.
"""

from agentm_toolbox._file_ops import FileToolbox, Result
from agentm_toolbox._state import ReadStateStore, content_hash_for

__all__ = ["FileToolbox", "ReadStateStore", "Result", "content_hash_for"]
