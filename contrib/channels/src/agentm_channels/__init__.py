"""agentm-channels — multi-channel chat gateway for AgentM.

Architecture (two-queue message bus + channel registry):

    ┌──────────────┐  inbound   ┌──────────────────┐  prompt
    │   Channels   │ ─────────► │   MessageBus     │ ────────► AgentSession
    │  (Feishu,    │            │  (asyncio.Queue) │           per session_key
    │   Slack,     │ ◄─────────┐│                  │ ◄─────── EventBus
    │   …)         │  outbound  │                  │
    └──────────────┘            └──────────────────┘

Adding a new channel = drop a single file under
``agentm_channels.channels.<name>`` extending
:class:`agentm_channels.base.BaseChannel`. Auto-discovery picks it up
from config (``channels.<name>.enabled: true``).
"""

from __future__ import annotations

import os
from pathlib import Path


__version__ = "0.1.0"


def default_socket_url() -> str:
    """Conventional ``unix://`` URL shared by gateway and clients.

    Uses ``$XDG_RUNTIME_DIR/agentm-gw.sock`` when set (per-user runtime
    dir on Linux desktops; cleared on logout), else
    ``/tmp/agentm-gw-<uid>.sock`` (uid-suffixed to avoid clobber on
    shared hosts). Peer-cred auth still restricts who can connect.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and Path(runtime).is_dir():
        return f"unix://{Path(runtime) / 'agentm-gw.sock'}"
    return f"unix:///tmp/agentm-gw-{os.geteuid()}.sock"


DEFAULT_SOCKET_URL = default_socket_url()
