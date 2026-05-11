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

__version__ = "0.1.0"
