"""Feishu / Lark gateway channel for AgentM.

A long-running daemon that drives :class:`agentm.harness.AgentSession`
instances from inbound Feishu chat events. The gateway is one of several
*presenters* (alongside the CLI and the Textual TUI) — there is no formal
``Channel`` abstraction in AgentM core; this package introduces the
concept under ``contrib/channels/`` so future Slack / Discord adapters
can settle into the same shape.

Surface:

- :class:`~agentm_feishu.chat_source.ChatSource` — abstract input/output
  seam (inbound message stream + outbound send / card update).
- :class:`~agentm_feishu.chat_source.StubChatSource` — in-memory
  implementation used by tests and local development.
- :class:`~agentm_feishu.feishu_source.FeishuChatSource` — production
  adapter backed by ``lark_oapi.channel.FeishuChannel`` (requires the
  ``[feishu]`` extra).
- :class:`~agentm_feishu.gateway.FeishuGateway` — composes a
  ``ChatSource`` with AgentM's ``SessionManager`` + ``AgentSession``.
- ``agentm-feishu-gateway`` — console entry point.
"""

from __future__ import annotations

__version__ = "0.1.0"
