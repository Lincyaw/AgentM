"""Builtin ``peer_send`` atom — same-process delegation to another session (§4).

Moved from the deleted ``agentm-worker`` package and rewritten for the
single-process gateway: there is no wire round-trip, no correlation_id,
no hops. The gateway holds every session in one ``dict[session_key,
AgentSession]``, so delegating to another conversation is a direct
in-process call.

§11 single-file atom: it never imports a gateway module. The host (the
gateway's SessionManager) publishes a ``peer_messaging`` service — a small
callable surface — into each session's service registry; this atom looks
it up at install time and registers the ``peer_send`` tool against it.
Mounting the atom outside the gateway is a no-op-with-clear-error: install
fails fast when the service is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

PEER_MESSAGING_SERVICE = "peer_messaging"

log = logging.getLogger("agentm.extensions.peer_send")


@runtime_checkable
class PeerMessaging(Protocol):
    """Host-supplied surface the atom calls into.

    Implemented by the gateway's session host. Same-process and tiny:
    deliver one prompt to a peer session and return its final text.
    """

    async def send_peer(self, *, to: str, content: str) -> str:
        """Deliver ``content`` to the peer session keyed ``to`` and return
        the peer's final assistant text. Raises ``KeyError`` when ``to``
        names no live session."""
        ...


MANIFEST = ExtensionManifest(
    name="peer_send",
    description=(
        "Send a prompt to another chat session held by the same gateway "
        "process and return its reply. Same-process dict lookup — no wire "
        "round-trip. Opt-in: requires a host that publishes a peer_messaging "
        "service (the agentm gateway). Enable by listing this atom on the "
        "gateway's commands.atoms.allow."
    ),
    registers=("tool:peer_send",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    mountable_via_command=True,
    requires=(),
)


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:  # noqa: ARG001
    peer_messaging = api.get_service(PEER_MESSAGING_SERVICE)
    if peer_messaging is None:
        raise RuntimeError(
            "peer_send requires a 'peer_messaging' service in the session "
            "registry. This atom only works inside the agentm gateway; "
            "mounting it elsewhere has no effect."
        )

    async def _execute(args: dict[str, Any]) -> ToolResult:
        to_raw = args.get("to")
        content_raw = args.get("content")
        if not isinstance(to_raw, str) or not to_raw:
            return _error("peer_send: 'to' must be a non-empty session_key")
        if not isinstance(content_raw, str) or not content_raw:
            return _error("peer_send: 'content' must be a non-empty string")
        try:
            reply = await peer_messaging.send_peer(to=to_raw, content=content_raw)
        except KeyError:
            return _error(
                f"peer_send: no live session for session_key {to_raw!r}"
            )
        except Exception as exc:
            log.exception("peer_send: send_peer failed (to=%s)", to_raw)
            return _error(f"peer_send: delivery failed: {exc}")
        return _ok(f"ok to={to_raw}\nreply:\n{reply}")

    api.register_tool(
        FunctionTool(
            name="peer_send",
            description=(
                "Send a prompt to another chat session in this gateway and "
                "return its reply. Use for delegation between conversations. "
                "The destination is the target conversation's session_key."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Destination session_key.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Prompt content to deliver.",
                    },
                },
                "required": ["to", "content"],
                "additionalProperties": False,
            },
            fn=_execute,
            metadata={"role": "peer_send"},
        )
    )
