"""``tool_peer_send`` — opt-in atom that lets an agent delegate to another peer.

Phase 6 of the channels gateway/client split. See
``.claude/designs/client-server-architecture.md`` §8.

Single-file §11 atom: no atom-to-atom imports, no ``harness.session``
imports, no ``core._internal`` imports. The atom expects the host
process (``agentm-worker``) to have published a ``peer_messaging``
service into the session's service registry via
``AgentSession.set_service``. The protocol that service implements is
the small surface declared below; the worker's :class:`WorkerRunner`
fulfills it.

The atom is **mountable_via_command**: it does not auto-install. To
enable agent-to-agent calls, the gateway operator must (1) start a
worker that publishes ``peer_messaging`` (i.e. ``agentm-worker``) and
(2) put ``tool_peer_send`` on the gateway's ``commands.atoms.allow``
list so ``/atom:install tool_peer_send`` succeeds.

Design choices captured here, not in the brief:

* The tool returns a structured plain-text payload (``ok=…``,
  ``correlation_id=…``, ``content=…``) instead of raw JSON so the
  LLM sees something it can paraphrase. Future iterations may switch
  to a structured ``ToolResult`` content list once we have a
  metadata-rich result type to round-trip.
* Timeouts and ``wait_for_reply=False`` both clean up the pending
  future. A late reply for either case is logged and dropped by the
  runner — no zombie state survives across turns.
"""

from __future__ import annotations

import logging
from typing import Any, Final, Protocol, runtime_checkable

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


PEER_MESSAGING_SERVICE: Final[str] = "peer_messaging"

log = logging.getLogger("agentm_worker.peer_send_atom")


@runtime_checkable
class PeerMessaging(Protocol):
    """Host-supplied protocol the atom calls into.

    Implemented by :class:`agentm_worker.runner.WorkerRunner`. Kept
    intentionally small: send one envelope, optionally await one reply.
    The gateway-facing wire client (``WireClient``) is owned by the
    worker; the atom never sees it directly.
    """

    async def send_peer(
        self,
        *,
        to: str,
        content: str,
        correlation_id: str,
    ) -> None:
        """Push one ``KIND_INBOUND`` envelope toward peer ``to``.

        The host is responsible for tagging ``peer_kind=agent_worker``
        and propagating ``root_session_key`` / ``hops`` per the wire
        contract. The atom does **not** pre-validate ``to`` — the
        gateway is the authority on which peers exist.
        """
        ...

    async def await_peer_reply(
        self,
        correlation_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Block until a reply envelope with ``correlation_id`` lands.

        Returns the reply envelope's ``body`` dict. Raises
        :class:`TimeoutError` on timeout. The host is responsible for
        cleaning the entry up on timeout so a late reply is dropped.
        """
        ...

    def new_correlation_id(self) -> str:
        """Mint a fresh correlation id (uuid4 hex is fine)."""
        ...


MANIFEST = ExtensionManifest(
    name="tool_peer_send",
    description=(
        "Send a prompt to another agent worker (or chat client) via the "
        "channels gateway. Opt-in; requires a host that publishes a "
        "peer_messaging service. Enabled by listing this atom on the "
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


_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "to": {
            "type": "string",
            "description": "Destination peer id or worker name.",
        },
        "content": {
            "type": "string",
            "description": "Prompt content to deliver.",
        },
        "wait_for_reply": {
            "type": "boolean",
            "default": True,
            "description": (
                "When true (default), block until the destination replies "
                "or the timeout fires. When false, return immediately with "
                "the assigned correlation_id; any reply that arrives later "
                "is logged and discarded."
            ),
        },
        "timeout_seconds": {
            "type": "integer",
            "default": 60,
            "description": (
                "Maximum seconds to wait for a reply. Ignored when "
                "wait_for_reply=false."
            ),
        },
    },
    "required": ["to", "content"],
    "additionalProperties": False,
}


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=text)], is_error=True
    )


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:  # noqa: ARG001
    peer_messaging = api.get_service(PEER_MESSAGING_SERVICE)
    if peer_messaging is None:
        # Refuse to register: the tool would always fail at call time.
        # Failing at install surfaces the misconfiguration to the
        # operator immediately rather than burying it in a tool error
        # the LLM has to parse.
        raise RuntimeError(
            "tool_peer_send requires a 'peer_messaging' service in the "
            "session registry. This atom only works inside agentm-worker; "
            "mounting it from another host has no effect."
        )

    async def _execute(args: dict[str, Any]) -> ToolResult:
        to_raw = args.get("to")
        content_raw = args.get("content")
        if not isinstance(to_raw, str) or not to_raw:
            return _error("peer_send: 'to' must be a non-empty string")
        if not isinstance(content_raw, str):
            return _error("peer_send: 'content' must be a string")
        wait_for_reply = bool(args.get("wait_for_reply", True))
        timeout_seconds = float(args.get("timeout_seconds", 60))
        if timeout_seconds <= 0:
            return _error("peer_send: 'timeout_seconds' must be > 0")

        correlation_id = peer_messaging.new_correlation_id()
        try:
            await peer_messaging.send_peer(
                to=to_raw,
                content=content_raw,
                correlation_id=correlation_id,
            )
        except Exception as exc:
            log.exception("peer_send: send_peer failed (to=%s)", to_raw)
            return _error(f"peer_send: send failed: {exc}")

        if not wait_for_reply:
            return _ok(
                "ok=true wait_for_reply=false "
                f"correlation_id={correlation_id} to={to_raw}"
            )

        try:
            reply_body = await peer_messaging.await_peer_reply(
                correlation_id, timeout_seconds
            )
        except TimeoutError:
            return _error(
                f"peer_send: timed out after {timeout_seconds:g}s "
                f"(correlation_id={correlation_id} to={to_raw})"
            )
        except Exception as exc:
            log.exception(
                "peer_send: await_peer_reply failed (correlation_id=%s)",
                correlation_id,
            )
            return _error(f"peer_send: await reply failed: {exc}")

        reply_content = ""
        if isinstance(reply_body, dict):
            raw = reply_body.get("content")
            if isinstance(raw, str):
                reply_content = raw
        return _ok(
            f"ok=true correlation_id={correlation_id} to={to_raw}\n"
            f"reply:\n{reply_content}"
        )

    api.register_tool(
        FunctionTool(
            name="peer_send",
            description=(
                "Send a prompt to another agent worker via the channels "
                "gateway. Use for delegation. The destination must be "
                "reachable via the gateway's worker registry by name or "
                "peer_id."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"role": "peer_send"},
        )
    )


__all__ = ["MANIFEST", "PEER_MESSAGING_SERVICE", "PeerMessaging", "install"]
