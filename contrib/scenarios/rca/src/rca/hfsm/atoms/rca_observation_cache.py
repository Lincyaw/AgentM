"""``rca_observation_cache`` — memoise idempotent tool calls through L1.

Phase 1 implementation of design §3.2 second role (ObservationLog as
memoisation): a tool that is registered with ``metadata['idempotent'] =
True`` short-circuits on a re-invocation with identical args by returning
the cached observation text instead of re-executing.

Mechanism — wrap-at-agent-start (the canonical AgentM seam, matching
``file_mutation_queue`` / ``tool_filter``). At ``AgentStartEvent`` time
the atom iterates ``api.tools``, finds entries whose ``metadata`` declares
``idempotent=True``, and replaces them by index with a ``_CachedTool``
adapter. The adapter:

1. Computes the call signature: ``sha256(tool_name + canonical_json(args))``
   with sorted keys and ``(",", ":")`` separators so two semantically
   identical args dicts produce the same hash.
2. Looks up the signature in the L1 ObservationLog via
   ``rca.hgraph.read.get_observation_by_signature``.
3. On a hit: emit a ``DiagnosticEvent`` (``source='rca_observation_cache'``,
   message starts with ``'tool_call_cached'``) and return the cached
   observation text as the ``ToolResult``.
4. On a miss: delegate to the wrapped tool, then write the result text
   back to the ObservationLog with the same signature so future calls
   hit. The cache write routes through ``rca.gate.apply`` with a
   ``record_observation`` op, keeping the single-writer property intact.

Why not a pre-execution interception hook? ``ToolCallEvent`` handlers can
only ``{"block": True, "reason": ...}`` the call (which synthesises an
ERROR tool result, not a cached success). ``ToolResultEvent`` fires
after execution, defeating the cost-saving purpose. Wrapping the
registered tool at install time is the documented seam — see the plan's
"FALLBACK" instructions and the precedent in ``file_mutation_queue``.

contract:

* One MANIFEST + one ``install``. No atom-to-atom imports — the cache
  reaches the store via ``api.get_service('rca.hgraph.read')`` and the
  gate via ``api.get_service('rca.gate')``. The scenario's pure
  ``schema`` / ``updates`` modules are imported as pure data — they are
  not atoms.
* Module-level state is ``Final`` (no mutable globals).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import Any

from agentm.core.abi import (
    AgentStartEvent,
    DiagnosticEvent,
    ExtensionAPI,
    TextContent,
    Tool,
    ToolOutcome,
    ToolResult,
)
from agentm.extensions import ExtensionManifest

from rca.hfsm.schema import Observation
from rca.hfsm.updates import UpdateProposal

MANIFEST = ExtensionManifest(
    name="rca_observation_cache",
    description=(
        "Memoise idempotent tool calls through the L1 ObservationLog. "
        "Wraps tools that declare metadata['idempotent']=True at "
        "agent_start; non-idempotent tools pass through untouched."
    ),
    registers=("event:agent_start",),
    config_schema=None,
    requires=("rca_hgraph_store", "rca_falsification_gate"),
)

# ---------------------------------------------------------------------------
# Cache wrapper. Public attribute surface mirrors the ``Tool`` Protocol
# (name / description / parameters / execute) so the kernel sees an indistin-
# guishable replacement.
# ---------------------------------------------------------------------------


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _tool_signature(tool_name: str, args: dict[str, Any]) -> str:
    """``sha256(tool_name + canonical_json(args))`` — matches the public helper
    in ``rca_evidence_tools._tool_signature``.

    Re-implemented here (not imported) because forbids atom-to-atom
    imports. The canonicalisation rule is the public contract; both
    implementations must stay in sync.
    """

    canonical = tool_name + ":" + _canonical_json(args)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extract_text(result: ToolResult | ToolOutcome) -> str:
    """Best-effort cache-key text for a tool's ``ToolResult``.

    Cache stores the concatenated ``TextContent.text`` of the result. Image
    content is dropped — caching binary blobs is out of scope for Phase 1.
    """

    if isinstance(result, ToolResult):
        tr = result
    else:
        inner = getattr(result, "result", None)
        if not isinstance(inner, ToolResult):
            return ""
        tr = inner
    parts: list[str] = []
    for chunk in tr.content:
        if isinstance(chunk, TextContent):
            parts.append(chunk.text)
    return "\n".join(parts)


class _CachedTool:
    """``Tool`` Protocol adapter wrapping an idempotent tool with L1 lookup.

    ``read_handle`` is the ``rca.hgraph.read`` service; ``gate`` is the
    ``rca.gate`` service. Both are captured at agent_start, by which time
    every install has run.
    """

    def __init__(
        self,
        wrapped: Tool,
        *,
        read_handle: Any,
        gate: Any,
        emit: Any,
    ) -> None:
        self._wrapped = wrapped
        self._read = read_handle
        self._gate = gate
        self._emit = emit
        # Mirror the Tool Protocol surface.
        self.name = wrapped.name
        self.description = wrapped.description
        self.parameters = wrapped.parameters
        # Preserve metadata so downstream consumers (observability,
        # tool_filter) still see the original tags.
        self.metadata = dict(getattr(wrapped, "metadata", {}) or {})

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
    ) -> ToolResult | ToolOutcome:
        signature = _tool_signature(self.name, args)
        cached = self._read.get_observation_by_signature(signature)
        if cached is not None:
            await self._emit(signature, hit=True)
            return ToolResult(content=[TextContent(type="text", text=cached.text)])
        # Cache miss — delegate. Exceptions propagate per FunctionTool's
        # own contract; the loop converts them into ToolErrorEvent.
        out = await self._wrapped.execute(args, signal=signal)
        text = _extract_text(out)
        if text:
            obs = Observation(
                id=f"O-cache-{uuid.uuid4().hex[:12]}",
                text=text,
                source_tool_call=self.name,
                tool_signature=signature,
                ts=time.time(),
            )
            # Route the write through the gate so the single-writer
            # property (design §7.4) is preserved. record_observation has
            # no precondition and never downgrades.
            self._gate.apply(UpdateProposal(op="record_observation", observation=obs))
        return out


# ---------------------------------------------------------------------------
# Install: subscribe to agent_start, swap idempotent tools by index.
# ---------------------------------------------------------------------------


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    read_handle = api.get_service("rca.hgraph.read")
    gate = api.get_service("rca.gate")
    if read_handle is None or gate is None:
        raise RuntimeError(
            "rca_observation_cache: rca.hgraph.read and rca.gate must be "
            "published; install rca_hgraph_store and rca_falsification_gate "
            "before this atom"
        )

    wrapped_names: set[str] = set()

    async def _emit(signature: str, *, hit: bool) -> None:
        # DiagnosticEvent.message contains the literal "tool_call_cached"
        # sentinel so subscribers (observability sink, tests) can filter
        # without parsing JSON. Phase 1 keeps this on the generic
        # diagnostic channel; a typed event lives in the Phase 2 backlog.
        await api.events.emit(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="info",
                source="rca_observation_cache",
                message=(f"tool_call_cached signature={signature} hit={hit}"),
            ),
        )

    def _on_agent_start(_: AgentStartEvent) -> None:
        tools = api.tools
        for index, tool in enumerate(tools):
            if tool.name in wrapped_names:
                continue
            metadata = getattr(tool, "metadata", None) or {}
            if not isinstance(metadata, dict):
                continue
            if not bool(metadata.get("idempotent", False)):
                continue
            tools[index] = _CachedTool(
                tool,
                read_handle=read_handle,
                gate=gate,
                emit=_emit,
            )
            wrapped_names.add(tool.name)

    api.on(AgentStartEvent.CHANNEL, _on_agent_start)
