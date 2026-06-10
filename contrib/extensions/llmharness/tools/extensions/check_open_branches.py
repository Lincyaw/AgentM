"""§11 single-file check atom: flag dec/hyp events with no outgoing data edge.

A ``dec`` or ``hyp`` event without any outgoing
:class:`~llmharness.schema.EdgeKind.DATA` edge is the v3-graph
approximation of issue #134's "discarded alternative without closing
evidence" — the agent registered a choice or claim but produced no
evidence-flow follow-up linking back to it.

Output: one :class:`~llmharness.schema.Finding` per such open event.
Category: ``"open_branches"``.

§11 contract: single file, no atom-to-atom imports, no
``core._internal`` import, no ``harness.session`` import.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from ..runtime.registry import SERVICE_KEY, AuditCheckRegistry, CheckContext
from ..schema import EdgeKind, EventKind, Finding

class CheckOpenBranchesConfig(BaseModel):
    model_config = {"extra": "allow"}


MANIFEST = ExtensionManifest(
    name="check_open_branches",
    description=(
        "Reference audit check: flags dec/hyp events that have no "
        "outgoing data edge as open branches. Registers itself via the "
        "llmharness.audit_registry service."
    ),
    registers=(),
    config_schema=CheckOpenBranchesConfig,
    api_version=1,
    tier=1,
)


_OPEN_KINDS = frozenset({EventKind.DEC, EventKind.HYP})


class _OpenBranchesCheck:
    """Pure check: dec/hyp events with no outgoing data edge."""

    name: str = "open_branches"

    def __call__(self, ctx: CheckContext) -> list[Finding]:
        outgoing_data: set[int] = {edge.src for edge in ctx.edges if edge.kind is EdgeKind.DATA}
        findings: list[Finding] = []
        for ev in ctx.events:
            if ev.kind not in _OPEN_KINDS:
                continue
            if ev.id in outgoing_data:
                continue
            findings.append(
                Finding(
                    category="open_branches",
                    description=(
                        f"{ev.kind.value} event #{ev.id} {ev.summary!r}: "
                        "no downstream data edge found"
                    ),
                    related_event_ids=(ev.id,),
                )
            )
        return findings


def install(api: ExtensionAPI, config: CheckOpenBranchesConfig) -> None:
    """Register the open-branches check on the parent audit registry."""
    registry = api.get_service(SERVICE_KEY)
    if not isinstance(registry, AuditCheckRegistry):
        raise RuntimeError(
            "audit registry service not published; mount llmharness.atom first"
        )
    registry.register_check(_OpenBranchesCheck())


__all__ = ["MANIFEST", "install"]
