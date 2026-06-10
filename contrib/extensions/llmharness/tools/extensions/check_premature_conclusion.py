"""§11 single-file check atom: flag concl events with thin incoming evidence.

Heuristic — a ``concl`` event whose total incoming edge count
(``data`` + ``ref`` combined, from any earlier event) is strictly
less than 2 is flagged as a premature conclusion. The threshold
matches "thin evidence": a single supporting edge into a conclusion
is the canonical premature-conclusion shape from issue #134. This is
deliberately conservative — checks are advisory.

Output: one :class:`~llmharness.schema.Finding` per such concl event.
Category: ``"premature_conclusion"``.

§11 contract: single file, no atom-to-atom imports, no
``core._internal`` import, no ``harness.session`` import.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from ..runtime.registry import SERVICE_KEY, AuditCheckRegistry, CheckContext
from ..schema import EventKind, Finding

MANIFEST = ExtensionManifest(
    name="check_premature_conclusion",
    description=(
        "Reference audit check: flags concl events with strictly fewer "
        "than two incoming edges (data+ref combined) as premature. "
        "Registers itself via the llmharness.audit_registry service."
    ),
    registers=(),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)


_PREMATURE_THRESHOLD = 2


class _PrematureConclusionCheck:
    """Pure check: concl events with incoming-edge count below threshold."""

    name: str = "premature_conclusion"

    def __call__(self, ctx: CheckContext) -> list[Finding]:
        incoming: dict[int, int] = {}
        for edge in ctx.edges:
            incoming[edge.dst] = incoming.get(edge.dst, 0) + 1

        findings: list[Finding] = []
        for ev in ctx.events:
            if ev.kind is not EventKind.CONCL:
                continue
            count = incoming.get(ev.id, 0)
            if count >= _PREMATURE_THRESHOLD:
                continue
            findings.append(
                Finding(
                    category="premature_conclusion",
                    description=(
                        f"concl event #{ev.id} {ev.summary!r}: "
                        f"{count} incoming edge(s), below threshold "
                        f"{_PREMATURE_THRESHOLD}"
                    ),
                    related_event_ids=(ev.id,),
                )
            )
        return findings


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    """Register the premature-conclusion check on the parent audit registry."""

    del config  # no configuration knobs
    registry = api.get_service(SERVICE_KEY)
    if not isinstance(registry, AuditCheckRegistry):
        raise RuntimeError(
            "audit registry service not published; mount llmharness.adapter first"
        )
    registry.register_check(_PrematureConclusionCheck())


__all__ = ["MANIFEST", "install"]
