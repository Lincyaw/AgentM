"""§11 single-file check atom: flag repeated tool-call signatures.

Heuristic — two distinct ``act`` events sharing an identical
``summary`` count as a repeat. The v3 extractor prompt does not
prescribe a fixed ``tool_name(args)`` shape for ``act`` summaries
(see ``audit/extractor/prompt.py``), so a stricter parser would over-
fit a contract the extractor is not held to. Equality on the
summary string is the most defensible signature available without
an inline arg-hash field on :class:`~llmharness.schema.Event`.

Output: one :class:`~llmharness.schema.Finding` per group of
≥2 events sharing an act-summary, with the full event-id tuple in
``related_event_ids``. Category: ``"repeated_actions"``.

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
    name="check_repeated_actions",
    description=(
        "Reference audit check: flags two-or-more act events sharing an "
        "identical summary as a repeated-action signal. Registers itself "
        "via the llmharness.audit_registry service."
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


class _RepeatedActionsCheck:
    """Pure check: groups identical-summary ``act`` events."""

    name: str = "repeated_actions"

    def __call__(self, ctx: CheckContext) -> list[Finding]:
        # Group act events by their summary, preserving first-seen order
        # so output is deterministic.
        order: list[str] = []
        groups: dict[str, list[int]] = {}
        for ev in ctx.events:
            if ev.kind is not EventKind.ACT:
                continue
            key = ev.summary
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(ev.id)

        findings: list[Finding] = []
        for summary in order:
            ids = groups[summary]
            if len(ids) < 2:
                continue
            findings.append(
                Finding(
                    category="repeated_actions",
                    description=(
                        f"act-summary {summary!r} appears {len(ids)} times (events: {list(ids)})"
                    ),
                    related_event_ids=tuple(ids),
                )
            )
        return findings


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    """Register the repeated-actions check on the parent audit registry."""

    del config  # no configuration knobs
    registry = api.get_service(SERVICE_KEY)
    if not isinstance(registry, AuditCheckRegistry):
        raise RuntimeError(
            "audit registry service not published; mount llmharness.atom first"
        )
    registry.register_check(_RepeatedActionsCheck())


__all__ = ["MANIFEST", "install"]
