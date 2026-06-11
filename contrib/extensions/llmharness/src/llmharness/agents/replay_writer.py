"""Replay sidecar writer — subscribes to firing events and writes ReplayRecord JSONL.

Pluggable: load this atom alongside llmharness.atom to enable replay
recording. Without it, no sidecar is written and replay tools won't
have data to work with.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest

from llmharness.replay.record import (
    ReplayRecord,
    audit_session_id,
    now_ns,
    replay_log_path,
    write_record,
)
from llmharness.schema import AUDITOR_FIRED, EXTRACTOR_FIRED, FiringEvent

MANIFEST = ExtensionManifest(
    name="replay_writer",
    description="Write replay sidecar JSONL on each extractor/auditor firing.",
    registers=("event:llmharness.extractor_fired", "event:llmharness.auditor_fired"),
    requires=("llmharness",),
)

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    sid = audit_session_id(api)
    sidecar = replay_log_path(api.cwd, sid)
    trace_id = api.root_session_id

    def _on_firing(event: FiringEvent) -> None:
        write_record(sidecar, ReplayRecord(
            phase=event.phase,
            turn_index=event.turn_index,
            session_id=sid,
            trace_id=trace_id,
            ts_ns=now_ns(),
            compose_kwargs={},
            payload=event.payload,
            provider=None,
            output=event.output,
            status=event.status,
        ))

    api.on(EXTRACTOR_FIRED, _on_firing)
    api.on(AUDITOR_FIRED, _on_firing)
