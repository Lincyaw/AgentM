"""LLM-as-harness: cognitive-audit AgentM extension.

Core package — agent definitions + composable primitives.

The runtime entry point is ``llmharness.atom``, loaded via
``AgentSessionConfig(extensions=[("llmharness.atom", {})])``.

Orchestration tools (replay, distill, aggregate, eval) live outside
this package under ``tools/`` and compose from the primitives exported
here.
"""

from .agents.auditor.auditor_tools import (
    AUDITOR_TERMINATION_REASON,
    AUDITOR_TOOL_NAMES,
    AUDITOR_TOOLS,
)
from .agents.auditor.prompt import load_auditor_prompt
from .agents.extractor.extractor_tools import (
    EXTRACTOR_TERMINATION_REASON,
    EXTRACTOR_TOOL_NAMES,
)
from .agents.extractor.prompt import load_extractor_prompt
from .primitives import (
    AuditorInput,
    AuditorOutput,
    AuditorSettings,
    CumulativeAuditState,
    ExtractorInput,
    ExtractorOutput,
    build_auditor_input,
    build_extractor_input,
    process_auditor_output,
    process_extractor_output,
    serialize_full_trajectory,
)
from .replay.record import ReplayRecord, Status, iter_records, write_record
from .schema import (
    Edge,
    EdgeKind,
    Event,
    EventKind,
    Finding,
    Phase,
    Reminder,
    Verdict,
)

__all__ = [
    "AUDITOR_TERMINATION_REASON",
    "AUDITOR_TOOLS",
    "AUDITOR_TOOL_NAMES",
    "EXTRACTOR_TERMINATION_REASON",
    "EXTRACTOR_TOOL_NAMES",
    "AuditorInput",
    "AuditorOutput",
    "AuditorSettings",
    "CumulativeAuditState",
    "Edge",
    "EdgeKind",
    "Event",
    "EventKind",
    "ExtractorInput",
    "ExtractorOutput",
    "Finding",
    "Phase",
    "Reminder",
    "ReplayRecord",
    "Status",
    "Verdict",
    "build_auditor_input",
    "build_extractor_input",
    "iter_records",
    "load_auditor_prompt",
    "load_extractor_prompt",
    "process_auditor_output",
    "process_extractor_output",
    "serialize_full_trajectory",
    "write_record",
]
