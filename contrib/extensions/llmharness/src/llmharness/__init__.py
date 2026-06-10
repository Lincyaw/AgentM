"""LLM-as-harness: cognitive-audit AgentM extension.

Core package — self-contained agent definitions + shared primitives.

The runtime entry point is ``llmharness.atom``, loaded via
``AgentSessionConfig(extensions=[("llmharness.atom", {})])``.
"""

from .agents.auditor.auditor_tools import (
    AUDITOR_TERMINATION_REASON,
    AUDITOR_TOOL_NAMES,
    AUDITOR_TOOLS,
)
from .agents.auditor.prompt import load_auditor_prompt
from .agents.extractor.prompt import load_extractor_prompt
from .agents.extractor.tools.finalize_extraction import (
    FINALIZE_EXTRACTION_REASON as EXTRACTOR_TERMINATION_REASON,
)
from .primitives import (
    AuditorInput,
    AuditorOutput,
    AuditorSettings,
    CumulativeAuditState,
    build_auditor_input,
    process_auditor_output,
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
    "AuditorInput",
    "AuditorOutput",
    "AuditorSettings",
    "CumulativeAuditState",
    "Edge",
    "EdgeKind",
    "Event",
    "EventKind",
    "Finding",
    "Phase",
    "Reminder",
    "ReplayRecord",
    "Status",
    "Verdict",
    "build_auditor_input",
    "iter_records",
    "load_auditor_prompt",
    "load_extractor_prompt",
    "process_auditor_output",
    "serialize_full_trajectory",
    "write_record",
]
