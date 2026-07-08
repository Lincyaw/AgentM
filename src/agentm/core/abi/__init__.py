"""AgentM kernel ABI — the single import surface for atoms.

Atoms import from ``agentm.core.abi`` only; direct sub-module imports
(``from agentm.core.abi.events import ...``) are forbidden by the §11
validator. This keeps the public surface explicit and auditable.

The ABI re-exports the complete atom-facing vocabulary:

- Message data model, stream boundary, termination hints
- Event bus + typed events (kernel + runtime lifecycle)
- Tool contract, extension API, loop
- Operations, resources, catalog, session, roles
- Telemetry Protocol, trace reader, skills
"""

from __future__ import annotations

# -- bus ---------------------------------------------------------------------
from .bus import (
    EventBus,
    EventBusObserver,
    Handler,
    ObserverCallback,
    ObserverRegistration,
)

# -- events ------------------------------------------------------------------
from .events import (
    AfterCompactEvent,
    AgentEndEvent,
    AgentStartEvent,
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    BackgroundActivityEvent,
    BeforeAgentStartEvent,
    BeforeCompactEvent,
    BeforeInstallAtomEvent,
    BeforeSendToLlmEvent,
    BeforeUnloadAtomEvent,
    BudgetExhausted,
    BusPriority,
    ChildSessionEndEvent,
    ChildSessionExtendingEvent,
    ChildSessionStartEvent,
    CommandDispatchedEvent,
    ContextEvent,
    CostBudgetExceededEvent,
    DecideTurnActionEvent,
    DiagnosticEvent,
    EntryAppendedEvent,
    Event,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
    ExtensionUnloadEvent,
    HookContract,
    Inject,
    InputEvent,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    LoopAction,
    MaxTurnsExhausted,
    MessageAppendedEvent,
    MessagePersistedEvent,
    ModelEndTurn,
    NoPendingInput,
    PlanSubmittedEvent,
    ProviderProtocolViolation,
    ProviderTruncated,
    ResolveSubagentEvent,
    ResourceWriteEvent,
    ResourcesDiscoverEvent,
    SessionHeaderEmittedEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
    SignalAborted,
    Step,
    Stop,
    StreamDeltaEvent,
    TerminationCause,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolTerminated,
    TurnEndEvent,
    TurnObservation,
    TurnStartEvent,
)

# -- loop --------------------------------------------------------------------
from .loop import AgentLoop, LoopConfig

# -- manifest ----------------------------------------------------------------
from .manifest import ChannelEffects as ChannelEffects
from .manifest import ExtensionManifest as ExtensionManifest

# -- presenter ---------------------------------------------------------------
from .presenter import PHASE_GLYPHS, Phase

# -- provider ----------------------------------------------------------------
from .provider import ProviderConfig, ProviderManifest, ProviderResolver

# -- retry -------------------------------------------------------------------
from .retry import RetryPolicy

# -- session_store -----------------------------------------------------------
from .session_store import SessionState, SessionStore

# -- telemetry ---------------------------------------------------------------
from .telemetry import SessionTelemetry

# -- messages ----------------------------------------------------------------
from .messages import (
    AgentMessage,
    AssistantContent,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    text_message,
    tool_result,
)

# -- stream ------------------------------------------------------------------
from .stream import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    StreamFn,
    TextDelta,
    ThinkingDelta,
    ToolCallArgsDelta,
    ToolCallArgsParseError,
    ToolCallEnd,
    ToolCallStart,
)

# -- termination -------------------------------------------------------------
from .termination import (
    Aborted,
    EndTurn,
    MaxTokens,
    PauseTurn,
    ProviderError,
    TerminationHint,
    ToolUseExpected,
    VendorSpecific,
)

# -- tool --------------------------------------------------------------------
from .tool import (
    FILE_OP_EDIT,
    FILE_OP_METADATA_KEY,
    FILE_OP_READ,
    FILE_OP_WRITE,
    FunctionTool,
    TOOL_EXECUTION_DOMAIN_EVENT_LOOP,
    TOOL_EXECUTION_DOMAIN_METADATA_KEY,
    TOOL_EXECUTION_DOMAIN_PROCESS,
    TOOL_EXECUTION_DOMAIN_SANDBOX,
    TOOL_EXECUTION_DOMAIN_THREAD,
    TOOL_RESULT_FORMAT_METADATA_KEY,
    Tool,
    ToolContinue,
    ToolExecutionDomain,
    ToolOutcome,
    ToolResult,
    ToolTerminate,
)

# -- tool execution ----------------------------------------------------------
from .tool_executor import (
    ToolExecutionDomainUnavailable,
    ToolProcessFailed,
    ToolProcessTerminated,
    execute_tool_call,
    tool_execution_domain,
)

# -- extension ---------------------------------------------------------------
from .extension import (
    AtomInfo,
    CommandSpec,
    CommandDispatcher,
    ExtensionAPI,
    ExtensionLoadError,
    ExtensionStaleError,
    InstallAtomResult,
    ReloadResult,
    UnloadAtomResult,
    Unsubscribe,
)

# -- operations --------------------------------------------------------------
from .operations import (
    BashOperations,
    ExecResult,
)

# -- resource ----------------------------------------------------------------
from .resource import (
    BatchHandle,
    PathClass,
    ResourceWriter,
    WriteResult,
    WriterAuthor,
)

# -- catalog -----------------------------------------------------------------
from .catalog import ActiveSetFingerprint, atom_decisions_path

# -- compaction --------------------------------------------------------------
from .compaction import (
    CompactionDetails,
    CompactionPrompts,
    CompactionResult,
    CompactionSettings,
    ContextUsageSnapshot,
    PROMPT_BRANCH_SUMMARY,
    PROMPT_BRANCH_SUMMARY_PREAMBLE,
    PROMPT_SUMMARIZATION,
    PROMPT_SUMMARIZATION_SYSTEM,
    PROMPT_UPDATE_SUMMARIZATION,
)

# -- session -----------------------------------------------------------------
from .session import (
    ENTRY_MATERIALIZERS,
    ENTRY_TYPE_BRANCH_SUMMARY,
    ENTRY_TYPE_COMPACTION,
    ENTRY_TYPE_MESSAGE,
    SessionEntry,
)

# -- session_config ----------------------------------------------------------
from .session_config import AgentSessionConfig

# -- skill -------------------------------------------------------------------
from .skill import SkillDiagnostic, SkillRecord

# -- prompt_template ---------------------------------------------------------
from .prompt_template import PromptRegistry, PromptTemplateRecord

# -- roles -------------------------------------------------------------------
from .roles import (
    APPROVAL_MANAGER_SERVICE,
    ARTIFACT_STORE_SERVICE,
    COMMAND_PARSER,
    COMPACTION_PROMPTS,
    COST_QUERY_SERVICE,
    GATEWAY_SCHEDULER_SERVICE,
    LOOP_BUDGET_SERVICE,
    MODEL_RESOLVER_SERVICE,
    PARENT_PROVIDER_CONFIG_KEY,
    PROMPT_REGISTRY,
    PROMPT_TEMPLATES_SERVICE,
    PROVIDER_INHERITOR,
    RETRY_POLICY_SERVICE,
    SESSION_STORE_SERVICE,
    SLASH_COMMAND_DISPATCHER_SERVICE,
    SUB_AGENT_RUNTIME,
    SYSTEM_PROMPT_PROVIDER,
    WIRE_CHILD_FORWARDER_SERVICE,
    WIRE_OUTBOUND_SERVICE,
)

# -- trace reader (lib, re-exported for access) -------------------------
from agentm.core.lib.trace_reader import (  # noqa: E402
    LogRecord,
    SessionIdentity,
    Span,
    TraceReader,
    attr,
)

__all__ = [
    # bus
    "EventBus",
    "EventBusObserver",
    "Handler",
    "ObserverCallback",
    "ObserverRegistration",
    # events
    "AfterCompactEvent",
    "AgentEndEvent",
    "AgentStartEvent",
    "ApiRegisterEvent",
    "ApiSendUserMessageEvent",
    "BackgroundActivityEvent",
    "BeforeAgentStartEvent",
    "BeforeCompactEvent",
    "BeforeInstallAtomEvent",
    "BeforeSendToLlmEvent",
    "BeforeUnloadAtomEvent",
    "BudgetExhausted",
    "BusPriority",
    "ChildSessionEndEvent",
    "ChildSessionExtendingEvent",
    "ChildSessionStartEvent",
    "CommandDispatchedEvent",
    "ContextEvent",
    "CostBudgetExceededEvent",
    "DecideTurnActionEvent",
    "DiagnosticEvent",
    "EntryAppendedEvent",
    "Event",
    "ExtensionInstallEvent",
    "ExtensionReloadEvent",
    "ExtensionUnloadEvent",
    "HookContract",
    "Inject",
    "InputEvent",
    "LlmRequestEndEvent",
    "LlmRequestStartEvent",
    "LoopAction",
    "MaxTurnsExhausted",
    "MessageAppendedEvent",
    "MessagePersistedEvent",
    "ModelEndTurn",
    "NoPendingInput",
    "PlanSubmittedEvent",
    "ProviderProtocolViolation",
    "ProviderTruncated",
    "ResolveSubagentEvent",
    "ResourceWriteEvent",
    "ResourcesDiscoverEvent",
    "SessionHeaderEmittedEvent",
    "SessionReadyEvent",
    "SessionShutdownEvent",
    "SignalAborted",
    "Step",
    "Stop",
    "StreamDeltaEvent",
    "TerminationCause",
    "ToolCallEvent",
    "ToolErrorEvent",
    "ToolResultEvent",
    "ToolTerminated",
    "TurnEndEvent",
    "TurnObservation",
    "TurnStartEvent",
    # loop
    "AgentLoop",
    "LoopConfig",
    # manifest
    "ChannelEffects",
    "ExtensionManifest",
    # presenter
    "PHASE_GLYPHS",
    "Phase",
    # provider
    "ProviderConfig",
    "ProviderManifest",
    "ProviderResolver",
    # retry
    "RetryPolicy",
    # session_store
    "SessionState",
    "SessionStore",
    # telemetry
    "SessionTelemetry",
    # messages
    "AgentMessage",
    "AssistantContent",
    "AssistantMessage",
    "ImageContent",
    "TextContent",
    "ThinkingBlock",
    "ToolCallBlock",
    "ToolResultBlock",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "text_message",
    "tool_result",
    # stream
    "AssistantStreamEvent",
    "MessageEnd",
    "Model",
    "StreamFn",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallArgsDelta",
    "ToolCallArgsParseError",
    "ToolCallEnd",
    "ToolCallStart",
    # termination
    "Aborted",
    "EndTurn",
    "MaxTokens",
    "PauseTurn",
    "ProviderError",
    "TerminationHint",
    "ToolUseExpected",
    "VendorSpecific",
    # tool
    "FILE_OP_EDIT",
    "FILE_OP_METADATA_KEY",
    "FILE_OP_READ",
    "FILE_OP_WRITE",
    "FunctionTool",
    "TOOL_EXECUTION_DOMAIN_EVENT_LOOP",
    "TOOL_EXECUTION_DOMAIN_METADATA_KEY",
    "TOOL_EXECUTION_DOMAIN_PROCESS",
    "TOOL_EXECUTION_DOMAIN_SANDBOX",
    "TOOL_EXECUTION_DOMAIN_THREAD",
    "TOOL_RESULT_FORMAT_METADATA_KEY",
    "Tool",
    "ToolContinue",
    "ToolExecutionDomain",
    "ToolOutcome",
    "ToolResult",
    "ToolTerminate",
    "ToolExecutionDomainUnavailable",
    "ToolProcessFailed",
    "ToolProcessTerminated",
    "execute_tool_call",
    "tool_execution_domain",
    # extension
    "AtomInfo",
    "CommandSpec",
    "CommandDispatcher",
    "ExtensionAPI",
    "ExtensionLoadError",
    "ExtensionStaleError",
    "InstallAtomResult",
    "ReloadResult",
    "UnloadAtomResult",
    "Unsubscribe",
    # operations
    "BashOperations",
    "ExecResult",
    # resource
    "BatchHandle",
    "PathClass",
    "ResourceWriter",
    "WriteResult",
    "WriterAuthor",
    # catalog
    "ActiveSetFingerprint",
    "atom_decisions_path",
    # compaction
    "CompactionDetails",
    "CompactionPrompts",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageSnapshot",
    "PROMPT_BRANCH_SUMMARY",
    "PROMPT_BRANCH_SUMMARY_PREAMBLE",
    "PROMPT_SUMMARIZATION",
    "PROMPT_SUMMARIZATION_SYSTEM",
    "PROMPT_UPDATE_SUMMARIZATION",
    # session
    "ENTRY_MATERIALIZERS",
    "ENTRY_TYPE_BRANCH_SUMMARY",
    "ENTRY_TYPE_COMPACTION",
    "ENTRY_TYPE_MESSAGE",
    "SessionEntry",
    # session_config
    "AgentSessionConfig",
    # skill
    "SkillDiagnostic",
    "SkillRecord",
    # prompt_template
    "PromptRegistry",
    "PromptTemplateRecord",
    # roles
    "APPROVAL_MANAGER_SERVICE",
    "ARTIFACT_STORE_SERVICE",
    "COMMAND_PARSER",
    "COMPACTION_PROMPTS",
    "COST_QUERY_SERVICE",
    "GATEWAY_SCHEDULER_SERVICE",
    "LOOP_BUDGET_SERVICE",
    "MODEL_RESOLVER_SERVICE",
    "PARENT_PROVIDER_CONFIG_KEY",
    "PROMPT_REGISTRY",
    "PROMPT_TEMPLATES_SERVICE",
    "PROVIDER_INHERITOR",
    "RETRY_POLICY_SERVICE",
    "SESSION_STORE_SERVICE",
    "SLASH_COMMAND_DISPATCHER_SERVICE",
    "SUB_AGENT_RUNTIME",
    "SYSTEM_PROMPT_PROVIDER",
    "WIRE_CHILD_FORWARDER_SERVICE",
    "WIRE_OUTBOUND_SERVICE",
    # trace reader
    "LogRecord",
    "SessionIdentity",
    "Span",
    "TraceReader",
    "attr",
]
