"""AgentM kernel ABI — the single import surface for atoms.

Atoms import from ``agentm.core.abi`` only; direct sub-module imports
(``from agentm.core.abi.events import ...``) are forbidden by the §11
validator. This keeps the public surface explicit and auditable.

The ABI re-exports the complete atom-facing vocabulary:

- Message data model, stream boundary, termination hints
- Event bus + typed events (kernel + runtime lifecycle)
- Tool contract, session API
- Operations, resources, catalog, roles
- Telemetry Protocol, trace reader, skills
- Trajectory model (Turn, Round, Trigger, ContextPolicy)
"""

from __future__ import annotations

# -- bus ---------------------------------------------------------------------
from .bus import (
    BusPriority,
    Envelope,
    Event,
    EventBus,
    Handler,
)

# -- events ------------------------------------------------------------------
from .events import (
    BeforeRunEvent,
    BeforeSendEvent,
    BudgetExhausted,
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    ContextEvent,
    DecideEvent,
    DiagnosticEvent,
    Inject,
    LoopAction,
    MaxTurnsExhausted,
    ModelEndTurn,
    NoPendingInput,
    ProviderTruncated,
    RunEndEvent,
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
    TurnBeginEvent,
    TurnCommittedEvent,
    TurnObservation,
)

# -- trajectory --------------------------------------------------------------
from .trajectory import (
    Outcome,
    Round,
    ToolRecord,
    Turn,
    TurnMeta,
    TurnRef,
)

# -- trigger -----------------------------------------------------------------
from .trigger import (
    BackgroundCompletion,
    ContinueTrigger,
    Injection,
    MonitorFire,
    SubagentResult,
    Trigger,
    TriggerRenderer,
    UserInput,
)

# -- context -----------------------------------------------------------------
from .context import (
    ContextPolicy,
    PolicyContext,
    build_context,
    build_context_sync,
    render_trigger,
    turn_to_messages,
)

# -- store -------------------------------------------------------------------
from .store import (
    SessionMeta,
    TrajectoryStore,
)

# -- tree --------------------------------------------------------------------
from .tree import (
    EdgeKind,
    SessionEdge,
    SessionGraphProtocol,
    SessionNode,
)

# -- codec -------------------------------------------------------------------
from .codec import (
    CodecRegistry,
    DEFAULT_CODEC,
    RawTrigger,
    TriggerCodec,
    deserialize_message,
    serialize_message,
)

# -- services ----------------------------------------------------------------
from .services import (
    ServiceNotFound,
    ServiceRegistry,
    ServiceTypeMismatch,
)

# -- session_api -------------------------------------------------------------
from .session_api import (
    AtomAPI,
    SessionContext,
    SpawnedSession,
    Unsubscribe,
)

# -- lifecycle ---------------------------------------------------------------
from .lifecycle import (
    AbandonEvent,
    ForkEvent,
    LifecycleHook,
    LifecycleHookRegistry,
    ReplayEvent,
    ResumeEvent,
)

# -- manifest ----------------------------------------------------------------
from .manifest import ChannelEffects as ChannelEffects
from .manifest import ExtensionManifest as ExtensionManifest

# -- presenter ---------------------------------------------------------------
from .presenter import PHASE_GLYPHS, Phase

# -- provider ----------------------------------------------------------------
from .provider import ProviderConfig, ProviderManifest, ProviderResolver

# -- retry -------------------------------------------------------------------
from .retry import RetryPolicy

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

# -- v1 backward compat (atoms still import these) -----------------------
from ._v1_compat import (  # noqa: F401
    AgentEndEvent,
    AgentLoop,
    AgentSessionConfig,
    AgentStartEvent,
    AfterCompactEvent,
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    AtomInfo,
    BackgroundActivityEvent,
    BeforeAgentStartEvent,
    BeforeCompactEvent,
    BeforeInstallAtomEvent,
    BeforeSendToLlmEvent,
    BeforeUnloadAtomEvent,
    ChildSessionExtendingEvent,
    CommandDispatchedEvent,
    CommandDispatcher,
    CommandSpec,
    CompactionDetails,
    CompactionPrompts,
    CompactionResult,
    CompactionSettings,
    ContextUsageSnapshot,
    CostBudgetExceededEvent,
    DecideTurnActionEvent,
    ENTRY_MATERIALIZERS,
    ENTRY_TYPE_BRANCH_SUMMARY,
    ENTRY_TYPE_COMPACTION,
    ENTRY_TYPE_MESSAGE,
    EntryAppendedEvent,
    EventBusObserver,
    ExtensionAPI,
    ExtensionInstallEvent,
    ExtensionLoadError,
    ExtensionReloadEvent,
    ExtensionStaleError,
    ExtensionUnloadEvent,
    HookContract,
    InputEvent,
    InstallAtomResult,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    LoopConfig,
    MUTABLE_EVENT_FIELDS_BY_TYPE,
    MessageAppendedEvent,
    MessagePersistedEvent,
    ObserverCallback,
    ObserverRegistration,
    PlanSubmittedEvent,
    PROMPT_BRANCH_SUMMARY,
    PROMPT_BRANCH_SUMMARY_PREAMBLE,
    PROMPT_SUMMARIZATION,
    PROMPT_SUMMARIZATION_SYSTEM,
    PROMPT_UPDATE_SUMMARIZATION,
    ReloadResult,
    Renderer,
    ResolveSubagentEvent,
    ResourceWriteEvent,
    ResourcesDiscoverEvent,
    SessionEntry,
    SessionHeaderEmittedEvent,
    SessionState,
    SessionStore,
    TurnEndEvent,
    TurnStartEvent,
    UnloadAtomResult,
)

# -- trace reader (lib, re-exported for access) -------------------------
from agentm.core.lib.trace_reader import (  # noqa: F401
    LogRecord,
    SessionIdentity,
    Span,
    TraceReader,
    attr,
)

__all__ = [
    # bus
    "BusPriority",
    "Envelope",
    "Event",
    "EventBus",
    "Handler",
    # events
    "BeforeRunEvent",
    "BeforeSendEvent",
    "BudgetExhausted",
    "ChildSessionEndEvent",
    "ChildSessionStartEvent",
    "ContextEvent",
    "DecideEvent",
    "DiagnosticEvent",
    "Inject",
    "LoopAction",
    "MaxTurnsExhausted",
    "ModelEndTurn",
    "NoPendingInput",
    "ProviderTruncated",
    "RunEndEvent",
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
    "TurnBeginEvent",
    "TurnCommittedEvent",
    "TurnObservation",
    # trajectory
    "Outcome",
    "Round",
    "ToolRecord",
    "Turn",
    "TurnMeta",
    "TurnRef",
    # trigger
    "BackgroundCompletion",
    "ContinueTrigger",
    "Injection",
    "MonitorFire",
    "SubagentResult",
    "Trigger",
    "TriggerRenderer",
    "UserInput",
    # context
    "ContextPolicy",
    "PolicyContext",
    "build_context",
    "build_context_sync",
    "render_trigger",
    "turn_to_messages",
    # store
    "SessionMeta",
    "TrajectoryStore",
    # tree
    "EdgeKind",
    "SessionEdge",
    "SessionGraphProtocol",
    "SessionNode",
    # codec
    "CodecRegistry",
    "DEFAULT_CODEC",
    "RawTrigger",
    "TriggerCodec",
    "deserialize_message",
    "serialize_message",
    # services
    "ServiceNotFound",
    "ServiceRegistry",
    "ServiceTypeMismatch",
    # session_api
    "AtomAPI",
    "SessionContext",
    "SpawnedSession",
    "Unsubscribe",
    # lifecycle
    "AbandonEvent",
    "ForkEvent",
    "LifecycleHook",
    "LifecycleHookRegistry",
    "ReplayEvent",
    "ResumeEvent",
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
]
