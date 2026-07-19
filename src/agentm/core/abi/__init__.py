"""AgentM kernel ABI — the single import surface for atoms.

Atoms import from ``agentm.core.abi`` only; direct sub-module imports
(``from agentm.core.abi.events import ...``) are forbidden by the §11
validator. This keeps the public surface explicit and auditable.
"""

from __future__ import annotations

# -- bus ---------------------------------------------------------------------
from .bus import (  # noqa: E402
    BusPriority,
    Envelope,
    Event,
    EventBus,
    EventBusObserver,
    Handler,
)

# -- events (kernel) ---------------------------------------------------------
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

# -- events (domain, atom-to-atom) ------------------------------------------
from .events import (  # noqa: F811
    AfterCompactEvent,
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    BackgroundActivityEvent,
    BeforeCompactEvent,
    BeforeInstallAtomEvent,
    BeforeUnloadAtomEvent,
    CommandDispatchedEvent,
    CostBudgetExceededEvent,
    EntryAppendedEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
    ExtensionUnloadEvent,
    InputEvent,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    MessageAppendedEvent,
    MessagePersistedEvent,
    PlanSubmittedEvent,
    ResolveSubagentEvent,
    ResourceWriteEvent,
    ResourcesDiscoverEvent,
    SessionHeaderEmittedEvent,
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
    AgentSessionConfig,
    AtomAPI,
    LoopConfig,
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

# -- presenter ---------------------------------------------------------------
from .presenter import PHASE_GLYPHS, Phase  # noqa: F401

# -- provider ----------------------------------------------------------------
from .provider import ProviderConfig, ProviderManifest, ProviderResolver  # noqa: F401

# -- retry -------------------------------------------------------------------
from .retry import RetryPolicy  # noqa: F401

# -- telemetry ---------------------------------------------------------------
from .telemetry import SessionTelemetry  # noqa: F401

# -- command -----------------------------------------------------------------
from .command import (
    CommandDispatcher,
    CommandSpec,
    DispatchResult,
)

# -- compaction --------------------------------------------------------------
from .compaction import (
    ENTRY_MATERIALIZERS,
    ENTRY_TYPE_BRANCH_SUMMARY,
    ENTRY_TYPE_COMPACTION,
    ENTRY_TYPE_MESSAGE,
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
    SessionEntry,
)

# -- extension errors --------------------------------------------------------
from agentm.core.runtime.extension import ExtensionLoadError  # noqa: F401

# -- trace reader (lib, re-exported for access) ------------------------------
from agentm.core.lib.trace_reader import (  # noqa: F401
    LogRecord,
    SessionIdentity,
    Span,
    TraceReader,
    attr,
)

# ---------------------------------------------------------------------------
# Non-import definitions AFTER all imports (avoids E402)
# ---------------------------------------------------------------------------


class ExtensionStaleError(RuntimeError):
    """Raised when an atom's source has changed since it was loaded."""
    ...


# Type alias — atoms typed against the legacy name still import it
ExtensionAPI = AtomAPI

# Event aliases — atoms may import these names
TurnStartEvent = TurnBeginEvent
TurnEndEvent = TurnCommittedEvent
AgentStartEvent = BeforeRunEvent
AgentEndEvent = RunEndEvent
BeforeAgentStartEvent = BeforeRunEvent
BeforeSendToLlmEvent = BeforeSendEvent
DecideTurnActionEvent = DecideEvent

__all__ = [
    # bus
    "BusPriority", "Envelope", "Event", "EventBus", "EventBusObserver", "Handler",
    # events (kernel)
    "BeforeRunEvent", "BeforeSendEvent", "BudgetExhausted",
    "ChildSessionEndEvent", "ChildSessionStartEvent",
    "ContextEvent", "DecideEvent", "DiagnosticEvent",
    "Inject", "LoopAction", "MaxTurnsExhausted", "ModelEndTurn",
    "NoPendingInput", "ProviderTruncated",
    "RunEndEvent", "SessionReadyEvent", "SessionShutdownEvent",
    "SignalAborted", "Step", "Stop", "StreamDeltaEvent",
    "TerminationCause", "ToolCallEvent", "ToolErrorEvent",
    "ToolResultEvent", "ToolTerminated",
    "TurnBeginEvent", "TurnCommittedEvent", "TurnObservation",
    # events (domain)
    "AfterCompactEvent", "ApiRegisterEvent", "ApiSendUserMessageEvent",
    "BackgroundActivityEvent", "BeforeCompactEvent",
    "BeforeInstallAtomEvent", "BeforeUnloadAtomEvent",
    "CommandDispatchedEvent", "CostBudgetExceededEvent",
    "EntryAppendedEvent", "ExtensionInstallEvent",
    "ExtensionReloadEvent", "ExtensionUnloadEvent",
    "InputEvent", "LlmRequestEndEvent", "LlmRequestStartEvent",
    "MessageAppendedEvent", "MessagePersistedEvent",
    "PlanSubmittedEvent", "ResolveSubagentEvent",
    "ResourceWriteEvent", "ResourcesDiscoverEvent",
    "SessionHeaderEmittedEvent",
    # event aliases
    "TurnStartEvent", "TurnEndEvent", "AgentStartEvent", "AgentEndEvent",
    "BeforeAgentStartEvent", "BeforeSendToLlmEvent", "DecideTurnActionEvent",
    # trajectory
    "Outcome", "Round", "ToolRecord", "Turn", "TurnMeta", "TurnRef",
    # trigger
    "BackgroundCompletion", "ContinueTrigger", "Injection", "MonitorFire",
    "SubagentResult", "Trigger", "TriggerRenderer", "UserInput",
    # context
    "ContextPolicy", "PolicyContext",
    "build_context", "build_context_sync", "render_trigger", "turn_to_messages",
    # store
    "SessionMeta", "TrajectoryStore",
    # tree
    "EdgeKind", "SessionEdge", "SessionGraphProtocol", "SessionNode",
    # codec
    "CodecRegistry", "DEFAULT_CODEC", "RawTrigger", "TriggerCodec",
    "deserialize_message", "serialize_message",
    # services
    "ServiceNotFound", "ServiceRegistry", "ServiceTypeMismatch",
    # session_api
    "AgentSessionConfig", "AtomAPI", "ExtensionAPI",
    "LoopConfig", "SessionContext", "SpawnedSession", "Unsubscribe",
    # lifecycle
    "AbandonEvent", "ForkEvent", "LifecycleHook", "LifecycleHookRegistry",
    "ReplayEvent", "ResumeEvent",
    # manifest
    "ChannelEffects", "ExtensionManifest",
    # messages
    "AgentMessage", "AssistantContent", "AssistantMessage",
    "ImageContent", "TextContent", "ThinkingBlock",
    "ToolCallBlock", "ToolResultBlock", "ToolResultMessage",
    "Usage", "UserMessage", "text_message", "tool_result",
    # stream
    "AssistantStreamEvent", "MessageEnd", "Model", "StreamFn",
    "TextDelta", "ThinkingDelta", "ToolCallArgsDelta",
    "ToolCallArgsParseError", "ToolCallEnd", "ToolCallStart",
    # termination
    "Aborted", "EndTurn", "MaxTokens", "PauseTurn",
    "ProviderError", "TerminationHint", "ToolUseExpected", "VendorSpecific",
    # tool
    "FILE_OP_EDIT", "FILE_OP_METADATA_KEY", "FILE_OP_READ", "FILE_OP_WRITE",
    "FunctionTool",
    "TOOL_EXECUTION_DOMAIN_EVENT_LOOP", "TOOL_EXECUTION_DOMAIN_METADATA_KEY",
    "TOOL_EXECUTION_DOMAIN_PROCESS", "TOOL_EXECUTION_DOMAIN_SANDBOX",
    "TOOL_EXECUTION_DOMAIN_THREAD", "TOOL_RESULT_FORMAT_METADATA_KEY",
    "Tool", "ToolContinue", "ToolExecutionDomain", "ToolOutcome",
    "ToolResult", "ToolTerminate",
    "ToolExecutionDomainUnavailable", "ToolProcessFailed",
    "ToolProcessTerminated", "execute_tool_call", "tool_execution_domain",
    # operations
    "BashOperations", "ExecResult",
    # resource
    "BatchHandle", "PathClass", "ResourceWriter", "WriteResult", "WriterAuthor",
    # catalog
    "ActiveSetFingerprint", "atom_decisions_path",
    # skill
    "SkillDiagnostic", "SkillRecord",
    # prompt_template
    "PromptRegistry", "PromptTemplateRecord",
    # command
    "CommandDispatcher", "CommandSpec", "DispatchResult",
    # compaction
    "CompactionDetails", "CompactionPrompts", "CompactionResult",
    "CompactionSettings", "ContextUsageSnapshot",
    "ENTRY_MATERIALIZERS", "ENTRY_TYPE_BRANCH_SUMMARY",
    "ENTRY_TYPE_COMPACTION", "ENTRY_TYPE_MESSAGE",
    "PROMPT_BRANCH_SUMMARY", "PROMPT_BRANCH_SUMMARY_PREAMBLE",
    "PROMPT_SUMMARIZATION", "PROMPT_SUMMARIZATION_SYSTEM",
    "PROMPT_UPDATE_SUMMARIZATION", "SessionEntry",
    # presenter / provider / retry / telemetry
    "PHASE_GLYPHS", "Phase",
    "ProviderConfig", "ProviderManifest", "ProviderResolver",
    "RetryPolicy", "SessionTelemetry",
    # errors
    "ExtensionStaleError", "ExtensionLoadError",
    # roles
    "APPROVAL_MANAGER_SERVICE", "ARTIFACT_STORE_SERVICE",
    "COMMAND_PARSER", "COMPACTION_PROMPTS", "COST_QUERY_SERVICE",
    "GATEWAY_SCHEDULER_SERVICE", "LOOP_BUDGET_SERVICE",
    "MODEL_RESOLVER_SERVICE", "PARENT_PROVIDER_CONFIG_KEY",
    "PROMPT_REGISTRY", "PROMPT_TEMPLATES_SERVICE", "PROVIDER_INHERITOR",
    "RETRY_POLICY_SERVICE", "SESSION_STORE_SERVICE",
    "SLASH_COMMAND_DISPATCHER_SERVICE", "SUB_AGENT_RUNTIME",
    "SYSTEM_PROMPT_PROVIDER", "WIRE_CHILD_FORWARDER_SERVICE",
    "WIRE_OUTBOUND_SERVICE",
]
