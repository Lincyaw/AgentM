# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Session-facing ABI types.

The session config is shared by embedders and atoms that spawn children.
Runtime-only objects are allowed here when they are explicitly SDK host
injection points, but policy stays outside the factory: a session consumes
extension specs, and optional scenario names are resolved by a caller-supplied
loader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Callable, Literal, Protocol, TypeAlias, runtime_checkable

from agentm.core.abi.cancel import CancelReason, CancelSignal
from agentm.core.abi.catalog import AtomCatalog, VersionedResourceStore
from agentm.core.abi.lifecycle import EffectScope, EnvironmentRestoreFailureHandler
from agentm.core.abi.messages import AgentMessage, JsonValue, freeze_json
from agentm.core.abi.permission import PermissionPolicy
from agentm.core.abi.operations import EnvironmentOperations
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.tool import Tool
from agentm.core.abi.tool_executor import ToolExecutor
from agentm.core.abi.tool_orchestration import ToolOrchestrator
from agentm.core.abi.provider import (
    ProviderConfig,
    ProviderResolver,
    ProviderSessionIdentity,
)
from agentm.core.abi.resource import (
    ResourceReader,
    ResourceStore,
    ResourceTxn,
    ResourceWriter,
)
from agentm.core.abi.bus import EventBus, Handler
from agentm.core.abi.context import ContextPolicy
from agentm.core.abi.services import ServiceRegistry, ServiceScope
from agentm.core.abi.store import (
    TrajectoryStore,
)
from agentm.core.abi.trajectory import Turn
from agentm.core.abi.codec import TriggerCodec
from agentm.core.abi.trigger import Trigger, TriggerPriority, TriggerRenderer

Unsubscribe = Callable[[], None]
ExtensionSourceKind = Literal["module", "file"]
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ExtensionSource:
    """Portable identity for executable extension code."""

    kind: ExtensionSourceKind
    location: str
    digest: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"module", "file"}:
            raise ValueError(f"invalid extension source kind: {self.kind!r}")
        if not isinstance(self.location, str) or not self.location:
            raise TypeError("extension source location must be non-empty")
        if self.kind == "module":
            if self.digest is not None:
                raise ValueError("module extension source cannot carry a digest")
            return
        if not Path(self.location).is_absolute():
            raise ValueError("file extension source location must be absolute")
        if (
            not isinstance(self.digest, str)
            or _SHA256_RE.fullmatch(self.digest) is None
        ):
            raise ValueError(
                "file extension source digest must be sha256:<64 lowercase hex>"
            )

    @property
    def module_name(self) -> str:
        """Return the stable Python module identity used by the loader."""

        if self.kind == "module":
            return self.location
        assert self.digest is not None
        identity = hashlib.sha256(
            f"{self.location}\0{self.digest}".encode("utf-8")
        ).hexdigest()
        return f"_agentm_source_{identity}"


@dataclass(frozen=True, slots=True)
class ExtensionSpec:
    """Canonical extension source plus immutable JSON configuration."""

    source: ExtensionSource
    config: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, ExtensionSource):
            raise TypeError("extension spec source must be an ExtensionSource")
        if not isinstance(self.config, Mapping):
            raise TypeError("extension spec config must be a mapping")
        if not all(isinstance(key, str) for key in self.config):
            raise TypeError("extension spec config keys must be strings")
        frozen = freeze_json(self.config)
        if not isinstance(frozen, Mapping):
            raise TypeError("extension spec config must be an object")
        object.__setattr__(self, "config", frozen)

    @classmethod
    def from_module(
        cls,
        module: str,
        config: Mapping[str, JsonValue] | None = None,
    ) -> "ExtensionSpec":
        return cls(
            source=ExtensionSource(kind="module", location=module),
            config={} if config is None else config,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        digest: str,
        config: Mapping[str, JsonValue] | None = None,
    ) -> "ExtensionSpec":
        return cls(
            source=ExtensionSource(
                kind="file",
                location=path,
                digest=digest,
            ),
            config={} if config is None else config,
        )

    @property
    def module_path(self) -> str:
        return self.source.module_name

    def with_config(self, config: Mapping[str, JsonValue]) -> "ExtensionSpec":
        return ExtensionSpec(source=self.source, config=config)


ExtensionShorthand: TypeAlias = tuple[str, Mapping[str, JsonValue]]
ExtensionInput: TypeAlias = ExtensionSpec | ExtensionShorthand


def normalize_extension_spec(value: ExtensionInput) -> ExtensionSpec:
    """Normalize the public module tuple shorthand into a canonical spec."""

    if isinstance(value, ExtensionSpec):
        return ExtensionSpec(source=value.source, config=value.config)
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not isinstance(value[0], str)
        or not isinstance(value[1], Mapping)
    ):
        raise TypeError("extension must be ExtensionSpec or (module, config) tuple")
    return ExtensionSpec.from_module(value[0], value[1])


ChildCancellationMode = Literal["inherit", "independent"]
ConfigSource = Literal[
    "explicit",
    "atom_override",
    "env",
    "project_config",
    "user_config",
    "scenario_default",
    "provider_default",
]
SESSION_CONFIG_PRECEDENCE: tuple[ConfigSource, ...] = (
    "explicit",
    "atom_override",
    "env",
    "project_config",
    "user_config",
    "scenario_default",
    "provider_default",
)


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """Driver loop budget — max turns and tool calls per session."""

    max_turns: int | None = None
    max_tool_calls: int | None = None


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Resolved scenario configuration.

    ``extensions`` is the only load-bearing output. ``base_dir`` is optional
    metadata for atoms that resolve scenario-local files, such as
    ``system_prompt.prompt_file``.
    """

    extensions: Sequence[ExtensionInput]
    base_dir: str | None = None


@runtime_checkable
class ScenarioLoader(Protocol):
    """Resolve a scenario name to extension specs.

    Hosts can back this with files, a database, an in-memory registry, or any
    other configuration source. The runtime factory does not know those
    storage locations.
    """

    def __call__(self, scenario: str) -> ScenarioSpec | Sequence[ExtensionInput]: ...


@dataclass(slots=True)
class AgentSessionConfig:
    """Configuration for creating or spawning a session.

    Primary path: pass ``extensions`` directly. ``scenario`` is only a named
    indirection resolved through ``scenario_loader``. The core runtime does not
    own a built-in scenario registry; packaged helpers live outside core.
    """

    cwd: str = ""
    scenario: str | None = None
    scenario_loader: ScenarioLoader | None = None
    spec_resolver: SessionSpecResolver | None = None
    extensions: list[ExtensionInput] | None = None
    extra_extensions: list[ExtensionInput] = field(default_factory=list)
    extra_tools: list[Tool] = field(default_factory=list)
    atom_config_overrides: dict[str, dict[str, JsonValue]] = field(default_factory=dict)
    provider: ExtensionInput | None = None
    provider_resolver: ProviderResolver | None = None
    stream_fn: StreamFn | None = None
    model: Model | None = None
    system: str | None = None
    resource_reader: ResourceReader | None = None
    resource_store: ResourceStore | None = None
    resource_writer: ResourceWriter | None = None
    tool_executor: ToolExecutor | None = None
    tool_orchestrator: ToolOrchestrator | None = None
    permission_policy: PermissionPolicy | None = None
    effect_scope: EffectScope | None = None
    environment_operations: EnvironmentOperations | None = None
    environment_restore_failure_handler: EnvironmentRestoreFailureHandler | None = None
    versioned_resource_store: VersionedResourceStore | None = None
    atom_catalog: AtomCatalog | None = None
    bus: EventBus | None = None
    trajectory_store: TrajectoryStore | None = None
    initial_turns: list[Turn] = field(default_factory=list)
    tool_allowlist: list[str] | None = None
    purpose: str = "subagent"
    loop_config: LoopConfig | None = None
    experiment: dict[str, JsonValue] | None = None
    session_id: str | None = None
    root_session_id: str | None = None
    parent_session_id: str | None = None
    cancel_signal: CancelSignal | None = None
    parent_cancellation: ChildCancellationMode = "inherit"


@dataclass(frozen=True, slots=True)
class ConfigValueProvenance:
    """Typed provenance for one resolved config value."""

    path: str
    source: ConfigSource
    source_ref: str | None = None
    value_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedSessionSpec:
    """Resolved session composition/config plus provenance."""

    scenario: str | None
    extensions: tuple[ExtensionSpec, ...]
    atom_config: Mapping[str, Mapping[str, JsonValue]] = field(default_factory=dict)
    provider: ExtensionSpec | None = None
    provider_identity: ProviderSessionIdentity | None = None
    value_provenance: tuple[ConfigValueProvenance, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class SessionSpecResolver(Protocol):
    """Host-owned resolver for scenario, user config, env, and overrides."""

    def resolve(self, request: AgentSessionConfig) -> ResolvedSessionSpec: ...


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Propagating identity + config context for the session graph.

    Every child session gets a derived context with updated identity.
    Atoms read ``api.ctx`` to know who they are and where they sit in
    the graph.
    """

    session_id: str = ""
    root_session_id: str = ""
    parent_session_id: str | None = None
    depth: int = 0
    cwd: str = ""
    purpose: str = "root"
    scenario: str | None = None
    scenario_dir: str | None = None

    def child(
        self,
        *,
        session_id: str,
        purpose: str = "subagent",
        cwd: str | None = None,
        scenario: str | None = None,
        scenario_dir: str | None = None,
    ) -> SessionContext:
        """Derive a child context with inherited + overridden fields."""
        return SessionContext(
            session_id=session_id,
            root_session_id=self.root_session_id,
            parent_session_id=self.session_id,
            depth=self.depth + 1,
            cwd=self.cwd if cwd is None else cwd,
            purpose=purpose,
            scenario=self.scenario if scenario is None else scenario,
            scenario_dir=scenario_dir
            if scenario_dir is not None
            else self.scenario_dir,
        )


@runtime_checkable
class AtomAPI(Protocol):
    """The complete surface atoms interact with.

    Atoms receive this at ``install(api, config)`` time.  Every method
    is typed — no ``Any`` in the signatures.
    """

    # --- Context (identity + graph position) ---------------------------------

    @property
    def ctx(self) -> SessionContext:
        """Session identity, depth, cwd, purpose, scenario."""
        ...

    # --- Bus (event subscribe + emit) ----------------------------------------

    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = 500,
    ) -> Unsubscribe:
        """Subscribe to a bus channel.  Returns unsubscribe function."""
        ...

    @property
    def bus(self) -> EventBus:
        """Direct bus access for emit/emit_sync."""
        ...

    # --- Tool / Policy registration ------------------------------------------

    def register_tool(self, tool: Tool) -> None: ...

    def register_context_policy(
        self, policy: ContextPolicy, *, priority: int = 500
    ) -> None: ...

    def register_trigger_renderer(
        self, source: str, renderer: TriggerRenderer
    ) -> None: ...

    def register_trigger_codec(self, source: str, codec: TriggerCodec) -> None:
        """Register a codec for custom trigger persistence."""
        ...

    def register_operations(
        self,
        *,
        replace: bool = False,
        service_scope: ServiceScope = "session",
        **kwargs: object,
    ) -> None:
        """Register named operation services, such as ``bash``."""
        ...

    def register_resource_writer(
        self,
        writer: ResourceWriter,
        *,
        replace: bool = False,
    ) -> None:
        """Register the write-side resource boundary for mutating file tools."""
        ...

    def get_resource_writer(self) -> ResourceWriter | None:
        """Return the resource boundary, when the host or an atom registered one."""
        ...

    def register_resource_reader(
        self,
        reader: ResourceReader,
        *,
        replace: bool = False,
    ) -> None:
        """Register the read-side ResourceRef dereference boundary."""
        ...

    def get_resource_reader(self) -> ResourceReader | None:
        """Return the ResourceRef read boundary, when registered."""
        ...

    def register_resource_store(
        self,
        store: ResourceStore,
        *,
        replace: bool = False,
    ) -> None:
        """Register durable logical ResourceRef read/write storage."""
        ...

    def get_resource_store(self) -> ResourceStore | None:
        """Return durable logical ResourceRef storage, when registered."""
        ...

    def get_resource_txn(self) -> ResourceTxn | None:
        """Return the active turn-scoped resource transaction, when present."""
        ...

    def register_tool_executor(
        self,
        executor: ToolExecutor,
        *,
        replace: bool = False,
    ) -> None:
        """Register the execution boundary for tool calls."""
        ...

    def get_tool_executor(self) -> ToolExecutor | None:
        """Return the tool execution boundary, when registered."""
        ...

    def register_tool_orchestrator(
        self,
        orchestrator: ToolOrchestrator,
        *,
        replace: bool = False,
    ) -> None:
        """Register the batch scheduling boundary for tool calls."""
        ...

    def get_tool_orchestrator(self) -> ToolOrchestrator | None:
        """Return the batch scheduling boundary, when registered."""
        ...

    def register_permission_policy(
        self,
        policy: PermissionPolicy,
        *,
        replace: bool = False,
    ) -> None:
        """Register the permission decision boundary for tool calls."""
        ...

    def get_permission_policy(self) -> PermissionPolicy | None:
        """Return the permission policy, when registered."""
        ...

    def register_effect_scope(
        self,
        scope: EffectScope,
        *,
        replace: bool = False,
    ) -> None:
        """Register the lifecycle boundary for turn/fork/resume effects."""
        ...

    def get_effect_scope(self) -> EffectScope | None:
        """Return the effect lifecycle boundary, when registered."""
        ...

    def register_versioned_resource_store(
        self,
        store: VersionedResourceStore,
        *,
        replace: bool = False,
    ) -> None:
        """Register the immutable SDK resource version store."""
        ...

    def get_versioned_resource_store(self) -> VersionedResourceStore | None:
        """Return the versioned SDK resource store, when registered."""
        ...

    def register_atom_catalog(
        self,
        catalog: AtomCatalog,
        *,
        replace: bool = False,
    ) -> None:
        """Register the active-set catalog for resolved atoms."""
        ...

    def get_atom_catalog(self) -> AtomCatalog | None:
        """Return the atom active-set catalog, when registered."""
        ...

    def register_provider(
        self,
        name: str,
        config: ProviderConfig,
        *,
        replace: bool = False,
    ) -> None:
        """Register or replace an LLM provider."""
        ...

    def has_provider(self, name: str) -> bool:
        """Return True when a provider name is already registered."""
        ...

    def get_provider(self, name: str | None = None) -> ProviderConfig | None:
        """Return the named provider, or the active provider when omitted."""
        ...

    # --- Trigger (unified input) ---------------------------------------------

    def push_trigger(
        self,
        trigger: Trigger,
        *,
        priority: TriggerPriority = "next",
        target_session_id: str | None = None,
        target_agent_id: str | None = None,
        origin: str | None = None,
        mode: str = "prompt",
        is_meta: bool = False,
        skip_commands: bool = False,
        meta: dict[str, JsonValue] | None = None,
    ) -> object:
        """Push a trigger into the session's queue."""
        ...

    def interrupt(self, reason: CancelReason | str = "user_cancel") -> None:
        """Request cooperative cancellation of the session's active work."""
        ...

    def track_background(self) -> AbstractContextManager[None]:
        """Bracket a background unit so idle() waits for it."""
        ...

    # --- Session data (read-only) --------------------------------------------

    def get_messages(self) -> list[AgentMessage]:
        """Message list from committed turns (sync, no policies)."""
        ...

    def get_turns(self) -> Sequence[Turn]:
        """Committed turns (read-only)."""
        ...

    @property
    def store(self) -> TrajectoryStore | None:
        """Durable trajectory store, when the host configured persistence."""
        ...

    # --- Services (typed DI) -------------------------------------------------

    @property
    def services(self) -> ServiceRegistry: ...

    # --- Child session -------------------------------------------------------

    async def spawn(
        self,
        *,
        purpose: str = "subagent",
        tools: list[Tool] | None = None,
        system: str | None = None,
        model: Model | None = None,
        scenario: str | None = None,
        cwd: str | None = None,
        max_turns: int | None = None,
        extra_services: ServiceRegistry | None = None,
        cancel_signal: CancelSignal | None = None,
        parent_cancellation: ChildCancellationMode = "inherit",
    ) -> "SpawnedSession":
        """Spawn a lightweight child inheriting parent's config.

        Only override what you need — everything else inherits from
        the parent (tools, model, system, policies, graph, store).
        """
        ...

    async def spawn_child_session(
        self,
        config: AgentSessionConfig,
    ) -> "SpawnedSession":
        """Spawn a fully-constructed child session from config.

        Goes through the factory pipeline: resolves scenario, loads
        extensions, installs atoms.  Use this when the child needs a
        different scenario or extension set from the parent.
        """
        ...

    # --- Model access --------------------------------------------------------

    @property
    def model(self) -> Model | None: ...

    @property
    def experiment(self) -> dict[str, JsonValue] | None: ...


@runtime_checkable
class SpawnedSession(Protocol):
    """Handle to a spawned child session."""

    @property
    def session_id(self) -> str: ...

    async def prompt(
        self,
        text: str,
        *,
        priority: TriggerPriority = "next",
        origin: str | None = "human",
        mode: str = "prompt",
    ) -> object: ...

    async def run(self, text: str) -> list[AgentMessage]:
        """Start, prompt, wait, return messages (blocking convenience)."""
        ...

    def push_trigger(
        self,
        trigger: Trigger,
        *,
        priority: TriggerPriority = "next",
        target_session_id: str | None = None,
        target_agent_id: str | None = None,
        origin: str | None = None,
        mode: str = "prompt",
        is_meta: bool = False,
        skip_commands: bool = False,
        meta: dict[str, JsonValue] | None = None,
    ) -> object: ...

    def interrupt(self, reason: CancelReason | str = "user_cancel") -> None: ...

    def compact(self) -> None: ...

    async def idle(self, timeout: float | None = None) -> bool: ...

    async def shutdown(self) -> None: ...

    def get_messages(self) -> list[AgentMessage]: ...

    def status(self) -> dict[str, str | int | list[str]]: ...


__all__ = [
    "AgentSessionConfig",
    "AtomAPI",
    "ConfigSource",
    "ChildCancellationMode",
    "ConfigValueProvenance",
    "ExtensionInput",
    "ExtensionSource",
    "ExtensionSourceKind",
    "ExtensionSpec",
    "LoopConfig",
    "ResolvedSessionSpec",
    "ScenarioLoader",
    "ScenarioSpec",
    "SESSION_CONFIG_PRECEDENCE",
    "SessionSpecResolver",
    "SessionContext",
    "SpawnedSession",
    "Unsubscribe",
    "normalize_extension_spec",
]
