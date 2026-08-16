# code-health: ignore-file[AM025] -- runtime composes plugin, service, and trajectory boundary values
"""Extension loader — validate an atom and install it through AtomAPI."""

# code-health: ignore-file[AM022] -- validates dynamically imported Python plugin contracts

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
from contextlib import suppress
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType
from typing import Literal, TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ValidationError as PydanticValidationError

from agentm.core.abi.bus import (
    EventBus,
    EventBusObserver,
    EventReducer,
    Handler,
)
from agentm.core.abi.effects import EffectBody, EffectHandle
from agentm.core.abi.errors import ExtensionLoadError
from agentm.core.abi.events import ExtensionInstallEvent
from agentm.core.abi.manifest import (
    ExtensionManifest,
    provided_capability_keys,
    requirement_key,
)
from agentm.core.abi.messages import JsonValue, thaw_json
from agentm.core.abi.session_api import (
    AgentSessionConfig,
    AtomAPI,
    ChildCancellationMode,
    ExtensionSource,
    ExtensionSpec,
    SessionContext,
    SpawnedSession,
)
from agentm.core.abi.services import ServiceRegistry, ServiceScope
from agentm.core.abi.store import TrajectoryDiagnostic
from agentm.core.lib.async_cancel import await_known_outcome
from agentm.core.runtime.atom_context import AtomContext
from agentm.extensions.validate import (  # code-health: ignore[AM010] -- constitution-listed contract mechanism
    ValidationIssue,
    extension_helper_imports,
    validate_atom_file,
    validate_atom_package,
)

if TYPE_CHECKING:
    from agentm.core.abi.cancel import CancelReason, CancelSignal
    from agentm.core.abi.codec import TriggerCodec
    from agentm.core.abi.context import ContextPolicy
    from agentm.core.abi.messages import AgentMessage
    from agentm.core.abi.provider import ProviderConfig
    from agentm.core.abi.store import TrajectoryStore
    from agentm.core.abi.stream import Model, StreamFn
    from agentm.core.abi.tool import Tool
    from agentm.core.abi.trajectory import Turn
    from agentm.core.abi.trigger import (
        Trigger,
        TriggerPriority,
        TriggerRenderer,
    )
    from agentm.core.runtime.session import Session


_FILE_EXTENSION_LOAD_LOCK = threading.Lock()

_INACTIVE_API = (
    "atom cannot {action} during installation; defer work to SessionReadyEvent. "
    "A task that reads this after the installation is over is a task of an "
    "installation that failed: the api of a failed install is never activated, "
    "so this will not start working and the task should stop."
)
"""Refusal for the surface an atom may not touch before its ``install`` returns.

One string for the two facades, which is also what keeps the second sentence
true of both: activation happens once, for an installation that got that far,
and nothing clears it -- so an api still refusing is either mid-install or the
residue of one that never landed.
"""


class _AtomEventBusFacade(EventBus):
    """Atom-visible bus surface without lifecycle-control capabilities.

    Subscribing writes into the atom's own context segment; emitting reaches
    the session's bus, which dispatches to every linked segment.  That
    asymmetry is the whole model in one class: a write goes where the object
    is, a read resolves up the chain -- so an atom that has been unlinked can
    still emit, which is what a superseded ``atom_watch`` needs in order to
    report the handover it is in the middle of.
    """

    __slots__ = ("__active", "__context", "__session")

    def __init__(
        self,
        session: "Session",
        context: AtomContext,
        active: Callable[[], bool],
    ) -> None:
        super().__init__()
        self.__session = session
        self.__context = context
        self.__active = active

    def _require_active(self, action: str) -> None:
        if not self.__active():
            raise RuntimeError(_INACTIVE_API.format(action=action))

    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = 500,
        owner: str | None = None,
    ) -> Callable[[], None]:
        # ``owner`` exists on ``EventBus.on`` and is accepted here only so this
        # stays substitutable for one. It is dropped rather than honoured:
        # attribution is which segment the subscription lands in, and this
        # object can reach exactly one.
        del owner
        return self.__context.on(channel, handler, priority=priority)

    def add_observer(
        self,
        observer: EventBusObserver,
        *,
        owner: str | None = None,
    ) -> Callable[[], None]:
        del owner
        return self.__context.add_observer(observer)

    async def emit(self, channel: str, event: Any) -> list[Any]:
        self._require_active("emit events")
        return await self.__session.bus.emit(channel, event)

    async def emit_decision(self, channel: str, event: Any) -> list[Any]:
        self._require_active("emit decision events")
        return await self.__session.bus.emit_decision(channel, event)

    async def emit_reduced(
        self,
        channel: str,
        event: Any,
        reducer: EventReducer,
    ) -> tuple[Any, list[Any]]:
        self._require_active("emit transform events")
        return await self.__session.bus.emit_reduced(channel, event, reducer)

    def emit_sync(self, channel: str, event: Any) -> list[Any]:
        self._require_active("emit events")
        return self.__session.bus.emit_sync(channel, event)

    def freeze_clear(self) -> None:
        raise PermissionError("atoms cannot freeze the session event bus")

    def clear(self) -> None:
        raise PermissionError("atoms cannot clear the session event bus")


class _AtomAPIFacade:
    """The object one installation is handed: its context, plus the session.

    Every write goes to ``self.__context`` -- the atom's own tables -- and
    every read and lifecycle call goes to the session.  There is no owner slot,
    no ``for_owner``, and nothing to revoke: an installation that has been
    unlinked keeps this object and keeps reading through it, while its writes
    land in tables nobody aggregates.

    ``__active`` is the one flag here and it is only about *during* install: an
    atom may not emit, spawn, or push a trigger before its own ``install()``
    has returned.  It is set once and never cleared, so it can never start
    refusing something that used to work.
    """

    __slots__ = ("__active", "__bus", "__context", "__session")

    def __init__(self, session: "Session", context: AtomContext) -> None:
        self.__active = False
        self.__session = session
        self.__context = context
        self.__bus = _AtomEventBusFacade(session, context, self._is_active)

    def _is_active(self) -> bool:
        return self.__active

    def activate(self) -> None:
        self.__active = True

    def _require_active(self, action: str) -> None:
        if not self.__active:
            raise RuntimeError(_INACTIVE_API.format(action=action))

    @property
    def ctx(self) -> SessionContext:
        return self.__session.ctx

    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = 500,
    ) -> Callable[[], None]:
        return self.__context.on(channel, handler, priority=priority)

    @property
    def bus(self) -> EventBus:
        return self.__bus

    def register_tool(self, tool: "Tool") -> None:
        self.__context.register_tool(tool)

    def register_context_policy(
        self,
        policy: "ContextPolicy",
        *,
        priority: int = 500,
    ) -> None:
        self.__context.register_context_policy(policy, priority=priority)

    def register_trigger_renderer(
        self,
        source: str,
        renderer: "TriggerRenderer",
    ) -> None:
        self.__context.register_trigger_renderer(source, renderer)

    def register_trigger_codec(self, source: str, codec: "TriggerCodec") -> None:
        self.__context.register_trigger_codec(source, codec)

    def register_operations(
        self,
        *,
        replace: bool = False,
        service_scope: ServiceScope = "session",
        **kwargs: object,
    ) -> None:
        self.__context.register_operations(
            replace=replace,
            service_scope=service_scope,
            **kwargs,
        )

    def register_provider(
        self,
        name: str,
        config: "ProviderConfig",
        *,
        replace: bool = False,
    ) -> None:
        self.__context.register_provider(name, config, replace=replace)

    def effect(
        self,
        body: EffectBody,
        *,
        provides: str = "",
        retain: str = "",
        compensate: str = "",
        subject: object = None,
    ) -> EffectHandle:
        return self.__context.effect(
            body,
            provides=provides,
            retain=retain,
            compensate=compensate,
            subject=subject,
        )

    async def settle(self) -> None:
        await self.__context.settle()

    def has_provider(self, name: str) -> bool:
        return self.__session.has_provider(name)

    def get_provider(self, name: str | None = None) -> "ProviderConfig | None":
        return self.__session.get_provider(name)

    def push_trigger(
        self,
        trigger: "Trigger",
        *,
        priority: "TriggerPriority" = "next",
        target_session_id: str | None = None,
        target_agent_id: str | None = None,
        origin: str | None = None,
        mode: str = "prompt",
        is_meta: bool = False,
        skip_commands: bool = False,
        meta: dict[str, JsonValue] | None = None,
    ) -> object:
        self._require_active("push triggers")
        return self.__session.push_trigger(
            trigger,
            priority=priority,
            target_session_id=target_session_id,
            target_agent_id=target_agent_id,
            origin=origin,
            mode=mode,
            is_meta=is_meta,
            skip_commands=skip_commands,
            meta=meta,
        )

    def interrupt(self, reason: "CancelReason | str" = "user_cancel") -> None:
        self._require_active("interrupt the session")
        self.__session.interrupt(reason)

    def track_background(self) -> AbstractContextManager[None]:
        self._require_active("start background work")
        return self.__session.track_background()

    def get_messages(self) -> list["AgentMessage"]:
        return self.__session.get_messages()

    def get_turns(self) -> Sequence["Turn"]:
        return self.__session.get_turns()

    @property
    def store(self) -> "TrajectoryStore | None":
        return self.__session.store

    @property
    def services(self) -> ServiceRegistry:
        """This atom's own registry, which resolves up into the session's."""

        return self.__context.services

    async def spawn(
        self,
        *,
        purpose: str = "subagent",
        tools: list["Tool"] | None = None,
        system: str | None = None,
        model: "Model | None" = None,
        stream_fn: "StreamFn | None" = None,
        scenario: str | None = None,
        cwd: str | None = None,
        max_turns: int | None = None,
        extra_services: ServiceRegistry | None = None,
        cancel_signal: "CancelSignal | None" = None,
        parent_cancellation: ChildCancellationMode = "inherit",
    ) -> SpawnedSession:
        self._require_active("spawn child sessions")
        return await self.__session.spawn(
            purpose=purpose,
            tools=tools,
            system=system,
            model=model,
            stream_fn=stream_fn,
            scenario=scenario,
            cwd=cwd,
            max_turns=max_turns,
            extra_services=extra_services,
            cancel_signal=cancel_signal,
            parent_cancellation=parent_cancellation,
        )

    async def spawn_child_session(
        self,
        config: AgentSessionConfig,
    ) -> SpawnedSession:
        self._require_active("spawn child sessions")
        return await self.__session.spawn_child_session(config)

    async def install_extension(
        self,
        extension: ExtensionSpec | str,
        config: dict[str, Any] | None = None,
        *,
        trigger: str = "runtime",
        replace: bool = False,
    ) -> None:
        self._require_active("install extensions")
        await self.__session.install_extension(
            extension, config, trigger=trigger, replace=replace
        )

    def installed_atoms(self) -> tuple[ExtensionSpec, ...]:
        return self.__session.installed_atoms()

    def uninstall_extension(self, atom: ExtensionSpec | str) -> bool:
        self._require_active("uninstall extensions")
        return self.__session.uninstall_extension(atom)

    @property
    def model(self) -> "Model | None":
        return self.__session.model

    @property
    def experiment(self) -> dict[str, JsonValue] | None:
        return self.__session.experiment


async def _record_runtime_install(
    api: "Session",
    *,
    atom_name: str,
    module_path: str,
    trigger: str,
    superseded: str | None,
    error: str | None = None,
) -> None:
    """Note in the trajectory that this session's atom set changed mid-run.

    The active-set fingerprint describes the composition the session started
    with and is not recomputed, so without this a run that gained or replaced
    an atom reads in the record like one that did not. The per-turn tool digest
    shows that something changed; this says what.

    This is the audit record; ``TurnMeta.atom_installs`` is the replay one. Two
    things only this can carry: an install that was refused, and one that
    landed with no turn following it to attach to.
    """

    store = api.store
    if store is None:
        return
    level: Literal["info", "warning", "error"]
    if error is not None:
        phase = "install_failed"
        level = "warning"
        message = f"atom {atom_name!r} was refused by the running session"
    elif superseded is not None:
        phase = "supersede"
        level = "info"
        message = f"atom {atom_name!r} replaced {superseded}"
    else:
        phase = "install"
        level = "info"
        message = f"atom {atom_name!r} installed into the running session"
    diagnostic = TrajectoryDiagnostic(
        id=uuid4().hex,
        session_id=api.id,
        timestamp=time.time(),
        level=level,
        source="extension",
        phase=phase,
        message=message,
        error_type=None if error is None else "ExtensionLoadError",
        error_detail=(
            f"module={module_path} trigger={trigger}"
            if error is None
            else f"module={module_path} trigger={trigger}: {error}"
        ),
        turn_id=None,
        turn_index=None,
        checkpoint_id=None,
    )
    try:
        await await_known_outcome(
            asyncio.to_thread(store.append_diagnostic, diagnostic)
        )
    except Exception as exc:  # noqa: BLE001 - a note must not fail the install
        logger.warning("could not record runtime install of {}: {}", atom_name, exc)


async def install_extension(
    api: "Session",
    extension: ExtensionSpec | str,
    config: dict[str, Any] | None = None,
    *,
    trigger: str = "session_start",
    runtime: bool = False,
    replace: bool = False,
) -> None:
    """Install one extension and emit the standard install lifecycle event."""

    spec = coerce_extension_spec(extension, config)
    module_path = spec.module_path
    started_ns = time.perf_counter_ns()
    name = module_path.rsplit(".", 1)[-1]
    await api.bus.emit(
        ExtensionInstallEvent.CHANNEL,
        ExtensionInstallEvent(
            name=name,
            module_path=module_path,
            phase="start",
            trigger=trigger,
        ),
    )
    error: str | None = None
    superseded: str | None = None
    # Taken before anything moves. The contents of the shared stores only: the
    # context tree is put back by the inverse of what this installation itself
    # linked and unlinked, not by a picture of it.
    # The manifest before the context, because the context's place in the
    # dependency graph is a function of what the manifest declares -- and its
    # rank has to be right from its first write, not from whenever the install
    # finishes.
    manifest = load_manifest_for_spec(spec)
    atom_name = manifest.name if manifest is not None else None
    context = AtomContext(
        api,
        spec,
        runtime=runtime,
        provides=frozenset(
            ()
            if manifest is None
            else provided_capability_keys(
                atom_name=manifest.name,
                registers=manifest.registers,
            )
        ),
        needs=frozenset(
            ()
            if manifest is None
            else (
                requirement_key(entry)
                for entry in (*manifest.requires, *manifest.after)
            )
        ),
    )
    atom_api = _AtomAPIFacade(api, context)
    superseded_departure = None
    try:
        if replace:
            # Unlink even when the module path is unchanged. It is derived from
            # the source digest, so a config-only reload resolves to the same
            # module, and skipping the detach would leave the previous
            # registrations in place for install() to collide with.
            #
            # By manifest name first, because a file atom is loaded under a
            # content-addressed module name and two revisions of one atom share
            # a name and nothing else. By module path when there is no manifest
            # to name: ``replace=True`` means supersede whatever is installed
            # here, and the session refuses a second live context for a path
            # that is.
            found = (
                None if atom_name is None else api.installed_atom_module_path(atom_name)
            )
            if found is None and api.context_for(module_path) is not None:
                found = module_path
            if found is not None:
                superseded = found
                superseded_departure = api.remove_atom_registrations(superseded)
        # Linked before ``install()`` runs, so what the atom writes is part of
        # the session while it is writing it -- the duplicate-tool check, the
        # capability verification and the atom's own reads all depend on that.
        #
        # A replacement takes the position its previous incarnation held, not
        # the end of the list. The installed set's order is what a child
        # replays, so a session that reloaded an atom would otherwise compose
        # its children differently from a cold start -- and reloading is the
        # common path, not the rare one: it is what ``atom_watch`` does every
        # time a scenario file changes.
        context.link_into(
            api,
            after=(
                None
                if superseded_departure is None
                else superseded_departure.preceded_by
            ),
        )
        result = load_extension(spec, atom_api)
        if inspect.isawaitable(result):
            await result
        # The context's own queued bodies, run by the context that queued them.
        # Nothing is handed out or taken back: settling is draining your own
        # log, and a body queued after this one returns is reported by the
        # digest as unsettled rather than silently accepted.
        await context.settle()
        # Here rather than at the end: ``activate`` opens the surface an atom
        # may not touch while it is installing, and everything below awaits.
        # A flag flipped after an await is a window in which a task the atom
        # started is refused something it is entitled to -- which is the shape
        # that has to stay closed. After this line there is no await left
        # between the atom's own code finishing and its api being usable.
        atom_api.activate()
        if runtime:
            await _record_runtime_install(
                api,
                atom_name=atom_name or name,
                module_path=module_path,
                trigger=trigger,
                superseded=superseded,
            )
        if (
            superseded_departure is not None
            and superseded_departure.residue is not None
        ):
            # The replacement landed, so what the previous incarnation held is
            # finally undone rather than merely set aside. Its tables are
            # already out of the session; this runs the writes that had no
            # table -- whatever it recorded through ``api.effect``.
            failures = superseded_departure.residue.revert()
            if failures:
                logger.warning(
                    "superseded atom {} left {} effect(s) that would not undo",
                    superseded,
                    len(failures),
                )
        # Last, and that placement is what the rollback below rests on: nothing
        # after this line can fail, so a context carrying this flag is one whose
        # installation finished. The context has been in the session's list
        # since ``link_into`` -- this does not add it, it says the install
        # landed, which is the difference between an atom the session is
        # running and one it is in the middle of installing.
        context.note_installed(atom_name)
        logger.debug("installed atom: {}", module_path)
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        try:
            # The whole rollback is an inverse now: no picture of the session
            # is taken before an install, because there is no store an install
            # writes into that it does not own. What is undone here is what
            # this installation did and nothing else -- an atom another task
            # detached while this one was awaiting stays detached, an atom that
            # finished installing beside it keeps everything it wrote, and an
            # embedder's own writes from a re-entrant register handler stand.
            #
            # The inverse of ``link_into``, safe to run when the link never
            # happened. Unlinking is also what takes the atom out of the
            # installed set -- the context is the record.
            context.unlink_from(api)
            # Its tables go with the unlink, and what it recorded through
            # ``api.effect`` is undone here because nothing else describes it.
            # An atom that kept its api can go on recording effects through it
            # afterwards -- a body is arbitrary code and runs where it is
            # called -- so the context is noted as departed and whatever it
            # records is undone at shutdown, exactly as a detached atom's is.
            api.note_departed_context(context)
            # ``withdraw`` rather than ``revert``: this atom never arrived, so
            # even the writes it was allowed to leave behind on departure --
            # its trigger codecs -- go back. Nothing can be naming them.
            context.suspend().withdraw()
            if superseded_departure is not None:
                # Nothing to un-revert: the survivor was set aside, not undone,
                # so it is simply given back what it held -- its tables, into
                # the same context object it is still holding the api of, and
                # its place in the session, among the neighbours it had.
                api.restore_atom_registrations(superseded_departure)
        except BaseException as rollback_error:
            raise BaseExceptionGroup(
                f"atom installation and rollback failed: {module_path}",
                (exc, rollback_error),
            ) from exc
        logger.exception("failed to install atom: {}", module_path)
        if runtime:
            with suppress(Exception):
                await _record_runtime_install(
                    api,
                    atom_name=name,
                    module_path=module_path,
                    trigger=trigger,
                    superseded=superseded,
                    error=error,
                )
        raise
    finally:
        await api.bus.emit(
            ExtensionInstallEvent.CHANNEL,
            ExtensionInstallEvent(
                name=name,
                module_path=module_path,
                phase="error" if error is not None else "end",
                duration_ns=time.perf_counter_ns() - started_ns,
                trigger=trigger,
                error=error,
            ),
        )


def load_extension(
    extension: ExtensionSpec | str,
    api: AtomAPI,
    config: dict[str, Any] | None = None,
    *,
    validate: bool = True,
) -> None | Awaitable[None]:
    """Load one extension source and invoke its ``install(api, config)``.

    Returns whatever ``install`` returns:
    - ``None`` for sync extensions (caller need not await).
    - An awaitable for async extensions (caller must await).

    Raises ``ExtensionLoadError`` on any failure.
    """

    spec = coerce_extension_spec(extension, config)
    module_path = spec.module_path
    module = load_extension_module(spec, validate=validate)

    install = module.__dict__.get("install")
    if install is None or not callable(install):
        raise ExtensionLoadError(
            module_path,
            AttributeError(f"module {module_path!r} has no callable 'install' symbol"),
        )

    # Validate config via the schema declared by the atom manifest.
    resolved_config: Any = thaw_json(spec.config)
    manifest = module.__dict__.get("MANIFEST")
    if manifest is not None:
        if not isinstance(manifest, ExtensionManifest):
            raise ExtensionLoadError(
                module_path,
                TypeError("MANIFEST must be an ExtensionManifest"),
            )
        schema_cls = manifest.config_schema
        if isinstance(schema_cls, type) and issubclass(
            schema_cls,
            PydanticBaseModel,
        ):
            try:
                resolved_config = schema_cls.model_validate(resolved_config)
            except PydanticValidationError as exc:
                raise ExtensionLoadError(
                    module_path,
                    ValueError(_format_config_validation_error(schema_cls, exc)),
                ) from exc

    try:
        result = install(api, resolved_config)
    except Exception as exc:
        raise ExtensionLoadError(module_path, exc) from exc
    if not inspect.isawaitable(result):
        return None
    awaitable_result = result

    async def _await_install() -> None:
        try:
            await awaitable_result
        except Exception as exc:
            raise ExtensionLoadError(module_path, exc) from exc

    return _await_install()


def load_manifest_for_spec(
    extension: ExtensionSpec | str,
) -> ExtensionManifest | None:
    """Return the MANIFEST declared by an atom, without installing it."""

    spec = coerce_extension_spec(extension, None)
    module = load_extension_module(spec)
    manifest = module.__dict__.get("MANIFEST")
    if manifest is None:
        return None
    if not isinstance(manifest, ExtensionManifest):
        raise ExtensionLoadError(
            spec.module_path,
            TypeError("MANIFEST must be an ExtensionManifest"),
        )
    return manifest


def coerce_extension_spec(
    extension: ExtensionSpec | str,
    config: dict[str, Any] | None,
) -> ExtensionSpec:
    if isinstance(extension, ExtensionSpec):
        if config is not None:
            raise TypeError(
                "config must be carried by ExtensionSpec when a canonical "
                "extension is provided"
            )
        return extension
    if not isinstance(extension, str) or not extension:
        raise TypeError("extension must be an ExtensionSpec or module string")
    return ExtensionSpec.from_module(extension, config)


def load_extension_module(
    spec: ExtensionSpec,
    *,
    validate: bool = True,
) -> ModuleType:
    """Load and return the module identified by a canonical extension spec."""

    if not isinstance(spec, ExtensionSpec):
        raise TypeError("extension module load requires ExtensionSpec")
    if validate:
        validate_extension_source(spec.source)
    if spec.source.kind == "module":
        try:
            return importlib.import_module(spec.source.location)
        except Exception as exc:
            raise ExtensionLoadError(spec.module_path, exc) from exc
    return _load_file_extension_module(spec.source)


def _load_file_extension_module(source: ExtensionSource) -> ModuleType:
    module_path = source.module_name
    with _FILE_EXTENSION_LOAD_LOCK:
        existing = sys.modules.get(module_path)
        if existing is not None:
            return existing

        path = Path(source.location)
        content = _read_verified_file_source(source)
        try:
            code = compile(content, str(path), "exec")
        except Exception as exc:
            raise ExtensionLoadError(module_path, exc) from exc

        module = ModuleType(module_path)
        module.__file__ = str(path)
        module.__package__ = ""
        module.__loader__ = None
        module.__spec__ = importlib.util.spec_from_loader(
            module_path,
            loader=None,
            origin=str(path),
        )
        module.__dict__["__agentm_source_digest__"] = source.digest
        sys.modules[module_path] = module
        try:
            exec(code, module.__dict__)
        except Exception as exc:
            sys.modules.pop(module_path, None)
            raise ExtensionLoadError(module_path, exc) from exc
        return module


def _read_verified_file_source(source: ExtensionSource) -> bytes:
    path = Path(source.location)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ExtensionLoadError(source.module_name, exc) from exc
    actual = "sha256:" + hashlib.sha256(content).hexdigest()
    if actual != source.digest:
        raise ExtensionLoadError(
            source.module_name,
            RuntimeError(
                f"extension source digest changed: {actual} != {source.digest}"
            ),
        )
    return content


def _format_config_validation_error(
    schema_cls: type[PydanticBaseModel],
    exc: PydanticValidationError,
) -> str:
    errors = exc.errors()
    missing: list[str] = []
    if isinstance(errors, list):
        for error in errors:
            if not isinstance(error, dict) or error.get("type") != "missing":
                continue
            loc = error.get("loc")
            if isinstance(loc, (tuple, list)):
                missing.append(".".join(str(part) for part in loc))
            elif loc:
                missing.append(str(loc))
    if missing:
        return (
            f"config for {schema_cls.__name__} is missing required field(s): "
            + ", ".join(missing)
        )
    return f"config for {schema_cls.__name__} is invalid: {exc}"


def validate_extension_source(source: ExtensionSource | str) -> None:
    """Run AST validation before importing an atom module."""

    if isinstance(source, str):
        source = ExtensionSource(kind="module", location=source)
    if not isinstance(source, ExtensionSource):
        raise TypeError(
            "extension validation requires ExtensionSource or module string"
        )
    module_path = source.module_name
    issues: list[ValidationIssue]

    if source.kind == "file":
        _read_verified_file_source(source)
        src_file = Path(source.location)
        issues = validate_atom_file(src_file)
        for helper in extension_helper_imports(src_file):
            issues.extend(
                _validate_extension_helper_source(
                    helper,
                    visited={module_path},
                )
            )
        _read_verified_file_source(source)
        _raise_blocking_validation_issues(module_path, issues)
        return

    try:
        module_spec = importlib.util.find_spec(source.location)
    except Exception as exc:
        raise ExtensionLoadError(module_path, exc) from exc
    if module_spec is None:
        raise ExtensionLoadError(
            module_path,
            ModuleNotFoundError(f"cannot resolve extension module {module_path!r}"),
        )

    issues = []
    visited: set[str] = {module_path}
    if module_spec.submodule_search_locations:
        for package_dir in module_spec.submodule_search_locations:
            issues.extend(
                validate_atom_package(package_dir, atom_package=source.location)
            )
    elif module_spec.origin is not None:
        src_file = Path(module_spec.origin)
        if src_file.suffix != ".py":
            return
        if src_file.name == "__init__.py":
            issues = validate_atom_package(
                src_file.parent, atom_package=source.location
            )
        else:
            issues = validate_atom_file(src_file)
            for helper in extension_helper_imports(src_file):
                issues.extend(
                    _validate_extension_helper_source(
                        helper,
                        visited=visited,
                    )
                )
    _raise_blocking_validation_issues(module_path, issues)


def _validate_extension_helper_source(
    module_path: str,
    *,
    visited: set[str],
) -> list[ValidationIssue]:
    if module_path in visited:
        return []
    visited.add(module_path)
    try:
        spec = importlib.util.find_spec(module_path)
    except Exception as exc:
        raise RuntimeError(
            f"extension validator cannot inspect helper {module_path!r}"
        ) from exc
    if spec is None:
        raise ModuleNotFoundError(
            f"extension validator cannot resolve helper {module_path!r}"
        )

    issues: list[ValidationIssue] = []
    if spec.submodule_search_locations:
        for package_dir in spec.submodule_search_locations:
            issues.extend(validate_atom_package(package_dir, atom_package=module_path))
    elif spec.origin is not None:
        src_file = Path(spec.origin)
        if src_file.suffix != ".py":
            return []
        if src_file.name == "__init__.py":
            issues.extend(
                validate_atom_package(src_file.parent, atom_package=module_path)
            )
        else:
            issues.extend(validate_atom_file(src_file))
            for helper in extension_helper_imports(src_file):
                issues.extend(
                    _validate_extension_helper_source(
                        helper,
                        visited=visited,
                    )
                )
    return issues


def _raise_blocking_validation_issues(
    module_path: str,
    issues: list[ValidationIssue],
) -> None:
    blocking = [i for i in issues if i.severity == "error"]
    if not blocking:
        return
    msg = "; ".join(f"[{i.rule}] {i.message}" for i in blocking[:5])
    raise ExtensionLoadError(
        module_path,
        RuntimeError(f"contract violation: {msg}"),
    )


__all__ = [
    "ExtensionLoadError",
    "install_extension",
    "load_extension",
    "load_extension_module",
    "validate_extension_source",
]
