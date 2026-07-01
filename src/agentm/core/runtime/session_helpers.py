"""Private helpers for :class:`AgentSession`."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import AgentMessage, EventBus, LoopConfig, TerminationCause
from agentm.core.abi.events import DiagnosticEvent, EntryAppendedEvent
from agentm.core.runtime.extension import ExtensionLoadError, ProviderConfig, ReadonlySession
from agentm.core.runtime.session_manager import SessionEntry, SessionManager


@dataclass(frozen=True, slots=True)
class AtomSource:
    """Auto-discovery source description.

    Folds the previously-triplicated builtin/contrib/user iteration
    in ``session_factory._resolve_extensions`` into a single loop. Each
    source pairs a human-readable ``label`` (used in diagnostic
    messages and as the ``source`` field on ``DiagnosticEvent``) with
    a zero-arg ``discover`` callable returning a manifest dict, and an
    optional ``skip_label`` prefix used to format "skipped …" messages
    when an entry's manifest demands config the auto-discovery path
    cannot supply.
    """

    label: str
    discover: Callable[[], dict[str, Any]]
    skip_label: str = ""


class SessionView:
    def __init__(
        self,
        sm: SessionManager,
        *,
        loop_config_getter: Callable[[], LoopConfig],
        bus: EventBus | None = None,
    ) -> None:
        self._sm = sm
        self._loop_config_getter = loop_config_getter
        # Bus is optional so the embedded ``ReadonlySession`` constructed
        # in tests / non-session contexts (e.g. session_factory pre-bus
        # construction) keeps working. Production sessions always pass
        # their event bus through so ``EntryAppendedEvent`` fires.
        self._bus = bus

    def get_messages(self) -> list[AgentMessage]:
        return self._sm.get_messages()

    def get_branch(self) -> list[SessionEntry]:
        return self._sm.get_active_branch()

    def get_leaf_id(self) -> str | None:
        return self._sm.get_leaf_id()

    def get_session_id(self) -> str:
        return self._sm.get_session_id()

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        return self._sm.get_entry(entry_id)

    def get_loop_config(self) -> LoopConfig:
        return self._loop_config_getter()

    def append_entry(
        self,
        type: str,
        payload: Any,
        parent_id: str | None = None,
    ) -> str:
        if parent_id is None:
            branch = self._sm.get_active_branch()
            parent_id = branch[-1].id if branch else None
        entry = SessionEntry(
            type=type,
            id=uuid.uuid4().hex,
            parent_id=parent_id,
            timestamp=now(),
            payload=payload,
        )
        self._sm.append(entry)
        # Fire AFTER the entry is durably appended — handler crashes
        # cannot corrupt session state. ``emit_sync`` because
        # ``append_entry`` is a sync API and we don't want to force the
        # caller into an async context.
        if self._bus is not None:
            self._bus.emit_sync(
                EntryAppendedEvent.CHANNEL,
                EntryAppendedEvent(
                    session_id=self._sm.get_session_id(),
                    entry_type=type,
                    entry_id=entry.id,
                    parent_id=parent_id,
                    payload=payload,
                ),
            )
        return entry.id


ReadonlySessionView: type[ReadonlySession] = SessionView


def now() -> float:
    return time.time()


def collect_system_replacement(returns: list[Any]) -> str | None:
    """Extract system prompt from return-dict pattern.

    .. deprecated::
        Backward-compatibility fallback only. Handlers should mutate
        ``event.system`` in place instead of returning
        ``{"system": "..."}``. The runtime checks ``event.system`` first
        and falls back to this helper only when no handler has set it via
        mutation. Will be removed once all handlers are migrated.
    """
    chosen: str | None = None
    for value in returns:
        if isinstance(value, dict) and value.get("system") is not None:
            candidate = value["system"]
            if isinstance(candidate, str):
                chosen = candidate
    return chosen


def collect_start_veto(returns: list[Any]) -> TerminationCause | None:
    """Extract veto cause from return-dict pattern.

    .. deprecated::
        Backward-compatibility fallback only. Handlers should set
        ``event.veto = <TerminationCause>`` instead of returning
        ``{"block": True, "cause": ...}``. The runtime checks
        ``event.veto`` first and falls back to this helper only when no
        handler has set it via the typed field. Will be removed once all
        handlers are migrated.
    """
    for value in returns:
        if not isinstance(value, dict) or value.get("block") is not True:
            continue
        cause = value.get("cause")
        if isinstance(cause, TerminationCause):
            return cause
    return None


async def collect_auto_discovered_atoms(
    *,
    bus: EventBus,
    sources: Iterable[AtomSource],
) -> list[tuple[str, dict[str, Any]]]:
    """Run every source's discovery callable and collect the loadable
    atoms in order. Each source is independent: a failure in one only
    skips its own entries while still running the rest.
    """
    selected: list[tuple[str, dict[str, Any]]] = []
    for source in sources:
        try:
            discovered = source.discover()
        except Exception as exc:  # noqa: BLE001
            await bus.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="error",
                    source="auto_discovery",
                    message=f"{source.label} atom discovery failed: {exc}",
                ),
            )
            continue
        for entry in discovered.values():
            if atom_requires_unsupplied_config(entry.manifest, {}):
                missing = missing_required_fields(entry.manifest, {})
                await bus.emit(
                    DiagnosticEvent.CHANNEL,
                    DiagnosticEvent(
                        level="info",
                        source="auto_discovery",
                        message=(
                            f"skipped {source.skip_label}{entry.name}: requires "
                            f"config keys {missing!r}; load via --scenario "
                            "or explicit extensions= list"
                        ),
                    ),
                )
                continue
            selected.append((entry.module_path, {}))
    return selected


def ensure_floor_atom(
    entries: list[tuple[str, dict[str, Any]]], module_path: str
) -> None:
    if not any(existing == module_path for existing, _ in entries):
        entries.insert(0, (module_path, {}))


def resolve_provider_config(
    providers: dict[str, ProviderConfig],
    resolver: Any,
    *,
    provider_path: str,
) -> ProviderConfig:
    active_name = resolver.resolve_provider(providers)
    if active_name is None or active_name not in providers:
        raise ExtensionLoadError(
            provider_path,
            RuntimeError(
                f"provider resolver selected unknown provider {active_name!r}"
            ),
        )
    return providers[active_name]


def atom_requires_unsupplied_config(manifest: Any, supplied: dict[str, Any]) -> bool:
    return bool(missing_required_fields(manifest, supplied))


def missing_required_fields(manifest: Any, supplied: dict[str, Any]) -> list[str]:
    schema = getattr(manifest, "config_schema", None)
    if not isinstance(schema, dict):
        return []
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    return [str(key) for key in required if key not in supplied]
