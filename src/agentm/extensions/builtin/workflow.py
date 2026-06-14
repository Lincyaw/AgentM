"""``workflow`` atom — run an LLM-authored Python orchestration script.

See ``.claude/designs/dynamic-workflow.md``. A *dynamic workflow* is a
model-authored script that orchestrates many child agent sessions
deterministically. The model writes the control flow (loops, branches,
fan-out, budget-driven scaling) as code at runtime; this atom runs it as a
single tool call and only the final result returns to the model's context.

The script is plain ``async`` Python, run as a coroutine on the host event
loop with a **curated namespace** that exposes only an orchestration SDK —
bound to *already-existing* APIs:

- ``agent(prompt, *, schema=, scenario=, isolation=, tool_allowlist=, extra_extensions=, atom_config=, retry=, timeout=)`` —
  :meth:`ExtensionAPI.spawn_child_session` → ``child.prompt`` → last
  ``AssistantMessage`` text → ``child.shutdown``. ``schema=PydanticModel``
  returns a validated model instance (auto-retry on validation failure).
  ``timeout=`` caps wall-clock seconds. A worker
  defaults to the **orchestrator's own scenario** (a clean reload of that
  curated atom set), not an auto-discovery of every builtin — so workers are
  slim and never carry the ``workflow`` tool themselves (anti-recursion is also
  enforced at install time via the ``purpose`` marker).
- ``parallel(aws)`` — ``await parallel([agent(p) for p in ...])``;
  ``asyncio.gather`` over the awaitables (each ``agent`` self-limits via a
  shared ``Semaphore``).
- ``pipeline(items, *stages)`` — run each item through ``stages`` (sync or
  async callables) independently, **no cross-item barrier**: item A may be in
  stage 2 while item B is still in stage 1.
- ``budget`` — a read-only view: ``budget.total`` / ``budget.spent()`` /
  ``budget.remaining()`` over aggregated child token spend.
- ``args`` — the caller-supplied ``args`` dict.
- ``log(msg)`` / ``phase(name)`` — fire-and-forget ``WorkflowPhaseEvent`` on
  the parent bus.

**Capability surface, not a sandbox.** The script runs at the *same authority*
as the agent that wrote it (which can already call ``bash`` / ``install_atom``
etc.), so this is not a security boundary. The curated ``__builtins__`` (data
ops only — no ``open`` / ``__import__`` / ``eval``) is a guardrail that keeps
honest scripts on the SDK and off the filesystem, not an escape-proof wall;
hard isolation, when needed, is the worker ``operations`` (agent_env backend) sandbox or
a process boundary — the same answer as for ``bash``. Determinism for the
resume journal is achieved by *not injecting* ``time`` / ``random`` rather than
by interception.

single-file contract: stdlib + ``agentm.core.abi.*`` +
``agentm.extensions.*``. No atom-to-atom imports — ``artifact_store`` is
reached via ``api.get_service`` and worker child sessions via
``api.spawn_child_session``.
"""

from __future__ import annotations

import ast
import asyncio
import builtins as _builtins
import contextlib
import hashlib
import importlib.util
import inspect
import json
from loguru import logger
import os
import sys
import textwrap
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, Protocol, TypedDict, TypeVar, overload

from pydantic import BaseModel, ValidationError

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    Event,
    EventBus,
    ExtensionAPI,
    ExtensionStaleError,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolResultMessage,
    TurnEndEvent,
)
from agentm.extensions import ExtensionManifest

_T = TypeVar("_T")
_M = TypeVar("_M", bound=BaseModel)

# ---------------------------------------------------------------------------
# Public type aliases — sharpen signatures beyond bare ``dict[str, Any]``.
# ---------------------------------------------------------------------------

JsonSchema = dict[str, Any]
"""A JSON Schema object (passed to ``schema=`` on ``agent()``)."""

AtomConfigMap = dict[str, dict[str, Any]]
"""Mapping of atom name → atom config dict (passed to ``atom_config=``)."""

ExtensionEntry = tuple[str, dict[str, Any]]
"""``(module_path, config)`` pair for ``extra_extensions``."""

AgentResult = str | dict[str, Any] | list[Any] | BaseModel
"""Return type of ``agent()`` — ``dict``/``list`` for structured output,
``str`` for free-text."""

WorkflowResult = str | dict[str, Any] | list[Any]
"""Return type of a workflow run."""

# ``Any`` is used deliberately only for genuinely arbitrary script values
# (the ``args`` payload, decoded-JSON tool params, the pipeline item / result
# types). Every host-side object reached through ``ExtensionAPI`` is given a
# real type or a minimal structural ``Protocol`` below.


class _ChildSession(Protocol):
    """The slice of a spawned child session this atom drives.

    ``spawn_child_session`` is typed ``-> Any`` in the ABI (so the import
    allow-list need not pull in ``core.runtime``); this Protocol re-types the
    handful of members we actually use, structurally."""

    bus: EventBus
    session_id: str

    async def prompt(self, message: str) -> list[AgentMessage]: ...
    async def shutdown(self) -> None: ...


class _ArtifactStore(Protocol):
    """The ``artifact_store`` service surface used for the resume journal."""

    async def list_artifacts(self, args: dict[str, Any]) -> ToolResult: ...
    async def read(self, args: dict[str, Any]) -> ToolResult: ...
    async def write_artifact(
        self,
        *,
        kind: str,
        title: str,
        body: str,
        tags: list[str] | None = None,
    ) -> dict[str, str]: ...


class BudgetSnapshot(TypedDict):
    """Aggregated child token spend (also returned in the tool ``extras``)."""

    spent: int
    input_tokens: int
    output_tokens: int
    total: int | None
    remaining: int | None


class RunSummary(TypedDict):
    """Post-execution summary of a workflow run."""

    agents_spawned: int
    agents_succeeded: int
    agents_failed: int
    agents_retried: int
    budget: BudgetSnapshot
    wall_clock_s: float


class WorkflowConfig(BaseModel):
    max_concurrency: int | None = None
    max_agents: int | None = None
    wall_clock_timeout_s: float | None = None
    default_scenario: str | None = None
    budget_tokens: int | None = None


MANIFEST = ExtensionManifest(
    name="workflow",
    description=(
        "Run an LLM-authored async Python orchestration script in a curated "
        "namespace that fans out deterministic child agent sessions via the "
        "agent/parallel/pipeline primitives. Only the final result returns to "
        "the model's context."
    ),
    registers=(
        "tool:workflow",
        "event:workflow_phase",
    ),
    config_schema=WorkflowConfig,
    # The workflow-local journal (resume) is backed by the artifact_store
    # service; the atom reaches it via api.get_service, never an import.
    requires=("artifact_store",),
    conflicts=(),
)

# Name of the service that backs the workflow-local resume journal.
_ARTIFACT_STORE_SERVICE: Final[str] = "artifact_store"

# ``purpose`` stamped on every worker child session. Doubles as the
# anti-recursion marker: a session with this purpose does not register the
# ``workflow`` tool (a worker cannot spawn its own workflow), so even when a
# worker auto-discovers all builtins it never gets a recursive workflow tool.
_WORKER_PURPOSE: Final[str] = "workflow"

# Lifetime backstop: a runaway script cannot spawn more than this many
# child sessions across the whole workflow run.
_AGENT_COUNT_BACKSTOP: Final[int] = 1000

# Worker-isolation atom (``isolation="agent_env"``). Soft/optional dependency:
# NOT in MANIFEST.requires (that would force every workflow user to load
# agent-env + the arl extra). Enforcement is the runtime availability guard in
# ``_WorkflowRun._spawn_and_drive``, not the validator. The module path is
# what goes into the child's extension list; the bare atom name (for the
# loaded-set check) is DERIVED from it rather than written as a literal — a
# bare-name literal would read as an undeclared hard dependency to the D4
# contract check.
_AGENT_ENV_ATOM_MODULE: Final[str] = "agentm.extensions.builtin.operations"
_AGENT_ENV_ATOM: Final[str] = _AGENT_ENV_ATOM_MODULE.rsplit(".", 1)[-1]

# structured_output atom: wired automatically when agent(schema=...) is used.
_STRUCTURED_OUTPUT_ATOM_MODULE: Final[str] = (
    "agentm.extensions.builtin.structured_output"
)
_STRUCTURED_OUTPUT_ATOM: Final[str] = _STRUCTURED_OUTPUT_ATOM_MODULE.rsplit(".", 1)[-1]

# Curated builtins handed to the script: data-manipulation only. This is a
# guardrail (keeps honest scripts on the SDK + off the filesystem), NOT an
# escape-proof sandbox — see the module docstring. ``open`` / ``__import__`` /
# ``eval`` / ``exec`` / ``compile`` / ``input`` / ``getattr`` etc. are omitted.
# ``time`` / ``random`` are likewise not provided, keeping the journal key
# replay-deterministic.
_SAFE_BUILTIN_NAMES: Final[tuple[str, ...]] = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "int",
    "isinstance",
    "issubclass",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "Exception",
    "ValueError",
    "KeyError",
    "IndexError",
    "TypeError",
    "RuntimeError",
    "StopIteration",
)
_SAFE_BUILTINS: Final[dict[str, Any]] = {
    name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES
}


@dataclass(slots=True)
class WorkflowPhaseEvent(Event):
    """A workflow surfacing a named phase or a free-text log line.

    Emitted on the parent session bus so a TUI / observer can follow the
    orchestration. Captured generically by the observability observer; not in
    ``_TO_OTEL_CHANNELS`` so it surfaces as a generic dispatch record, not a
    first-class span (see dynamic-workflow.md / the OTLP note).
    """

    CHANNEL: ClassVar[Literal["workflow_phase"]] = "workflow_phase"
    kind: Literal["phase", "log"] = "log"
    text: str = ""


@dataclass(slots=True)
class _BudgetService:
    """Cross-child token-spend aggregator.

    ``cost_budget`` keys spend in per-install local state — there is no
    cross-child aggregation, and child ``TurnEndEvent``s do **not** bubble to
    the parent bus. So this service subscribes directly on each child's bus
    (the child session object exposes ``.bus``) right after spawn, and sums
    ``TurnEndEvent.message.usage`` across every child of the workflow run.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_budget_tokens: int | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def attach(self, child: _ChildSession) -> None:
        """Subscribe to ``turn_end`` on the child's own bus.

        Accesses ``child.bus`` directly — the same child-session contract
        ``sub_agent`` relies on. We do **not** defensively swallow a missing
        bus: a configured ``budget_tokens`` ceiling backs ``loop-until-budget``
        scripts, so a silently-zero aggregator (ceiling never trips, spend
        runs away) is worse than failing loudly if the child contract breaks.
        """

        async def _on_turn_end(event: TurnEndEvent) -> None:
            usage = getattr(event.message, "usage", None)
            if usage is None:
                return
            async with self._lock:
                self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        child.bus.on(TurnEndEvent.CHANNEL, _on_turn_end)

    def snapshot(self) -> BudgetSnapshot:
        spent = self.input_tokens + self.output_tokens
        remaining: int | None = None
        if self.total_budget_tokens is not None:
            remaining = max(0, self.total_budget_tokens - spent)
        return {
            "spent": spent,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total": self.total_budget_tokens,
            "remaining": remaining,
        }


@dataclass(frozen=True, slots=True)
class _Budget:
    """Read-only ``budget`` view injected into the script."""

    _svc: _BudgetService

    @property
    def total(self) -> int | None:
        return self._svc.total_budget_tokens

    def spent(self) -> int:
        return self._svc.input_tokens + self._svc.output_tokens

    def remaining(self) -> int | None:
        total = self._svc.total_budget_tokens
        if total is None:
            return None
        return max(0, total - self.spent())


@dataclass(slots=True)
class _Journal:
    """Workflow-local journal backed by the ``artifact_store`` service.

    Each ``agent()`` result is keyed by ``hash(prompt, opts)`` and written as
    an artifact (``kind="workflow_journal"``, the key carried in ``tags`` and
    ``title``). On resume the host looks the key up first and returns the
    cached body without re-spawning. The store allocates its own ids, so the
    key->artifact mapping lives in the tags index, not the id.

    *Not* ``SessionStore.open`` — that is session-level transcript replay, the
    wrong granularity (design §3.3).
    """

    store: _ArtifactStore | None
    _cache: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def key(prompt: str, opts: dict[str, Any]) -> str:
        payload = json.dumps(
            {"prompt": prompt, "opts": opts},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    async def lookup(self, key: str) -> str | None:
        if key in self._cache:
            return self._cache[key]
        if self.store is None:
            return None
        listing = await self.store.list_artifacts(
            {"kind": "workflow_journal", "tags": [key], "limit": 1}
        )
        artifacts = (
            (listing.extras or {}).get("artifacts", []) if listing.extras else []
        )
        if not artifacts:
            return None
        artifact_id = artifacts[0].get("id")
        if not artifact_id:
            return None
        read = await self.store.read({"artifact_id": artifact_id})
        if read.is_error or not read.extras:
            return None
        body = read.extras.get("body")
        if isinstance(body, str):
            self._cache[key] = body
            return body
        return None

    async def record(self, key: str, body: str) -> None:
        self._cache[key] = body
        if self.store is None:
            return
        await self.store.write_artifact(
            kind="workflow_journal",
            title=key,
            body=body,
            tags=[key],
        )


def _default_concurrency() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(16, cpu - 2))


@dataclass(slots=True)
class _WorkflowRun:
    """One ``workflow`` tool invocation. Owns the journal, budget aggregator,
    concurrency cap and lifetime backstop, and exposes the orchestration SDK
    (``agent`` / ``parallel`` / ``pipeline`` / ``log`` / ``phase``) that gets
    injected — bound to this run — into the script namespace."""

    api: ExtensionAPI
    journal: _Journal
    budget_svc: _BudgetService
    semaphore: asyncio.Semaphore
    default_scenario: str | None = None
    max_agents: int = _AGENT_COUNT_BACKSTOP
    agents_spawned: int = 0
    agents_succeeded: int = 0
    agents_failed: int = 0
    agents_retried: int = 0
    args_payload: dict[str, Any] = field(default_factory=dict)
    cwd_override: str | None = None
    provider_override: tuple[str, dict[str, Any]] | None = None
    progress: list[dict[str, str]] = field(default_factory=list)
    child_sessions: list[dict[str, Any]] = field(default_factory=list)
    _agent_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _bg_tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    # --- SDK injected into the script namespace -------------------------------

    async def agent(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | JsonSchema | None = None,
        scenario: str | None = None,
        isolation: str | None = None,
        tool_allowlist: list[str] | None = None,
        extra_extensions: list[ExtensionEntry] | None = None,
        atom_config: AtomConfigMap | None = None,
        retry: int = 0,
        timeout: float | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        """Spawn one child agent session, drive it to completion, return its
        output. Journaled by ``hash(prompt, opts)`` — a re-run resumes from
        cache without re-spawning.

        ``schema=`` accepts a Pydantic BaseModel subclass. The atom converts
        it to a JSON Schema, injects a ``submit_result`` terminal tool into
        the child, validates the result with ``model_validate``, and returns
        the validated model instance. On validation failure the agent is
        re-prompted with the error (counts against ``retry``).

        ``retry=N`` retries on agent failure (empty output, ``_error``) AND
        on structured-output validation failure. Failed attempts are not
        journaled — only the final result is recorded.

        ``timeout=`` (seconds) caps wall-clock time for the agent call.
        ``TimeoutError`` is raised on expiry (counts as a retryable failure).

        ``session_id=`` resumes an existing session instead of starting
        fresh. The prompt is appended to the session's history. The resumed
        session retains its full prior context.
        """

        # Resolve Pydantic model → JSON Schema dict.
        from agentm.core.lib import pydantic_to_tool_schema

        model_cls: type[BaseModel] | None = None
        json_schema: JsonSchema | None = None
        if schema is not None:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                model_cls = schema
                json_schema = pydantic_to_tool_schema(schema)
            else:
                json_schema = schema

        # Exclude retry/timeout from the journal key — execution policy,
        # not semantic identity.
        opts: dict[str, Any] = {
            "schema": json_schema,
            "scenario": scenario,
            "isolation": isolation,
            "tool_allowlist": tool_allowlist,
            "extra_extensions": extra_extensions,
            "atom_config": atom_config,
        }
        key = _Journal.key(prompt, opts)
        cached = await self.journal.lookup(key)
        if cached is not None:
            parsed = _auto_parse(cached)
            if model_cls is not None and isinstance(parsed, dict):
                return model_cls.model_validate(parsed)
            return parsed

        current_prompt = prompt
        last_error: Exception | None = None

        for attempt in range(max(retry, 0) + 1):
            async with self._agent_lock:
                if self.agents_spawned >= self.max_agents:
                    raise RuntimeError(
                        f"workflow agent-count backstop reached "
                        f"({self.max_agents}); aborting runaway script"
                    )
                self.agents_spawned += 1

            try:
                async with self.semaphore:
                    coro = self._spawn_and_drive(
                        current_prompt,
                        scenario,
                        isolation,
                        tool_allowlist,
                        extra_extensions=extra_extensions,
                        atom_config=atom_config,
                        schema=json_schema,
                        session_id=session_id,
                    )
                    if timeout is not None:
                        result = await asyncio.wait_for(coro, timeout=timeout)
                    else:
                        result = await coro
            except TimeoutError as exc:
                last_error = exc
                if attempt < retry:
                    logger.warning(
                        f"workflow agent timed out (prompt={prompt}…), retrying ({retry - attempt} left)"
                    )
                    self.agents_retried += 1
                    current_prompt = _structured_retry_prompt(
                        prompt,
                        exc,
                        attempt + 1,
                    )
                    continue
                self.agents_failed += 1
                raise

            parsed = _auto_parse(result)

            if _is_agent_error(parsed) and attempt < retry:
                logger.warning(
                    f"workflow agent failed (prompt={prompt}…), retrying ({retry - attempt} left)"
                )
                self.agents_retried += 1
                continue

            # Validate against Pydantic model if provided.
            if model_cls is not None and not _is_agent_error(parsed):
                try:
                    validated = model_cls.model_validate(parsed)
                except (ValidationError, TypeError) as exc:
                    last_error = exc
                    if attempt < retry:
                        logger.warning(
                            f"workflow structured output invalid for {model_cls.__name__} (prompt={prompt}…), retrying ({retry - attempt} left)"
                        )
                        self.agents_retried += 1
                        current_prompt = _structured_retry_prompt(
                            prompt,
                            exc,
                            attempt + 1,
                        )
                        continue
                    self.agents_failed += 1
                    raise
                self.agents_succeeded += 1
                await self.journal.record(key, result)
                return validated

            if _is_agent_error(parsed):
                self.agents_failed += 1
            else:
                self.agents_succeeded += 1

            await self.journal.record(key, result)
            return parsed

        assert last_error is not None
        raise last_error

    async def parallel(self, aws: list[Awaitable[_T]]) -> list[_T]:
        """Barrier fan-out: await every awaitable, return results in order.
        Each ``agent`` self-limits via the shared Semaphore, so this is a thin
        ``gather``."""

        return list(await asyncio.gather(*aws))

    async def pipeline(
        self,
        items: list[Any],
        *stages: Callable[[Any], Any],
    ) -> list[Any]:
        """Run each item through ``stages`` independently — **no cross-item
        barrier**. Item A may be in stage 2 while item B is still in stage 1;
        wall-clock is the slowest single-item chain, not the sum of per-stage
        maxima. Stages may be sync or async callables."""

        async def _chain(item: Any) -> Any:
            cur: Any = item
            for stage in stages:
                cur = stage(cur)
                if inspect.isawaitable(cur):
                    cur = await cur
            return cur

        return list(await asyncio.gather(*(_chain(it) for it in items)))

    def log(self, msg: object) -> None:
        self._emit_phase("log", str(msg))

    def phase(self, name: object) -> None:
        self._emit_phase("phase", str(name))

    # --- internals -----------------------------------------------------------

    def _emit_phase(self, kind: Literal["phase", "log"], text: str) -> None:
        self.progress.append({"kind": kind, "text": text})
        # Fire-and-forget: schedule the emit on the running loop and hold a
        # reference so it is not GC'd mid-flight.
        with contextlib.suppress(RuntimeError):
            task = asyncio.create_task(
                self.api.events.emit(
                    WorkflowPhaseEvent.CHANNEL,
                    WorkflowPhaseEvent(kind=kind, text=text),
                )
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

    async def _spawn_and_drive(
        self,
        prompt: str,
        scenario: str | None,
        isolation: str | None,
        tool_allowlist: list[str] | None,
        *,
        extra_extensions: list[ExtensionEntry] | None = None,
        atom_config: AtomConfigMap | None = None,
        schema: JsonSchema | None = None,
        session_id: str | None = None,
    ) -> str:
        extensions: list[ExtensionEntry] = []
        atom_config_overrides: AtomConfigMap = dict(atom_config or {})
        # isolation="agent_env" selects the worker sandbox by listing the
        # operations atom (agent_env backend) in the child's extensions (policy by
        # composition, never a privileged config field). Soft, optional
        # dependency — guarded here rather than declared in MANIFEST.requires
        # (which would force every workflow user to load agent-env + the arl
        # extra + a K8s gateway). Fail at the agent() call with a clear message
        # rather than spawning a child with a bogus extension entry.
        if isolation == "agent_env":
            available = {atom.name for atom in self.api.list_atoms()}
            if _AGENT_ENV_ATOM not in available:
                raise RuntimeError(
                    f"workflow: isolation='agent_env' requires the "
                    f"{_AGENT_ENV_ATOM!r} atom to be loaded in this scenario, "
                    f"but it is not present. Add it to the scenario (and "
                    f"install the 'agent-env' extra) or drop the isolation arg."
                )
            extensions.append((_AGENT_ENV_ATOM_MODULE, {}))

        # schema= wires the structured_output atom into the child so the
        # worker gets a submit_result terminal tool shaped by the schema.
        if schema is not None:
            extensions.append((_STRUCTURED_OUTPUT_ATOM_MODULE, {}))
            atom_config_overrides[_STRUCTURED_OUTPUT_ATOM] = {"schema": schema}

        # User-supplied extra_extensions are appended last so they layer on
        # top of isolation and schema atoms.
        if extra_extensions:
            extensions.extend(extra_extensions)

        session_manager: Any = None
        if session_id is not None:
            from agentm.core.abi import SESSION_STORE_SERVICE

            store = self.api.get_service(SESSION_STORE_SERVICE)
            if store is None:
                raise RuntimeError(
                    "workflow: session_id requires the session_store service, "
                    "but it is not available in this session"
                )
            session_manager = store.open(session_id)

        config = AgentSessionConfig(
            cwd=self.cwd_override or self.api.cwd,
            # Resolved upfront via MODEL_RESOLVER_SERVICE; None = inherit parent's provider.
            provider=self.provider_override,
            scenario=scenario or self.default_scenario or self.api.scenario,
            extra_extensions=extensions,
            atom_config_overrides=atom_config_overrides,
            tool_allowlist=tool_allowlist,
            purpose=_WORKER_PURPOSE,
            session_manager=session_manager,
            session_id=session_id,
        )
        try:
            child: _ChildSession = await self.api.spawn_child_session(config)
            self.child_sessions.append({
                "session_id": child.session_id,
                "scenario": scenario or self.default_scenario,
                "prompt": prompt[:200],
            })
            self.budget_svc.attach(child)
            try:
                messages = await child.prompt(prompt)
                output = _final_session_output(messages)
                if not output:
                    return json.dumps(_build_agent_error_info(messages))
                return output
            finally:
                with contextlib.suppress(Exception):
                    await child.shutdown()
        except Exception as exc:
            logger.warning(
                f"workflow agent spawn/prompt failed: {type(exc).__name__}: {exc}"
            )
            return json.dumps(
                {
                    "_error": type(exc).__name__,
                    "detail": str(exc)[:500],
                }
            )


class WorkflowContext:
    """Typed interface for module-mode workflow scripts.

    Module scripts define ``async def run(ctx: WorkflowContext)`` and use
    ``ctx.agent()``, ``ctx.parallel()``, etc. — identical semantics to the
    exec-mode globals but with full IDE / type-checker support.

    Import path::

        from agentm.extensions.builtin.workflow import WorkflowContext
    """

    __slots__ = ("_run",)

    def __init__(self, run: _WorkflowRun) -> None:
        self._run = run

    @property
    def args(self) -> dict[str, Any]:
        """Caller-supplied ``args`` payload."""
        return self._run.args_payload

    @property
    def budget(self) -> _Budget:
        """Read-only token-spend view."""
        return _Budget(self._run.budget_svc)

    @overload
    async def agent(
        self,
        prompt: str,
        *,
        schema: type[_M],
        scenario: str | None = ...,
        isolation: str | None = ...,
        tool_allowlist: list[str] | None = ...,
        extra_extensions: list[ExtensionEntry] | None = ...,
        atom_config: AtomConfigMap | None = ...,
        retry: int = ...,
        timeout: float | None = ...,
        session_id: str | None = ...,
    ) -> _M: ...

    @overload
    async def agent(
        self,
        prompt: str,
        *,
        schema: None = ...,
        scenario: str | None = ...,
        isolation: str | None = ...,
        tool_allowlist: list[str] | None = ...,
        extra_extensions: list[ExtensionEntry] | None = ...,
        atom_config: AtomConfigMap | None = ...,
        retry: int = ...,
        timeout: float | None = ...,
        session_id: str | None = ...,
    ) -> AgentResult: ...

    async def agent(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | JsonSchema | None = None,
        scenario: str | None = None,
        isolation: str | None = None,
        tool_allowlist: list[str] | None = None,
        extra_extensions: list[ExtensionEntry] | None = None,
        atom_config: AtomConfigMap | None = None,
        retry: int = 0,
        timeout: float | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        """Spawn one child agent session and return its output.

        ``schema=PydanticModel`` returns a validated model instance.
        ``retry=N`` retries on agent failure or validation failure.
        ``timeout=`` caps wall-clock seconds for the agent call.
        ``session_id=`` resumes an existing session with the prompt.
        """
        return await self._run.agent(
            prompt,
            schema=schema,
            scenario=scenario,
            isolation=isolation,
            tool_allowlist=tool_allowlist,
            extra_extensions=extra_extensions,
            atom_config=atom_config,
            retry=retry,
            timeout=timeout,
            session_id=session_id,
        )

    async def parallel(self, aws: list[Awaitable[_T]]) -> list[_T]:
        """Barrier fan-out: await every awaitable, return results in order."""
        return await self._run.parallel(aws)

    async def pipeline(
        self,
        items: list[Any],
        *stages: Callable[[Any], Any],
    ) -> list[Any]:
        """Run each item through ``stages`` independently (no cross-item barrier)."""
        return await self._run.pipeline(items, *stages)

    def log(self, msg: str) -> None:
        """Emit a free-text progress line on the parent session bus."""
        self._run.log(msg)

    def phase(self, name: str) -> None:
        """Start a named phase (groups subsequent agents in the progress view)."""
        self._run.phase(name)


# ---------------------------------------------------------------------------
# Output extraction / error helpers
# ---------------------------------------------------------------------------


def _build_agent_error_info(messages: list[AgentMessage]) -> dict[str, Any]:
    """Build a structured error dict when a child agent produces no output."""
    info: dict[str, Any] = {"_error": "no_output", "turns": len(messages)}
    for msg in reversed(messages):
        if isinstance(msg, ToolResultMessage):
            for result_block in reversed(msg.content):
                if result_block.is_error:
                    for inner in reversed(result_block.content):
                        text = getattr(inner, "text", None)
                        if isinstance(text, str) and text:
                            info["last_tool_error"] = text[:500]
                            return info
    return info


def _structured_retry_prompt(original: str, exc: Exception, attempt: int) -> str:
    return (
        f"{original}\n\n"
        f"## Structured-output retry #{attempt}\n"
        f"Your previous response did not produce a valid structured result.\n"
        f"Error: {str(exc)[:2000]}\n\n"
        f"You MUST call the submit_result tool exactly once with a valid "
        f"`result` object conforming to the required JSON schema."
    )


def _is_agent_error(result: AgentResult) -> bool:
    """True if the result represents an agent failure."""
    if isinstance(result, dict) and "_error" in result:
        return True
    if isinstance(result, str) and not result:
        return True
    return False


def _final_session_output(messages: list[AgentMessage]) -> str:
    """Extract the agent's final output from a completed session.

    Scans backward for the last meaningful output:

    1. If the session's last ``ToolResultMessage`` contains a
       non-error result, return that text — this is the output from
       a ``ToolTerminate``-style finalize tool (e.g.
       ``submit_hop_verdict``, ``submit_final_report``).
    2. Otherwise fall back to the last ``AssistantMessage`` text block.
    3. If neither is found, return empty string and log a warning
       (the agent likely hit its budget without submitting a result).
    """

    # Walk backward — first non-error tool result or assistant text wins.
    for msg in reversed(messages):
        if isinstance(msg, ToolResultMessage):
            for result_block in reversed(msg.content):
                if result_block.is_error:
                    continue
                for inner in reversed(result_block.content):
                    text = getattr(inner, "text", None)
                    if isinstance(text, str) and text:
                        return text
        elif isinstance(msg, AssistantMessage):
            for block in reversed(msg.content):
                text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    return text

    logger.warning(
        "workflow agent produced no output — "
        "it may have exhausted its tool budget without calling a finalize tool"
    )
    return ""


def _auto_parse(text: str) -> str | dict[str, Any] | list[Any]:
    """Parse JSON if possible, otherwise return raw string.

    Every ToolTerminate finalize tool (``submit_hop_verdict``,
    ``submit_result`` from ``schema=``, etc.) returns JSON.  Auto-parsing
    means workflow scripts never need ``json.loads`` — they get a dict
    for structured agents and a string for free-text agents.
    """
    if not text:
        return text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return text


# ---------------------------------------------------------------------------
# Pre-execution validation
# ---------------------------------------------------------------------------

# Names available in the exec-mode curated namespace.
_SDK_NAMES: Final[frozenset[str]] = frozenset(
    {
        "agent",
        "parallel",
        "pipeline",
        "budget",
        "args",
        "json",
        "log",
        "phase",
    }
)
_EXEC_KNOWN_NAMES: Final[frozenset[str]] = frozenset(_SAFE_BUILTIN_NAMES) | _SDK_NAMES


@dataclass(frozen=True, slots=True)
class ScriptIssue:
    """One validation finding: ``error`` blocks execution, ``warning`` is advisory."""

    line: int
    message: str
    severity: Literal["error", "warning"]


class WorkflowValidationError(Exception):
    """Raised when pre-execution validation finds blocking issues."""

    def __init__(self, issues: list[ScriptIssue]) -> None:
        self.issues = issues
        detail = "\n".join(f"  line {i.line}: {i.message}" for i in issues)
        super().__init__(f"workflow script validation failed:\n{detail}")


def _detect_script_mode(source: str) -> Literal["exec", "module"]:
    """``module`` if the source defines a top-level ``async def run(...)``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "exec"
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
            return "module"
    return "exec"


def _validate_script(
    source: str,
    mode: Literal["exec", "module"],
) -> list[ScriptIssue]:
    """AST-level static checks before execution.

    Returns a list of issues. Errors block execution; warnings are logged
    and the script runs anyway. The caller (tool handler or programmatic
    API) surfaces errors so the agent / developer can fix the script.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [ScriptIssue(exc.lineno or 0, f"SyntaxError: {exc.msg}", "error")]

    if mode == "exec":
        return _validate_exec(tree)
    return _validate_module(tree)


_FORBIDDEN_CALLS: Final[frozenset[str]] = frozenset(
    {
        "open",
        "__import__",
        "eval",
        "exec",
        "compile",
        "input",
        "getattr",
    }
)


def _validate_exec(tree: ast.Module) -> list[ScriptIssue]:
    issues: list[ScriptIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            issues.append(
                ScriptIssue(
                    node.lineno,
                    "import not available in exec mode — use module mode "
                    "(define `async def run(ctx: WorkflowContext)`) for imports",
                    "error",
                )
            )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                issues.append(
                    ScriptIssue(
                        node.lineno,
                        f"'{node.func.id}()' is not available in the "
                        f"workflow namespace",
                        "error",
                    )
                )
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ("time", "random"):
                issues.append(
                    ScriptIssue(
                        node.lineno,
                        f"'{node.value.id}' is not available — workflow "
                        f"scripts must be deterministic for resume/journal",
                        "error",
                    )
                )
    return issues


def _validate_module(tree: ast.Module) -> list[ScriptIssue]:
    issues: list[ScriptIssue] = []
    run_func: ast.AsyncFunctionDef | None = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
            run_func = node
            break
    if run_func is None:
        issues.append(
            ScriptIssue(
                0,
                "module mode requires `async def run(ctx: WorkflowContext)`",
                "error",
            )
        )
        return issues
    params = run_func.args
    if not params.args and not params.posonlyargs:
        issues.append(
            ScriptIssue(
                run_func.lineno,
                "`run()` must accept a WorkflowContext parameter "
                "(e.g. `async def run(ctx: WorkflowContext)`)",
                "error",
            )
        )
    return issues


# ---------------------------------------------------------------------------
# Script execution — exec mode (inline / LLM-generated)
# ---------------------------------------------------------------------------


def _error(message: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=message)],
        is_error=True,
    )


def _coerce_result(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if raw is None:
        return ""
    try:
        return json.dumps(raw, ensure_ascii=False, default=str)
    except TypeError:
        return str(raw)


def _build_namespace(run: _WorkflowRun) -> dict[str, Any]:
    """The curated globals the script runs against — only the SDK + safe
    builtins. No ``import``, no ``open``, no ``time`` / ``random``."""

    return {
        "__builtins__": _SAFE_BUILTINS,
        "agent": run.agent,
        "parallel": run.parallel,
        "pipeline": run.pipeline,
        "budget": _Budget(run.budget_svc),
        "args": run.args_payload,
        "json": json,
        "log": run.log,
        "phase": run.phase,
    }


async def _run_user_script(script: str, ns: dict[str, Any]) -> object:
    """Compile the script as the body of an ``async def`` and await it, so the
    script can ``await agent(...)`` / ``parallel(...)`` directly. ``return`` in
    the script becomes the workflow result."""

    src = "async def __workflow__():\n" + textwrap.indent(script, "    ")
    code = compile(src, "<workflow-script>", "exec")
    exec(code, ns)  # noqa: S102 - curated namespace; same authority as the agent
    return await ns["__workflow__"]()


# ---------------------------------------------------------------------------
# Script execution — module mode (developer-written file scripts)
# ---------------------------------------------------------------------------


async def _run_module_script(source_path: Path, run: _WorkflowRun) -> object:
    """Import a file script as a Python module and call ``run(ctx)``.

    Module mode gives the script full import/filesystem access (same
    authority as a regular Python module). IDE type-checkers see the
    ``WorkflowContext`` signature and provide completions + diagnostics.

    When the script lives inside a package (``__init__.py`` chain), we walk
    up to find the package root, add its parent to ``sys.path``, and import
    via the fully qualified dotted name so relative imports work.
    """
    ctx = WorkflowContext(run)

    # Walk up from the script's directory to find the package root — the
    # topmost ancestor that still has __init__.py.
    parts = [source_path.stem]
    current = source_path.parent
    while (current / "__init__.py").is_file():
        parts.append(current.name)
        current = current.parent
    # `current` is the directory to put on sys.path (package root's parent).
    # `parts` reversed gives the fully qualified module name.
    parts.reverse()

    if len(parts) > 1:
        # Script is inside a package — use normal import machinery.
        module_name = ".".join(parts)
        path_entry = str(current)
        added = path_entry not in sys.path
        if added:
            sys.path.insert(0, path_entry)
        try:
            pkg_root = parts[0]
            module = importlib.import_module(module_name)
            run_fn = getattr(module, "run", None)
            if run_fn is None or not callable(run_fn):
                raise RuntimeError(
                    "module-mode script must define "
                    "`async def run(ctx: WorkflowContext)`"
                )
            return await run_fn(ctx)
        finally:
            if added:
                with contextlib.suppress(ValueError):
                    sys.path.remove(path_entry)
            stale = [
                k for k in sys.modules if k == pkg_root or k.startswith(pkg_root + ".")
            ]
            for k in stale:
                sys.modules.pop(k, None)
    else:
        # Standalone file — load by path with a unique module name.
        module_name = (
            f"_agentm_workflow_"
            f"{hashlib.sha256(str(source_path).encode()).hexdigest()[:12]}"
        )
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load workflow module from {source_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            run_fn = getattr(module, "run", None)
            if run_fn is None or not callable(run_fn):
                raise RuntimeError(
                    "module-mode script must define "
                    "`async def run(ctx: WorkflowContext)`"
                )
            return await run_fn(ctx)
        finally:
            sys.modules.pop(module_name, None)


_WORKFLOW_TOOL_PARAMS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "script": {
            "type": "string",
            "description": (
                "Async Python orchestration script. Available names: "
                "agent(prompt, *, schema=, scenario=, isolation=, "
                "tool_allowlist=, extra_extensions=, atom_config=, retry=0) "
                "(awaitable -> str | dict when schema set; retry=N retries on failure), "
                "parallel(list_of_awaitables) -> list, "
                "pipeline(items, *stages) -> list, budget (.total / .spent() / "
                ".remaining()), args (dict), json (module), log(msg), "
                "phase(name). Use await for agent / parallel / pipeline. "
                "`return <value>` becomes the tool result. Only "
                "data-manipulation builtins are available — "
                "no import / open / time / random. "
                "Mutually exclusive with script_path."
            ),
        },
        "script_path": {
            "type": "string",
            "description": (
                "Path to a pre-written async Python workflow script file. "
                "Mutually exclusive with script. The file is read and executed "
                "in the same curated namespace as an inline script."
            ),
        },
        "args": {
            "type": "object",
            "description": "Arbitrary JSON payload exposed to the script as args.",
            "additionalProperties": True,
        },
        "cwd": {
            "type": "string",
            "description": (
                "Working directory for child agent sessions spawned by "
                "the workflow. Defaults to the caller's cwd."
            ),
        },
        "model": {
            "type": "string",
            "description": (
                "Model name or config.toml profile for child agent "
                "sessions. Defaults to the caller's model."
            ),
        },
    },
    "additionalProperties": False,
}

_WORKFLOW_RUNNER_SERVICE: Final[str] = "workflow_runner"


class WorkflowRunner:
    """Programmatic entry point for running workflow scripts.

    Registered as the ``workflow_runner`` service by the workflow atom.
    External code (eval harnesses, CLI tools) can obtain it via
    ``session.get_service("workflow_runner")`` and call ``run_file`` /
    ``run_script`` without going through the tool interface.

    Return value semantics match ``agent()``: structured output (from
    ToolTerminate finalize tools) is auto-parsed to ``dict``/``list``;
    plain text is returned as ``str``.
    """

    def __init__(
        self,
        api: ExtensionAPI,
        *,
        concurrency: int,
        max_agents: int,
        wall_clock_timeout: float | None,
        default_scenario: str | None,
        budget_tokens: int | None,
    ) -> None:
        self._api = api
        self._concurrency = concurrency
        self._max_agents = max_agents
        self._wall_clock_timeout = wall_clock_timeout
        self._default_scenario = default_scenario
        self._budget_tokens = budget_tokens
        self._last_run: _WorkflowRun | None = None
        self._last_wall_clock_s: float = 0.0

    @property
    def last_agents_spawned(self) -> int:
        return self._last_run.agents_spawned if self._last_run else 0

    @property
    def last_budget_snapshot(self) -> BudgetSnapshot:
        if self._last_run is None:
            return BudgetSnapshot(
                spent=0,
                input_tokens=0,
                output_tokens=0,
                total=None,
                remaining=None,
            )
        return self._last_run.budget_svc.snapshot()

    @property
    def last_progress(self) -> list[dict[str, str]]:
        return self._last_run.progress if self._last_run else []

    @property
    def last_run_summary(self) -> RunSummary:
        if self._last_run is None:
            return RunSummary(
                agents_spawned=0,
                agents_succeeded=0,
                agents_failed=0,
                agents_retried=0,
                budget=BudgetSnapshot(
                    spent=0,
                    input_tokens=0,
                    output_tokens=0,
                    total=None,
                    remaining=None,
                ),
                wall_clock_s=0.0,
            )
        return RunSummary(
            agents_spawned=self._last_run.agents_spawned,
            agents_succeeded=self._last_run.agents_succeeded,
            agents_failed=self._last_run.agents_failed,
            agents_retried=self._last_run.agents_retried,
            budget=self._last_run.budget_svc.snapshot(),
            wall_clock_s=self._last_wall_clock_s,
        )

    async def _emit_delivery(
        self,
        run: _WorkflowRun,
        result: WorkflowResult,
        *,
        source_path: Path | None,
        args_payload: dict[str, Any],
    ) -> None:
        """Persist a ``workflow_delivery`` artifact with execution metadata.

        Best-effort: failures are logged, never raised."""
        store: _ArtifactStore | None = self._api.get_service(
            _ARTIFACT_STORE_SERVICE,
        )
        if store is None:
            return

        success: bool | None = None
        if isinstance(result, dict):
            success = result.get("success")

        summary = self.last_run_summary
        delivery: dict[str, Any] = {
            "script": str(source_path) if source_path else None,
            "trace_id": self._api.root_session_id,
            "session_id": self._api.session_id,
            "success": success,
            "execution": {
                "wall_clock_s": summary["wall_clock_s"],
                "agents_spawned": summary["agents_spawned"],
                "agents_succeeded": summary["agents_succeeded"],
                "agents_failed": summary["agents_failed"],
                "tokens": summary["budget"],
            },
            "progress": run.progress,
            "child_sessions": run.child_sessions,
            "args": args_payload,
            "result": result,
        }

        title = source_path.stem if source_path else "inline"
        tags = ["workflow_delivery"]
        if source_path:
            tags.append(source_path.stem)
        if isinstance(success, bool):
            tags.append("success" if success else "failure")

        try:
            await store.write_artifact(
                kind="workflow_delivery",
                title=f"delivery:{title}",
                body=json.dumps(delivery, ensure_ascii=False, default=str),
                tags=tags,
            )
        except Exception:
            logger.opt(exception=True).debug("workflow_delivery artifact write failed")

    async def run_file(
        self,
        path: str | Path,
        args: dict[str, Any] | None = None,
        *,
        cwd: str | None = None,
        provider: tuple[str, dict[str, Any]] | None = None,
    ) -> WorkflowResult:
        """Run a pre-written workflow script file."""
        sp = Path(path)
        if not sp.is_absolute():
            sp = Path(self._api.cwd) / sp
        sp = sp.resolve()
        if not sp.is_file():
            raise FileNotFoundError(f"workflow script not found: {sp}")
        script = sp.read_text(encoding="utf-8")
        return await self._execute(
            script,
            args or {},
            source_path=sp,
            cwd=cwd,
            provider=provider,
        )

    async def run_script(
        self,
        script: str,
        args: dict[str, Any] | None = None,
        *,
        cwd: str | None = None,
        provider: tuple[str, dict[str, Any]] | None = None,
    ) -> WorkflowResult:
        """Run an inline workflow script string."""
        if not script.strip():
            raise ValueError("workflow: empty script")
        return await self._execute(
            script,
            args or {},
            cwd=cwd,
            provider=provider,
        )

    async def _execute(
        self,
        script: str,
        args_payload: dict[str, Any],
        *,
        source_path: Path | None = None,
        cwd: str | None = None,
        provider: tuple[str, dict[str, Any]] | None = None,
    ) -> WorkflowResult:
        # --- pre-execution validation ---
        mode = _detect_script_mode(script)
        issues = _validate_script(script, mode)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise WorkflowValidationError(errors)
        for w in issues:
            if w.severity == "warning":
                logger.warning(f"workflow line {w.line}: {w.message}")

        self._last_run = run = _WorkflowRun(
            api=self._api,
            journal=_Journal(
                store=self._api.get_service(_ARTIFACT_STORE_SERVICE),
            ),
            budget_svc=_BudgetService(
                total_budget_tokens=self._budget_tokens,
            ),
            semaphore=asyncio.Semaphore(self._concurrency),
            default_scenario=self._default_scenario,
            max_agents=self._max_agents,
            args_payload=dict(args_payload),
            cwd_override=cwd,
            provider_override=provider,
        )

        try:
            bracket = self._api.track_background()
        except ExtensionStaleError:
            bracket = contextlib.nullcontext()

        t0 = time.monotonic()
        with bracket:
            if mode == "module" and source_path is not None:
                coro = _run_module_script(source_path, run)
            else:
                coro = _run_user_script(script, _build_namespace(run))

            if self._wall_clock_timeout is not None:
                raw = await asyncio.wait_for(coro, timeout=self._wall_clock_timeout)
            else:
                raw = await coro
        self._last_wall_clock_s = time.monotonic() - t0

        result = _auto_parse(_coerce_result(raw))

        await self._emit_delivery(
            run,
            result,
            source_path=source_path,
            args_payload=args_payload,
        )

        return result


def install(api: ExtensionAPI, config: WorkflowConfig) -> None:
    if api.purpose == _WORKER_PURPOSE:
        return

    concurrency = (
        config.max_concurrency
        if config.max_concurrency is not None and config.max_concurrency > 0
        else _default_concurrency()
    )
    max_agents = (
        config.max_agents
        if config.max_agents is not None and config.max_agents > 0
        else _AGENT_COUNT_BACKSTOP
    )
    wall_clock_timeout: float | None = (
        config.wall_clock_timeout_s
        if config.wall_clock_timeout_s is not None and config.wall_clock_timeout_s > 0
        else None
    )
    default_scenario = config.default_scenario
    budget_tokens: int | None = (
        config.budget_tokens
        if config.budget_tokens is not None and config.budget_tokens > 0
        else None
    )

    from agentm.core.abi import MODEL_RESOLVER_SERVICE

    model_resolver = api.get_service(MODEL_RESOLVER_SERVICE)

    runner = WorkflowRunner(
        api,
        concurrency=concurrency,
        max_agents=max_agents,
        wall_clock_timeout=wall_clock_timeout,
        default_scenario=default_scenario,
        budget_tokens=budget_tokens,
    )
    api.set_service(_WORKFLOW_RUNNER_SERVICE, runner)

    async def _run_workflow(args: dict[str, Any]) -> ToolResult:
        script = args.get("script")
        script_path_raw = args.get("script_path")

        if script_path_raw and script:
            return _error("workflow: 'script' and 'script_path' are mutually exclusive")

        cwd_arg = args.get("cwd")
        cwd: str | None = (
            str(cwd_arg) if isinstance(cwd_arg, str) and cwd_arg.strip() else None
        )

        model_arg = args.get("model")
        provider: tuple[str, dict[str, Any]] | None = None
        if isinstance(model_arg, str) and model_arg.strip():
            if model_resolver is None:
                return _error(
                    "workflow: model= requires the model_resolver service, "
                    "but it is not available in this session"
                )
            provider = model_resolver(model_arg.strip())
            if provider is None:
                return _error(f"workflow: unknown model profile '{model_arg}'")

        try:
            if isinstance(script_path_raw, str) and script_path_raw.strip():
                result = await runner.run_file(
                    script_path_raw,
                    args.get("args"),
                    cwd=cwd,
                    provider=provider,
                )
            elif isinstance(script, str) and script.strip():
                result = await runner.run_script(
                    script,
                    args.get("args"),
                    cwd=cwd,
                    provider=provider,
                )
            else:
                return _error("workflow: either 'script' or 'script_path' is required")
        except WorkflowValidationError as exc:
            return _error(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return _error(f"workflow: {exc}")
        except asyncio.TimeoutError:
            return _error(
                f"workflow: script exceeded wall-clock budget ({wall_clock_timeout}s)"
            )
        except Exception as exc:
            return _error(f"workflow script error: {type(exc).__name__}: {exc}")

        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=_coerce_result(result),
                )
            ],
            extras={
                "progress": runner.last_progress,
                "summary": runner.last_run_summary,
            },
        )

    api.register_tool(
        FunctionTool(
            name="workflow",
            description=(
                "Run an async Python orchestration script (inline or from a "
                "file via script_path) that fans out deterministic child agent "
                "sessions. Use for parallel verify / judge / sweep harnesses "
                "where you author the control flow as code rather than "
                "turn-by-turn. Names: agent, parallel, pipeline, budget, args, "
                "json, log, phase. The script runs in a curated "
                "namespace (data builtins only; no import / open / time / "
                "random). agent() results are "
                "journaled by hash(prompt, args) so re-running a workflow "
                "resumes from cache. agent(prompt, schema={...}) returns a "
                "parsed dict conforming to the JSON Schema."
            ),
            parameters=_WORKFLOW_TOOL_PARAMS,
            fn=_run_workflow,
            metadata={"workflow": True},
        )
    )
