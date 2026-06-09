"""``workflow`` atom — run an LLM-authored Python orchestration script.

See ``.claude/designs/dynamic-workflow.md``. A *dynamic workflow* is a
model-authored script that orchestrates many child agent sessions
deterministically. The model writes the control flow (loops, branches,
fan-out, budget-driven scaling) as code at runtime; this atom runs it as a
single tool call and only the final result returns to the model's context.

The script is plain ``async`` Python, run as a coroutine on the host event
loop with a **curated namespace** that exposes only an orchestration SDK —
bound to *already-existing* APIs:

- ``agent(prompt, *, schema=, scenario=, isolation=, tool_allowlist=, extra_extensions=, atom_config=)`` —
  :meth:`ExtensionAPI.spawn_child_session` → ``child.prompt`` → last
  ``AssistantMessage`` text → ``child.shutdown``. Returns the text. A worker
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

§11 single-file contract: stdlib + ``agentm.core.abi.*`` +
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
import logging
import os
import sys
import textwrap
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, Protocol, TypedDict, TypeVar

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.events import (
    Event,
    EventBus,
    TurnEndEvent,
)
from agentm.core.abi.extension import ExtensionAPI, ExtensionStaleError
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
)
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.extensions import ExtensionManifest

_log = logging.getLogger(__name__)
_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Public type aliases — sharpen signatures beyond bare ``dict[str, Any]``.
# ---------------------------------------------------------------------------

JsonSchema = dict[str, Any]
"""A JSON Schema object (passed to ``schema=`` on ``agent()``)."""

AtomConfigMap = dict[str, dict[str, Any]]
"""Mapping of atom name → atom config dict (passed to ``atom_config=``)."""

ExtensionEntry = tuple[str, dict[str, Any]]
"""``(module_path, config)`` pair for ``extra_extensions``."""

AgentResult = str | dict[str, Any] | list[Any]
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

    ``spawn_child_session`` is typed ``-> Any`` in the ABI (so the §11 import
    allow-list need not pull in ``core.runtime``); this Protocol re-types the
    handful of members we actually use, structurally."""

    bus: EventBus

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
    config_schema={
        "type": "object",
        "properties": {
            "max_concurrency": {"type": "integer", "minimum": 1},
            "max_agents": {"type": "integer", "minimum": 1},
            "wall_clock_timeout_s": {"type": ["number", "null"], "minimum": 0},
            "default_scenario": {"type": ["string", "null"]},
            # Hard token ceiling for one workflow run; backs budget.total /
            # budget.remaining() so a script can scale depth (loop-until-budget).
            # Null ⇒ no ceiling (remaining stays None, spend still tracked).
            "budget_tokens": {"type": ["integer", "null"], "minimum": 1},
        },
        "additionalProperties": False,
    },
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
# ``_WorkflowRun._spawn_and_drive``, not the §11 validator. The module path is
# what goes into the child's extension list; the bare atom name (for the
# loaded-set check) is DERIVED from it rather than written as a literal — a
# bare-name literal would read as an undeclared hard dependency to the D4
# contract check.
_AGENT_ENV_ATOM_MODULE: Final[str] = (
    "agentm.extensions.builtin.operations"
)
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
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "next", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ValueError", "KeyError", "IndexError", "TypeError",
    "RuntimeError", "StopIteration",
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
    _agent_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _bg_tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    # --- SDK injected into the script namespace -------------------------------

    async def agent(
        self,
        prompt: str,
        *,
        schema: JsonSchema | None = None,
        scenario: str | None = None,
        isolation: str | None = None,
        tool_allowlist: list[str] | None = None,
        extra_extensions: list[ExtensionEntry] | None = None,
        atom_config: AtomConfigMap | None = None,
        retry: int = 0,
    ) -> AgentResult:
        """Spawn one child agent session, drive it to completion, return its
        output. Journaled by ``hash(prompt, opts)`` — a re-run resumes from
        cache without re-spawning.

        Return type depends on the agent's output:

        - If the agent ends via a ``ToolTerminate`` finalize tool (e.g.
          ``submit_hop_verdict``, ``submit_result`` from ``schema=``), the
          tool result is returned as a parsed ``dict``/``list``.
        - Otherwise, the last assistant text is returned as ``str``.

        ``schema=`` injects a ``submit_result`` terminal tool shaped by the
        schema — it's one way to get structured output but not the only way.
        Agents with their own finalize tools (via ``ToolTerminate``) return
        structured data without needing ``schema=``.

        ``retry=N`` retries the agent up to N times on failure (empty output
        or ``_error`` result). Failed attempts are not journaled — only the
        final result is recorded."""

        # Exclude retry from the journal key — it's execution policy, not
        # semantic identity.
        opts: dict[str, Any] = {
            "schema": schema,
            "scenario": scenario,
            "isolation": isolation,
            "tool_allowlist": tool_allowlist,
            "extra_extensions": extra_extensions,
            "atom_config": atom_config,
        }
        key = _Journal.key(prompt, opts)
        cached = await self.journal.lookup(key)
        if cached is not None:
            return _auto_parse(cached)

        async with self._agent_lock:
            if self.agents_spawned >= self.max_agents:
                raise RuntimeError(
                    f"workflow agent-count backstop reached "
                    f"({self.max_agents}); aborting runaway script"
                )
            self.agents_spawned += 1

        async with self.semaphore:
            result = await self._spawn_and_drive(
                prompt, scenario, isolation, tool_allowlist,
                extra_extensions=extra_extensions,
                atom_config=atom_config,
                schema=schema,
            )

        parsed = _auto_parse(result)

        if _is_agent_error(parsed) and retry > 0:
            _log.warning(
                "workflow agent failed (prompt=%.60s…), retrying (%d left)",
                prompt, retry,
            )
            self.agents_retried += 1
            return await self.agent(
                prompt,
                schema=schema,
                scenario=scenario,
                isolation=isolation,
                tool_allowlist=tool_allowlist,
                extra_extensions=extra_extensions,
                atom_config=atom_config,
                retry=retry - 1,
            )

        if _is_agent_error(parsed):
            self.agents_failed += 1
        else:
            self.agents_succeeded += 1

        await self.journal.record(key, result)
        return parsed

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

        config = AgentSessionConfig(
            cwd=self.api.cwd,
            provider=None,  # inherit parent's provider (spawn auto-wires it)
            # Default the worker to the orchestrator's own scenario (a clean
            # reload of that curated atom set) rather than letting scenario=None
            # auto-discover ALL builtins — which would hand every worker the
            # full heavy toolset (bash/install_atom/…) and, on strict providers,
            # break tool-call grammar generation. Explicit arg > atom config >
            # parent scenario.
            scenario=scenario or self.default_scenario or self.api.scenario,
            extra_extensions=extensions,
            atom_config_overrides=atom_config_overrides,
            tool_allowlist=tool_allowlist,
            purpose=_WORKER_PURPOSE,
        )
        try:
            child: _ChildSession = await self.api.spawn_child_session(config)
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
            _log.warning(
                "workflow agent spawn/prompt failed: %s: %s",
                type(exc).__name__, exc,
            )
            return json.dumps({
                "_error": type(exc).__name__,
                "detail": str(exc)[:500],
            })


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

    async def agent(
        self,
        prompt: str,
        *,
        schema: JsonSchema | None = None,
        scenario: str | None = None,
        isolation: str | None = None,
        tool_allowlist: list[str] | None = None,
        extra_extensions: list[ExtensionEntry] | None = None,
        atom_config: AtomConfigMap | None = None,
        retry: int = 0,
    ) -> AgentResult:
        """Spawn one child agent session and return its output.

        Returns ``dict``/``list`` when the agent has a finalize tool or
        ``schema=`` is set; ``str`` otherwise.

        ``retry=N`` retries the agent up to N times on failure.
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

    _log.warning(
        "workflow agent produced no output — "
        "it may have exhausted its tool budget without calling a finalize tool"
    )
    return ""


def _auto_parse(text: str) -> AgentResult:
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
_SDK_NAMES: Final[frozenset[str]] = frozenset({
    "agent", "parallel", "pipeline", "budget", "args", "json", "log", "phase",
})
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


_FORBIDDEN_CALLS: Final[frozenset[str]] = frozenset({
    "open", "__import__", "eval", "exec", "compile", "input", "getattr",
})


def _validate_exec(tree: ast.Module) -> list[ScriptIssue]:
    issues: list[ScriptIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            issues.append(ScriptIssue(
                node.lineno,
                "import not available in exec mode — use module mode "
                "(define `async def run(ctx: WorkflowContext)`) for imports",
                "error",
            ))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                issues.append(ScriptIssue(
                    node.lineno,
                    f"'{node.func.id}()' is not available in the "
                    f"workflow namespace",
                    "error",
                ))
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ("time", "random"):
                issues.append(ScriptIssue(
                    node.lineno,
                    f"'{node.value.id}' is not available — workflow "
                    f"scripts must be deterministic for resume/journal",
                    "error",
                ))
    return issues


def _validate_module(tree: ast.Module) -> list[ScriptIssue]:
    issues: list[ScriptIssue] = []
    run_func: ast.AsyncFunctionDef | None = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
            run_func = node
            break
    if run_func is None:
        issues.append(ScriptIssue(
            0, "module mode requires `async def run(ctx: WorkflowContext)`",
            "error",
        ))
        return issues
    params = run_func.args
    if not params.args and not params.posonlyargs:
        issues.append(ScriptIssue(
            run_func.lineno,
            "`run()` must accept a WorkflowContext parameter "
            "(e.g. `async def run(ctx: WorkflowContext)`)",
            "error",
        ))
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
                k for k in sys.modules
                if k == pkg_root or k.startswith(pkg_root + ".")
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
                spent=0, input_tokens=0, output_tokens=0,
                total=None, remaining=None,
            )
        return self._last_run.budget_svc.snapshot()

    @property
    def last_run_summary(self) -> RunSummary:
        if self._last_run is None:
            return RunSummary(
                agents_spawned=0, agents_succeeded=0,
                agents_failed=0, agents_retried=0,
                budget=BudgetSnapshot(
                    spent=0, input_tokens=0, output_tokens=0,
                    total=None, remaining=None,
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

    async def run_file(
        self,
        path: str | Path,
        args: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Run a pre-written workflow script file."""
        sp = Path(path)
        if not sp.is_absolute():
            sp = Path(self._api.cwd) / sp
        sp = sp.resolve()
        if not sp.is_file():
            raise FileNotFoundError(f"workflow script not found: {sp}")
        script = sp.read_text(encoding="utf-8")
        return await self._execute(script, args or {}, source_path=sp)

    async def run_script(
        self,
        script: str,
        args: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Run an inline workflow script string."""
        if not script.strip():
            raise ValueError("workflow: empty script")
        return await self._execute(script, args or {})

    async def _execute(
        self,
        script: str,
        args_payload: dict[str, Any],
        *,
        source_path: Path | None = None,
    ) -> WorkflowResult:
        # --- pre-execution validation ---
        mode = _detect_script_mode(script)
        issues = _validate_script(script, mode)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise WorkflowValidationError(errors)
        for w in issues:
            if w.severity == "warning":
                _log.warning("workflow line %d: %s", w.line, w.message)

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

        return _auto_parse(_coerce_result(raw))


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    if api.purpose == _WORKER_PURPOSE:
        return

    max_concurrency = config.get("max_concurrency")
    concurrency = (
        int(max_concurrency)
        if isinstance(max_concurrency, int) and max_concurrency > 0
        else _default_concurrency()
    )
    max_agents_cfg = config.get("max_agents")
    max_agents = (
        int(max_agents_cfg)
        if isinstance(max_agents_cfg, int) and max_agents_cfg > 0
        else _AGENT_COUNT_BACKSTOP
    )
    wall_clock = config.get("wall_clock_timeout_s")
    wall_clock_timeout: float | None = (
        float(wall_clock)
        if isinstance(wall_clock, (int, float)) and wall_clock > 0
        else None
    )
    default_scenario_cfg = config.get("default_scenario")
    default_scenario = (
        default_scenario_cfg if isinstance(default_scenario_cfg, str) else None
    )
    budget_tokens_cfg = config.get("budget_tokens")
    budget_tokens: int | None = (
        int(budget_tokens_cfg)
        if isinstance(budget_tokens_cfg, int) and budget_tokens_cfg > 0
        else None
    )

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
            return _error(
                "workflow: 'script' and 'script_path' are mutually exclusive"
            )

        # Capture progress events so the agent can observe the execution
        # trajectory (phases, log lines) — not just the final result.
        progress: list[dict[str, str]] = []

        async def _capture_phase(event: WorkflowPhaseEvent) -> None:
            progress.append({"kind": event.kind, "text": event.text})

        unsub = api.events.on(WorkflowPhaseEvent.CHANNEL, _capture_phase)

        try:
            if isinstance(script_path_raw, str) and script_path_raw.strip():
                result = await runner.run_file(
                    script_path_raw, args.get("args"),
                )
            elif isinstance(script, str) and script.strip():
                result = await runner.run_script(script, args.get("args"))
            else:
                return _error(
                    "workflow: either 'script' or 'script_path' is required"
                )
        except WorkflowValidationError as exc:
            return _error(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return _error(f"workflow: {exc}")
        except asyncio.TimeoutError:
            return _error(
                f"workflow: script exceeded wall-clock budget "
                f"({wall_clock_timeout}s)"
            )
        except Exception as exc:
            return _error(
                f"workflow script error: {type(exc).__name__}: {exc}"
            )
        finally:
            unsub()

        return ToolResult(
            content=[TextContent(
                type="text", text=_coerce_result(result),
            )],
            extras={
                "progress": progress,
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
