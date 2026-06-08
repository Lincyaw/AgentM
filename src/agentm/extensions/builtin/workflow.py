"""``workflow`` atom — run an LLM-authored Python orchestration script.

See ``.claude/designs/dynamic-workflow.md``. A *dynamic workflow* is a
model-authored script that orchestrates many child agent sessions
deterministically. The model writes the control flow (loops, branches,
fan-out, budget-driven scaling) as code at runtime; this atom runs it as a
single tool call and only the final result returns to the model's context.

The script is plain ``async`` Python, run as a coroutine on the host event
loop with a **curated namespace** that exposes only an orchestration SDK —
bound to *already-existing* APIs:

- ``agent(prompt, *, scenario=, isolation=, tool_allowlist=)`` —
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
hard isolation, when needed, is the worker ``operations_agent_env`` sandbox or
a process boundary — the same answer as for ``bash``. Determinism for the
resume journal is achieved by *not injecting* ``time`` / ``random`` rather than
by interception.

§11 single-file contract: stdlib + ``agentm.core.abi.*`` +
``agentm.extensions.*``. No atom-to-atom imports — ``artifact_store`` is
reached via ``api.get_service`` and worker child sessions via
``api.spawn_child_session``.
"""

from __future__ import annotations

import asyncio
import builtins as _builtins
import contextlib
import hashlib
import inspect
import json
import os
import textwrap
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Literal, Protocol, TypedDict, TypeVar

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.events import (
    Event,
    EventBus,
    TurnEndEvent,
)
from agentm.core.abi.extension import ExtensionAPI, ExtensionStaleError
from agentm.core.abi.messages import AgentMessage, AssistantMessage
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.extensions import ExtensionManifest

_T = TypeVar("_T")


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
    "agentm.extensions.builtin.operations_agent_env"
)
_AGENT_ENV_ATOM: Final[str] = _AGENT_ENV_ATOM_MODULE.rsplit(".", 1)[-1]


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
    args_payload: dict[str, Any] = field(default_factory=dict)
    _agent_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _bg_tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    # --- SDK injected into the script namespace -------------------------------

    async def agent(
        self,
        prompt: str,
        *,
        scenario: str | None = None,
        isolation: str | None = None,
        tool_allowlist: list[str] | None = None,
    ) -> str:
        """Spawn one child agent session, drive it to a final reply, return its
        text. Journaled by ``hash(prompt, opts)`` — a re-run resumes from cache
        without re-spawning."""

        opts: dict[str, Any] = {
            "scenario": scenario,
            "isolation": isolation,
            "tool_allowlist": tool_allowlist,
        }
        key = _Journal.key(prompt, opts)
        cached = await self.journal.lookup(key)
        if cached is not None:
            return cached

        async with self._agent_lock:
            if self.agents_spawned >= self.max_agents:
                raise RuntimeError(
                    f"workflow agent-count backstop reached "
                    f"({self.max_agents}); aborting runaway script"
                )
            self.agents_spawned += 1

        async with self.semaphore:
            result = await self._spawn_and_drive(
                prompt, scenario, isolation, tool_allowlist
            )

        await self.journal.record(key, result)
        return result

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
    ) -> str:
        extensions: list[tuple[str, dict[str, Any]]] = []
        # isolation="agent_env" selects the worker sandbox by listing the
        # operations_agent_env atom in the child's extensions (policy by
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
            tool_allowlist=tool_allowlist,
            purpose=_WORKER_PURPOSE,
        )
        child: _ChildSession = await self.api.spawn_child_session(config)
        self.budget_svc.attach(child)
        try:
            messages = await child.prompt(prompt)
            return _final_assistant_text(messages)
        finally:
            with contextlib.suppress(Exception):
                await child.shutdown()


def _final_assistant_text(messages: list[AgentMessage]) -> str:
    """Last assistant message's joined text blocks (the ``tool_eval_run``
    extraction recipe: last text block wins)."""

    final_text = ""
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    final_text = text
    return final_text


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


_WORKFLOW_TOOL_PARAMS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "script": {
            "type": "string",
            "description": (
                "Async Python orchestration script. Available names: "
                "agent(prompt, *, scenario=, isolation=, tool_allowlist=) "
                "(awaitable -> str), parallel(list_of_awaitables) -> list, "
                "pipeline(items, *stages) -> list, budget (.total / .spent() / "
                ".remaining()), args (dict), log(msg), phase(name). Use await "
                "for agent / parallel / pipeline. `return <value>` becomes the "
                "tool result. Only data-manipulation builtins are available — "
                "no import / open / time / random."
            ),
        },
        "args": {
            "type": "object",
            "description": "Arbitrary JSON payload exposed to the script as args.",
            "additionalProperties": True,
        },
    },
    "required": ["script"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # Anti-recursion: a worker child session (spawned with purpose=workflow)
    # must not itself get the workflow tool, or a script could spawn workflows
    # without bound. Workers auto-discovering all builtins still hit this guard.
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

    async def _run_workflow(args: dict[str, Any]) -> ToolResult:
        script = args.get("script")
        if not isinstance(script, str) or not script.strip():
            return _error("workflow: 'script' (a Python string) is required")

        run = _WorkflowRun(
            api=api,
            journal=_Journal(store=api.get_service(_ARTIFACT_STORE_SERVICE)),
            budget_svc=_BudgetService(total_budget_tokens=budget_tokens),
            semaphore=asyncio.Semaphore(concurrency),
            default_scenario=default_scenario,
            max_agents=max_agents,
            args_payload=dict(args.get("args") or {}),
        )
        ns = _build_namespace(run)

        # Keep a one-shot host alive until the whole detached run (and any late
        # inbox post from a child) drains — matches sub_agent discipline.
        try:
            bracket = api.track_background()
        except ExtensionStaleError:
            bracket = contextlib.nullcontext()

        with bracket:
            try:
                if wall_clock_timeout is not None:
                    raw = await asyncio.wait_for(
                        _run_user_script(script, ns), timeout=wall_clock_timeout
                    )
                else:
                    raw = await _run_user_script(script, ns)
            except asyncio.TimeoutError:
                return _error(
                    f"workflow: script exceeded wall-clock budget "
                    f"({wall_clock_timeout}s)"
                )
            except Exception as exc:  # script errors surface as the tool error
                return _error(
                    f"workflow script error: {type(exc).__name__}: {exc}"
                )

        return ToolResult(
            content=[TextContent(type="text", text=_coerce_result(raw))],
            extras={
                "agents_spawned": run.agents_spawned,
                "budget": run.budget_svc.snapshot(),
            },
        )

    api.register_tool(
        FunctionTool(
            name="workflow",
            description=(
                "Run an LLM-authored async Python orchestration script that "
                "fans out deterministic child agent sessions. Use for parallel "
                "verify / judge / sweep harnesses where you author the control "
                "flow as code rather than turn-by-turn. Names: agent, parallel, "
                "pipeline, budget, args, log, phase. The script runs in a "
                "curated namespace (data builtins only; no import / open / time "
                "/ random). agent() results are journaled by hash(prompt, args) "
                "so re-running a workflow resumes from cache."
            ),
            parameters=_WORKFLOW_TOOL_PARAMS,
            fn=_run_workflow,
            metadata={"workflow": True},
        )
    )
