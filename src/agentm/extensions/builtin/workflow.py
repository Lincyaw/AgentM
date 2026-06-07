"""``workflow`` atom — run an LLM-authored JavaScript orchestration script.

See ``.claude/designs/dynamic-workflow.md``. A *dynamic workflow* is a
model-authored JS script that orchestrates many child agent sessions
deterministically. The model writes the control flow (loops, branches,
fan-out, budget-driven scaling) as code at runtime; this atom runs it as a
single tool call and only the final result returns to the model's context.

Because the script is model-generated it is **untrusted**, so it runs inside
a QuickJS isolate (a capability sandbox — nothing host-side is reachable
unless we inject it). We inject seven primitives bound to *already-existing*
APIs:

- ``agent(prompt, opts)`` — :meth:`ExtensionAPI.spawn_child_session` →
  ``child.prompt`` → last ``AssistantMessage`` text → ``child.shutdown``.
- ``parallel(thunks)`` — barrier fan-out via ``asyncio.gather`` + a
  ``Semaphore`` cap.
- ``pipeline(items, ...stageNames)`` — per-item staged map over the same cap
  (no global barrier between stages of distinct items).
- ``log(msg)`` — emit a ``WorkflowPhaseEvent`` on the parent bus.
- ``phase(name)`` — same, marking a named phase.
- ``budget()`` — read aggregated child token spend
  (:class:`_BudgetService`).
- ``args()`` — the caller-supplied ``args`` payload.

Nondeterministic JS globals (``Math.random``, ``Date.now``, argless
``new Date()``) are stripped so the workflow-local journal key
``hash(prompt, opts)`` stays sound across replays.

§11 single-file contract: stdlib + ``agentm.core.abi.*`` +
``agentm.core.lib.*`` + ``agentm.extensions.*`` + ``quickjs`` (optional
3rd-party). No atom-to-atom imports — ``artifact_store`` is reached via
``api.get_service`` and worker child sessions via
``api.spawn_child_session``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Literal

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.events import Event, TurnEndEvent
from agentm.core.abi.extension import ExtensionAPI, ExtensionStaleError
from agentm.core.abi.messages import AssistantMessage
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.extensions import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="workflow",
    description=(
        "Run an LLM-authored JavaScript orchestration script in a QuickJS "
        "isolate that fans out deterministic child agent sessions via the "
        "agent/parallel/pipeline primitives. Only the final result returns "
        "to the model's context."
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

# Lifetime backstop: a runaway script cannot spawn more than this many
# child sessions across the whole workflow run.
_AGENT_COUNT_BACKSTOP: Final[int] = 1000

# JS bootstrap: strip the three nondeterministic globals that would break
# replay-determinism of the journal key, then install ergonomic wrappers
# that marshal scalars-only across the boundary (QuickJS add_callable only
# round-trips int/float/str/bool — see dynamic-workflow.md). Strings on the
# wire, JSON on both sides.
_JS_BOOTSTRAP = r"""
// --- strip nondeterministic globals (journal-key soundness) ---
Math.random = function () { throw new Error('Math.random() disabled in workflow sandbox'); };
const __OrigDate = Date;
globalThis.Date = function (...a) {
  if (a.length === 0) { throw new Error('argless Date() disabled in workflow sandbox'); }
  return new __OrigDate(...a);
};
globalThis.Date.now = function () { throw new Error('Date.now() disabled in workflow sandbox'); };

// --- primitive wrappers (delegate to injected __host_* callables) ---
globalThis.agent = function (prompt, opts) {
  const spec = { prompt: prompt, opts: opts || {} };
  return JSON.parse(__host_agent(JSON.stringify(spec)));
};
globalThis.parallel = function (specs) {
  // specs: array of {prompt, opts}; returns array of results, barrier-joined.
  return JSON.parse(__host_parallel(JSON.stringify(specs || [])));
};
globalThis.pipeline = function (items, opts) {
  // items: array of {prompt, opts}; mapped over the same concurrency cap
  // without a per-stage global barrier.
  return JSON.parse(__host_pipeline(JSON.stringify({ items: items || [], opts: opts || {} })));
};
globalThis.log = function (msg) { __host_log(String(msg)); };
globalThis.phase = function (name) { __host_phase(String(name)); };
globalThis.budget = function () { return JSON.parse(__host_budget('')); };
globalThis.args = function () { return JSON.parse(__host_args('')); };
"""


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

    def attach(self, child: Any) -> None:
        """Subscribe to ``turn_end`` on the child's own bus."""

        bus = getattr(child, "bus", None)
        if bus is None:
            return

        async def _on_turn_end(event: TurnEndEvent) -> None:
            usage = getattr(event.message, "usage", None)
            if usage is None:
                return
            async with self._lock:
                self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        bus.on(TurnEndEvent.CHANNEL, _on_turn_end)

    def snapshot(self) -> dict[str, Any]:
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

    store: Any | None
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
    """One ``workflow`` tool invocation: owns the journal, budget aggregator,
    concurrency cap and lifetime backstop for a single script run."""

    api: ExtensionAPI
    journal: _Journal
    budget: _BudgetService
    semaphore: asyncio.Semaphore
    loop: asyncio.AbstractEventLoop
    default_scenario: str | None = None
    max_agents: int = _AGENT_COUNT_BACKSTOP
    agents_spawned: int = 0
    args_payload: dict[str, Any] = field(default_factory=dict)
    _agent_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def _run_one_agent(self, prompt: str, opts: dict[str, Any]) -> str:
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
            result = await self._spawn_and_drive(prompt, opts)

        await self.journal.record(key, result)
        return result

    async def _spawn_and_drive(self, prompt: str, opts: dict[str, Any]) -> str:
        scenario = opts.get("scenario") or self.default_scenario
        extensions: list[tuple[str, dict[str, Any]]] = []
        # opts.isolation selects the worker sandbox by listing the
        # operations_agent_env atom in the child's extensions (policy by
        # composition, never a privileged config field).
        if opts.get("isolation") == "agent_env":
            extensions.append(
                ("agentm.extensions.builtin.operations_agent_env", {})
            )
        tool_allowlist = opts.get("tool_allowlist")

        config = AgentSessionConfig(
            cwd=self.api.cwd,
            provider=None,  # inherit parent's provider (spawn auto-wires it)
            scenario=scenario,
            extra_extensions=extensions,
            tool_allowlist=tool_allowlist if isinstance(tool_allowlist, list) else None,
            purpose="workflow",
        )
        child = await self.api.spawn_child_session(config)
        self.budget.attach(child)
        try:
            messages = await child.prompt(prompt)
            return _final_assistant_text(messages)
        finally:
            with contextlib.suppress(Exception):
                await child.shutdown()

    async def run_parallel(self, specs: list[dict[str, Any]]) -> list[str]:
        tasks = [
            asyncio.create_task(
                self._run_one_agent(
                    str(spec.get("prompt", "")),
                    dict(spec.get("opts") or {}),
                )
            )
            for spec in specs
        ]
        return list(await asyncio.gather(*tasks))

    async def run_pipeline(
        self, items: list[dict[str, Any]], opts: dict[str, Any]
    ) -> list[str]:
        # No per-stage global barrier: each item flows independently, the
        # Semaphore alone caps concurrency. With a single implicit stage this
        # is equivalent to parallel(); the distinction matters once a script
        # composes staged thunks, which is scheduling policy only.
        del opts
        return await self.run_parallel(items)


def _final_assistant_text(messages: list[Any]) -> str:
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


def _load_quickjs() -> Any:
    """Import the optional QuickJS engine, raising a clear actionable error.

    ``quickjs`` (PetterS, archived) and ``quickjs-ng`` (maintained fork) are
    drop-in: both expose ``import quickjs``. The import is deferred to tool
    invocation so the atom *loads* cleanly without the engine and unrelated
    tests stay green.
    """

    try:
        import quickjs  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - engine-absent surface
        raise RuntimeError(
            "the 'workflow' tool requires a QuickJS engine. Install with: "
            "uv sync --extra workflow  (pulls in 'quickjs-ng')."
        ) from exc
    return quickjs


_WORKFLOW_TOOL_PARAMS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "script": {
            "type": "string",
            "description": (
                "JavaScript orchestration script. Has access to: "
                "agent(prompt, opts), parallel([{prompt,opts}]), "
                "pipeline([{prompt,opts}], opts), log(msg), phase(name), "
                "budget(), args(). Return a value (string or JSON-able) as the "
                "last expression — it becomes the tool result. Math.random / "
                "Date.now / argless new Date() are disabled."
            ),
        },
        "args": {
            "type": "object",
            "description": "Arbitrary JSON payload exposed to the script via args().",
            "additionalProperties": True,
        },
    },
    "required": ["script"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
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

    async def _run_workflow(args: dict[str, Any]) -> ToolResult:
        script = args.get("script")
        if not isinstance(script, str) or not script.strip():
            return _error("workflow: 'script' (a JS string) is required")
        try:
            quickjs = _load_quickjs()
        except RuntimeError as exc:
            return _error(str(exc))

        run = _WorkflowRun(
            api=api,
            journal=_Journal(store=api.get_service(_ARTIFACT_STORE_SERVICE)),
            budget=_BudgetService(),
            semaphore=asyncio.Semaphore(concurrency),
            loop=asyncio.get_running_loop(),
            default_scenario=default_scenario,
            max_agents=max_agents,
            args_payload=dict(args.get("args") or {}),
        )

        # Build the isolated context and inject the host callbacks. add_callable
        # callbacks are synchronous; to call into the async host APIs we
        # re-enter the captured loop and block the *worker* thread (not the
        # host loop) on the future.
        ctx = quickjs.Context()
        with contextlib.suppress(Exception):
            ctx.set_memory_limit(128 * 1024 * 1024)
            ctx.set_max_stack_size(1024 * 1024)

        def _call_async(coro: Any) -> Any:
            fut = asyncio.run_coroutine_threadsafe(coro, run.loop)
            return fut.result()

        def _host_agent(spec_json: str) -> str:
            spec = json.loads(spec_json)
            result = _call_async(
                run._run_one_agent(
                    str(spec.get("prompt", "")), dict(spec.get("opts") or {})
                )
            )
            return json.dumps(result, ensure_ascii=False)

        def _host_parallel(specs_json: str) -> str:
            specs = json.loads(specs_json)
            result = _call_async(run.run_parallel(list(specs)))
            return json.dumps(result, ensure_ascii=False)

        def _host_pipeline(payload_json: str) -> str:
            payload = json.loads(payload_json)
            result = _call_async(
                run.run_pipeline(
                    list(payload.get("items") or []),
                    dict(payload.get("opts") or {}),
                )
            )
            return json.dumps(result, ensure_ascii=False)

        def _emit_phase(kind: str, text: str) -> None:
            with contextlib.suppress(Exception):
                _call_async(
                    api.events.emit(
                        WorkflowPhaseEvent.CHANNEL,
                        WorkflowPhaseEvent(kind=kind, text=text),  # type: ignore[arg-type]
                    )
                )

        def _host_log(msg: str) -> str:
            _emit_phase("log", msg)
            return ""

        def _host_phase(name: str) -> str:
            _emit_phase("phase", name)
            return ""

        def _host_budget(_unused: str) -> str:
            return json.dumps(run.budget.snapshot(), ensure_ascii=False)

        def _host_args(_unused: str) -> str:
            return json.dumps(run.args_payload, ensure_ascii=False)

        ctx.add_callable("__host_agent", _host_agent)
        ctx.add_callable("__host_parallel", _host_parallel)
        ctx.add_callable("__host_pipeline", _host_pipeline)
        ctx.add_callable("__host_log", _host_log)
        ctx.add_callable("__host_phase", _host_phase)
        ctx.add_callable("__host_budget", _host_budget)
        ctx.add_callable("__host_args", _host_args)

        def _run_script() -> Any:
            ctx.eval(_JS_BOOTSTRAP)
            return ctx.eval(script)

        # Keep a one-shot host alive until the whole detached run (and any
        # late inbox post from a child) drains — matches sub_agent discipline.
        try:
            bracket = api.track_background()
        except ExtensionStaleError:
            bracket = contextlib.nullcontext()

        with bracket:
            try:
                if wall_clock_timeout is not None:
                    raw = await asyncio.wait_for(
                        asyncio.to_thread(_run_script), timeout=wall_clock_timeout
                    )
                else:
                    raw = await asyncio.to_thread(_run_script)
            except asyncio.TimeoutError:
                return _error(
                    f"workflow: script exceeded wall-clock budget "
                    f"({wall_clock_timeout}s)"
                )
            except Exception as exc:  # JS errors surface as the tool error
                return _error(f"workflow script error: {exc}")

        result_text = _coerce_js_result(raw)
        return ToolResult(
            content=[TextContent(type="text", text=result_text)],
            extras={
                "agents_spawned": run.agents_spawned,
                "budget": run.budget.snapshot(),
            },
        )

    api.register_tool(
        FunctionTool(
            name="workflow",
            description=(
                "Run an LLM-authored JavaScript orchestration script that fans "
                "out deterministic child agent sessions. Use for parallel "
                "verify/judge/sweep harnesses where you author the control flow "
                "as code rather than turn-by-turn. Primitives: agent, parallel, "
                "pipeline, log, phase, budget, args. The script runs in an "
                "isolated sandbox (no fs/net/shell); Math.random/Date.now are "
                "disabled. agent() results are journaled by hash(prompt,opts) so "
                "re-running a workflow resumes from cache."
            ),
            parameters=_WORKFLOW_TOOL_PARAMS,
            fn=_run_workflow,
            metadata={"workflow": True},
        )
    )


def _coerce_js_result(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if raw is None:
        return ""
    try:
        return json.dumps(raw, ensure_ascii=False)
    except TypeError:
        return str(raw)
