"""``workflow`` atom -- orchestration scripts with journal-based resume.

Single-file atom: journal, lineage, SDK primitives (agent / parallel /
pipeline), script runner, and tool registration are all inlined below.
"""

from __future__ import annotations

import ast
import asyncio
import builtins as _builtins
import hashlib
import inspect
import json
import os
import textwrap
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Final, Literal, TypeVar, cast

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    AtomAPI,
    AtomInstallPriority,
    Event,
    EventBus,
    FunctionTool,
    JsonValue,
    LoopConfig,
    TextContent,
    ToolCallBlock,
    ToolResult,
    ToolResultMessage,
    TurnCommittedEvent,
)
from agentm.core.lib import (
    error_result,
    expand_path_from_cwd,
    pydantic_to_tool_schema,
    text_result,
)
from agentm.extensions import ExtensionManifest

_T = TypeVar("_T")

_WORKER_PURPOSE: Final = "workflow"
_AGENT_COUNT_BACKSTOP: Final = 1000
_DEFAULT_MAX_AGENTS: Final = 200
_RUNNER_SERVICE: Final = "workflow_runner"
_JOURNAL_DIR: Final = ".agentm/workflow-journal"
_STRUCTURED_OUTPUT_MODULE: Final = "agentm.extensions.builtin.structured_output"
_STRUCTURED_OUTPUT_ATOM: Final = "structured_output"
_MIN_VERBATIM_LEN: Final = 12
_SNIPPET_CHARS: Final = 200

AgentResult = str | dict[str, JsonValue] | list[JsonValue] | BaseModel
WorkflowResult = str | dict[str, JsonValue] | list[JsonValue]

_SAFE_BUILTIN_NAMES: Final = (
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
_SAFE_BUILTINS: Final[dict[str, object]] = {
    n: getattr(_builtins, n)  # code-health: ignore[AM021]
    for n in _SAFE_BUILTIN_NAMES
}
_FORBIDDEN_CALLS: Final[frozenset[str]] = frozenset(
    {"open", "__import__", "eval", "exec", "compile", "input", "getattr"}
)


# ===== Journal =============================================================


@dataclass(slots=True)
class _JournalInvalidation:
    reason: str
    feedback: str | None
    carry_previous: bool


@dataclass(slots=True)
class _JournalEntry:
    key: str
    result: str
    prompt: str | None
    timestamp: float
    invalidated: bool = False


@dataclass(slots=True)
class _JournalState:
    cached: str | None = None
    invalidation: _JournalInvalidation | None = None


def _journal_key(prompt: str, opts: dict[str, object]) -> str:
    payload = json.dumps(
        {"prompt": prompt, "opts": opts}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _invalidation_rerun_prompt(
    prompt: str,
    inv: _JournalInvalidation,
    previous: str | None,
) -> str:
    lines = [
        prompt,
        "",
        "---",
        "A previous result for this task was invalidated. Redo the task.",
    ]
    if inv.reason.strip():
        lines.append(f"Why: {inv.reason.strip()}")
    if inv.feedback and inv.feedback.strip():
        lines.append(f"Guidance: {inv.feedback.strip()}")
    if inv.carry_previous and previous:
        lines += [
            "Previous result (for reference only):",
            "<previous_attempt>",
            previous,
            "</previous_attempt>",
        ]
    return "\n".join(lines)


def _journal_encode(prompt: str, result: str) -> str:
    return json.dumps(
        {"v": 2, "prompt": prompt, "result": result, "ts": time.time()},
        ensure_ascii=False,
    )


def _journal_decode(body: str) -> tuple[str | None, str, float]:
    try:
        p = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None, body, 0.0
    if isinstance(p, dict) and p.get("v") is not None:
        return p.get("prompt"), str(p.get("result", "")), float(p.get("ts", 0))
    return None, body, 0.0


@dataclass(slots=True)
class _Journal:
    root_dir: Path | None
    _cache: dict[str, str] = field(default_factory=dict)
    _inv_index: dict[str, tuple[float, _JournalInvalidation]] | None = None

    @classmethod
    def create(cls, cwd: str, root_session_id: str) -> _Journal:
        return cls(root_dir=Path(cwd) / _JOURNAL_DIR / root_session_id)

    def prime(self) -> None:
        if self.root_dir is None or not self.root_dir.is_dir():
            self._inv_index = {}
            return
        idx: dict[str, tuple[float, _JournalInvalidation]] = {}
        inv_dir = self.root_dir / "inv"
        if inv_dir.is_dir():
            for f in inv_dir.iterdir():
                if f.suffix != ".json":
                    continue
                try:
                    raw = json.loads(f.read_text(encoding="utf-8"))
                    ts = float(raw.get("ts", 0))
                    fb = raw.get("feedback")
                    inv = _JournalInvalidation(
                        reason=str(raw.get("reason", "")),
                        feedback=fb if isinstance(fb, str) else None,
                        carry_previous=bool(raw.get("carry_previous", False)),
                    )
                    existing = idx.get(f.stem)
                    if existing is None or ts > existing[0]:
                        idx[f.stem] = (ts, inv)
                except Exception:  # noqa: S110 BLE001
                    logger.debug("skipping corrupt journal {}", f)
        self._inv_index = idx

    def lookup(self, key: str) -> _JournalState:
        if key in self._cache:
            return _JournalState(cached=self._cache[key])
        if self.root_dir is None:
            return _JournalState()
        entry_file = self.root_dir / f"{key}.json"
        if not entry_file.is_file():
            return _JournalState()
        try:
            body = entry_file.read_text(encoding="utf-8")
        except OSError:
            return _JournalState()
        prompt, result, result_ts = _journal_decode(body)
        if self._inv_index is None:
            self.prime()
        assert self._inv_index is not None
        inv_entry = self._inv_index.get(key)
        invalidation: _JournalInvalidation | None = None
        if inv_entry is not None and inv_entry[0] > result_ts:
            invalidation = inv_entry[1]
        if invalidation is None:
            self._cache[key] = result
        return _JournalState(cached=result, invalidation=invalidation)

    def record(self, key: str, result: str, *, prompt: str) -> None:
        self._cache[key] = result
        if self.root_dir is None:
            return
        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            (self.root_dir / f"{key}.json").write_text(
                _journal_encode(prompt, result), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("workflow journal: write failed for {}: {}", key, exc)

    def write_invalidation(
        self,
        key: str,
        *,
        reason: str,
        feedback: str | None = None,
        carry_previous: bool = False,
    ) -> None:
        if self.root_dir is None:
            return
        inv_dir = self.root_dir / "inv"
        try:
            inv_dir.mkdir(parents=True, exist_ok=True)
            (inv_dir / f"{key}.json").write_text(
                json.dumps(
                    {
                        "reason": reason,
                        "feedback": feedback,
                        "carry_previous": carry_previous,
                        "ts": time.time(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("workflow journal: invalidation write failed: {}", exc)

    def key_exists(self, key: str) -> bool:
        if key in self._cache:
            return True
        if self.root_dir is None:
            return False
        return (self.root_dir / f"{key}.json").is_file()

    def load_entries(self) -> list[_JournalEntry]:
        if self.root_dir is None or not self.root_dir.is_dir():
            return []
        if self._inv_index is None:
            self.prime()
        assert self._inv_index is not None
        entries: list[_JournalEntry] = []
        for f in self.root_dir.iterdir():
            if f.suffix != ".json" or f.parent != self.root_dir:
                continue
            try:
                body = f.read_text(encoding="utf-8")
            except OSError:
                continue
            prompt, result, ts = _journal_decode(body)
            inv = self._inv_index.get(f.stem)
            entries.append(
                _JournalEntry(
                    key=f.stem,
                    result=result,
                    prompt=prompt,
                    timestamp=ts,
                    invalidated=inv is not None and inv[0] > ts,
                )
            )
        return entries


# ===== Lineage ==============================================================


@dataclass(slots=True)
class _LineageEdge:
    src: str
    dst: str


@dataclass(slots=True)
class _LineageGraph:
    edges: list[_LineageEdge]
    order_candidates: dict[str, list[str]]


def _derive_lineage(entries: list[_JournalEntry]) -> _LineageGraph:
    ordered = sorted(entries, key=lambda e: e.timestamp)
    edges: list[_LineageEdge] = []
    for si, src in enumerate(ordered):
        needle = src.result.strip()
        if len(needle) < _MIN_VERBATIM_LEN:
            continue
        fp = needle[:256]
        for dst in ordered[si + 1 :]:
            if dst.prompt is not None and fp in dst.prompt and needle in dst.prompt:
                edges.append(_LineageEdge(src=src.key, dst=dst.key))
    has_parent = {e.dst for e in edges}
    order_candidates: dict[str, list[str]] = {}
    prefix: list[str] = []
    for entry in ordered:
        if entry.prompt is not None and entry.key not in has_parent and prefix:
            order_candidates[entry.key] = list(prefix)
        prefix.append(entry.key)
    return _LineageGraph(edges=edges, order_candidates=order_candidates)


def _ancestors(graph: _LineageGraph, key: str) -> list[str]:
    parents_of: dict[str, list[str]] = {}
    for e in graph.edges:
        parents_of.setdefault(e.dst, []).append(e.src)
    seen: set[str] = set()
    frontier = [key]
    out: list[str] = []
    while frontier:
        node = frontier.pop(0)
        for parent in parents_of.get(node, []):
            if parent not in seen and parent != key:
                seen.add(parent)
                out.append(parent)
                frontier.append(parent)
    return out


# ===== SDK ==================================================================


@dataclass(frozen=True, slots=True)
class _WorkflowPhaseEvent(Event):
    CHANNEL: ClassVar[Literal["workflow_phase"]] = "workflow_phase"
    kind: Literal["phase", "log"] = "log"
    text: str = ""


@dataclass(slots=True)
class _BudgetTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    total: int | None = None

    def attach(self, child_bus: EventBus | None) -> None:
        if child_bus is None:
            return

        def _on_committed(event: TurnCommittedEvent) -> None:
            turn = event.turn
            if turn is None:
                return
            self.input_tokens += turn.meta.total_input_tokens
            self.output_tokens += turn.meta.total_output_tokens

        child_bus.on(TurnCommittedEvent.CHANNEL, _on_committed)

    @property
    def spent(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def remaining(self) -> int | None:
        if self.total is None:
            return None
        return max(0, self.total - self.spent)


@dataclass(frozen=True, slots=True)
class _Budget:
    _tracker: _BudgetTracker

    @property
    def total(self) -> int | None:
        return self._tracker.total

    def spent(self) -> int:
        return self._tracker.spent

    def remaining(self) -> int | None:
        return self._tracker.remaining


@dataclass(slots=True)
class _WorkflowRun:
    api: AtomAPI
    journal: _Journal
    budget: _BudgetTracker
    semaphore: asyncio.Semaphore
    default_scenario: str | None = None
    max_agents: int = _AGENT_COUNT_BACKSTOP
    agents_spawned: int = 0
    agents_succeeded: int = 0
    agents_failed: int = 0
    agents_retried: int = 0
    args_payload: dict[str, object] = field(default_factory=dict)
    cwd_override: str | None = None
    progress: list[dict[str, str]] = field(default_factory=list)
    child_sessions: list[dict[str, JsonValue]] = field(default_factory=list)
    _agent_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _bg_tasks: set[asyncio.Task[object]] = field(default_factory=set)

    async def agent(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | dict[str, JsonValue] | None = None,
        scenario: str | None = None,
        model: str | None = None,
        tool_allowlist: list[str] | None = None,
        extra_extensions: list[tuple[str, dict[str, JsonValue]]] | None = None,
        atom_config: dict[str, dict[str, JsonValue]] | None = None,
        retry: int = 0,
        timeout: float | None = None,
        label: str | None = None,
        max_turns: int | None = None,
    ) -> AgentResult:
        model_cls: type[BaseModel] | None = None
        json_schema: dict[str, JsonValue] | None = None
        if schema is not None:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                model_cls = schema
                json_schema = pydantic_to_tool_schema(schema)
            else:
                json_schema = schema
        opts: dict[str, object] = {
            "schema": json_schema,
            "scenario": scenario,
            "tool_allowlist": tool_allowlist,
            "extra_extensions": extra_extensions,
            "atom_config": atom_config,
        }
        if model is not None:
            opts["model"] = model

        key = _journal_key(prompt, opts)
        state = self.journal.lookup(key)
        if state.cached is not None and state.invalidation is None:
            parsed = _auto_parse(state.cached)
            if model_cls is not None and isinstance(parsed, dict):
                return model_cls.model_validate(parsed)
            return parsed

        current_prompt = prompt
        if state.invalidation is not None:
            current_prompt = _invalidation_rerun_prompt(
                prompt, state.invalidation, state.cached
            )

        last_error: Exception | None = None
        for attempt in range(max(retry, 0) + 1):
            async with self._agent_lock:
                if self.agents_spawned >= self.max_agents:
                    raise RuntimeError(
                        f"workflow agent-count backstop ({self.max_agents})"
                    )
                self.agents_spawned += 1
            try:
                async with self.semaphore:
                    coro = self._spawn_and_drive(
                        current_prompt,
                        scenario,
                        json_schema,
                        tool_allowlist=tool_allowlist,
                        extra_extensions=extra_extensions,
                        atom_config=atom_config,
                        label=label,
                        max_turns=max_turns,
                    )
                    result = await (
                        asyncio.wait_for(coro, timeout=timeout) if timeout else coro
                    )
            except TimeoutError as exc:
                last_error = exc
                if attempt < retry:
                    self.agents_retried += 1
                    current_prompt = _retry_prompt(prompt, exc, attempt + 1)
                    continue
                self.agents_failed += 1
                raise
            parsed = _auto_parse(result)
            if _is_error(parsed) and attempt < retry:
                self.agents_retried += 1
                continue
            if model_cls is not None and not _is_error(parsed):
                try:
                    validated = model_cls.model_validate(parsed)
                except (ValidationError, TypeError) as exc:
                    last_error = exc
                    if attempt < retry:
                        self.agents_retried += 1
                        current_prompt = _retry_prompt(prompt, exc, attempt + 1)
                        continue
                    self.agents_failed += 1
                    raise
                self.agents_succeeded += 1
                self.journal.record(key, result, prompt=prompt)
                return validated
            if _is_error(parsed):
                self.agents_failed += 1
            else:
                self.agents_succeeded += 1
            self.journal.record(key, result, prompt=prompt)
            return parsed
        assert last_error is not None
        raise last_error

    async def parallel(self, aws: list[Awaitable[_T]]) -> list[_T | None]:
        results = await asyncio.gather(*aws, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("workflow parallel: {}: {}", type(r).__name__, r)
        return [None if isinstance(r, BaseException) else r for r in results]

    async def pipeline(
        self, items: list[object], *stages: Callable[[object], object]
    ) -> list[object]:
        async def _chain(item: object) -> object:
            cur: object = item
            for stage in stages:
                try:
                    cur = stage(cur)
                    if inspect.isawaitable(cur):
                        cur = await cur
                except Exception as exc:
                    logger.warning("workflow pipeline stage failed: {}", exc)
                    return None
            return cur

        return list(await asyncio.gather(*(_chain(it) for it in items)))

    def log(self, msg: object) -> None:
        self._emit("log", str(msg))

    def phase(self, name: object) -> None:
        self._emit("phase", str(name))

    def _emit(self, kind: Literal["phase", "log"], text: str) -> None:
        self.progress.append({"kind": kind, "text": text})
        try:
            task = asyncio.create_task(
                self.api.bus.emit(
                    _WorkflowPhaseEvent.CHANNEL,
                    _WorkflowPhaseEvent(kind=kind, text=text),
                )
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        except RuntimeError:
            pass

    async def _spawn_and_drive(
        self,
        prompt: str,
        scenario: str | None,
        schema: dict[str, JsonValue] | None,
        *,
        tool_allowlist: list[str] | None = None,
        extra_extensions: list[tuple[str, dict[str, JsonValue]]] | None = None,
        atom_config: dict[str, dict[str, JsonValue]] | None = None,
        label: str | None = None,
        max_turns: int | None = None,
    ) -> str:
        extensions: list[tuple[str, dict[str, JsonValue]]] = []
        overrides: dict[str, dict[str, JsonValue]] = dict(atom_config or {})
        if schema is not None:
            extensions.append((_STRUCTURED_OUTPUT_MODULE, {}))
            overrides[_STRUCTURED_OUTPUT_ATOM] = {"schema": schema}
        if extra_extensions:
            extensions.extend(extra_extensions)
        config = AgentSessionConfig(
            cwd=self.cwd_override or self.api.ctx.cwd,
            scenario=scenario or self.default_scenario or self.api.ctx.scenario,
            extra_extensions=extensions,  # type: ignore[arg-type]
            atom_config_overrides=overrides,
            tool_allowlist=tool_allowlist,
            purpose=_WORKER_PURPOSE,
            parent_session_id=self.api.ctx.session_id,
            root_session_id=self.api.ctx.root_session_id,
            loop_config=LoopConfig(max_turns=max_turns) if max_turns else None,
        )
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                child = await self.api.spawn_child_session(config)
                self.child_sessions.append(
                    {
                        "session_id": child.session_id,
                        "scenario": scenario or self.default_scenario,
                        "label": label,
                        "prompt": prompt[:200],
                    }
                )
                self.budget.attach(
                    child.bus
                    if hasattr(child, "bus")  # code-health: ignore[AM021]
                    else None
                )
                try:
                    messages = await child.run(prompt)
                    output = _final_session_output(messages)
                    return (
                        output
                        if output
                        else json.dumps({"_error": "no_output", "turns": len(messages)})
                    )
                finally:
                    try:
                        await child.shutdown()
                    except Exception:  # noqa: S110 BLE001
                        logger.debug("child shutdown error (ignored)")
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    logger.warning("workflow agent spawn failed, retrying: {}", exc)
                    await asyncio.sleep(0.5)
                    continue
        return json.dumps(
            {
                "_error": type(last_exc).__name__ if last_exc else "unknown",
                "detail": str(last_exc)[:500] if last_exc else "",
            }
        )


# ===== Output extraction ====================================================


def _auto_parse(text: str) -> str | dict[str, JsonValue] | list[JsonValue]:
    if not text:
        return text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return text


def _is_error(result: AgentResult) -> bool:
    return (isinstance(result, dict) and "_error" in result) or (
        isinstance(result, str) and not result
    )


def _retry_prompt(original: str, exc: Exception, attempt: int) -> str:
    return f"{original}\n\n## Retry #{attempt}\nPrevious attempt failed: {str(exc)[:2000]}\n\nYou MUST call submit_result with a valid result."


def _final_session_output(messages: list[AgentMessage]) -> str:
    submit_names: set[str] = set()
    call_names: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolCallBlock):
                    call_names[block.id] = block.name
                    if block.name.startswith("submit_"):
                        submit_names.add(block.name)
    if submit_names:
        for msg in reversed(messages):
            if not isinstance(msg, ToolResultMessage):
                continue
            for rb in reversed(msg.content):
                if not rb.is_error and call_names.get(rb.tool_call_id) in submit_names:
                    for inner in reversed(rb.content):
                        if isinstance(inner, TextContent) and inner.text:
                            return inner.text
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage) and not any(
            isinstance(b, ToolCallBlock) for b in msg.content
        ):
            for block in reversed(msg.content):
                if isinstance(block, TextContent) and block.text:
                    return block.text
    return ""


# ===== Script validation + execution ========================================


def _validate_script(source: str) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [(exc.lineno or 0, f"SyntaxError: {exc.msg}")]
    issues: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            issues.append((node.lineno, "import not available in workflow scripts"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_CALLS
        ):
            issues.append((node.lineno, f"'{node.func.id}()' is not available"))
    return issues


def _build_namespace(run: _WorkflowRun) -> dict[str, object]:
    return {
        "__builtins__": _SAFE_BUILTINS,
        "agent": run.agent,
        "parallel": run.parallel,
        "pipeline": run.pipeline,
        "budget": _Budget(run.budget),
        "args": run.args_payload,
        "json": json,
        "log": run.log,
        "phase": run.phase,
    }


async def _exec_script(script: str, ns: dict[str, object]) -> object:
    src = "async def __workflow__():\n" + textwrap.indent(script, "    ")
    code = compile(src, "<workflow-script>", "exec")
    exec(code, ns)  # noqa: S102
    workflow_fn = cast("Callable[[], Awaitable[object]]", ns["__workflow__"])
    return await workflow_fn()


def _coerce(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if raw is None:
        return ""
    try:
        return json.dumps(raw, ensure_ascii=False, default=str)
    except TypeError:
        return str(raw)


# ===== WorkflowRunner service ===============================================


def _default_concurrency() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(16, cpu - 2))


class WorkflowRunner:
    def __init__(
        self,
        api: AtomAPI,
        *,
        max_concurrency: int,
        max_agents: int,
        default_scenario: str | None,
        budget_tokens: int | None,
        wall_clock_timeout: float | None,
    ) -> None:
        self._api = api
        self._concurrency = max_concurrency
        self._max_agents = max_agents
        self._default_scenario = default_scenario
        self._budget_tokens = budget_tokens
        self._wall_clock_timeout = wall_clock_timeout

    async def run_script(
        self,
        script: str,
        args: dict[str, object] | None = None,
        *,
        cwd: str | None = None,
    ) -> WorkflowResult:
        if not script.strip():
            raise ValueError("workflow: empty script")
        return await self._execute(script, args or {}, cwd=cwd)

    async def run_file(
        self,
        path: str | Path,
        args: dict[str, object] | None = None,
        *,
        cwd: str | None = None,
    ) -> WorkflowResult:
        sp = Path(path)
        if not sp.is_absolute():
            sp = expand_path_from_cwd(sp, self._api.ctx.cwd)
        sp = sp.resolve()
        if not sp.is_file():
            raise FileNotFoundError(f"workflow script not found: {sp}")
        return await self._execute(sp.read_text(encoding="utf-8"), args or {}, cwd=cwd)

    async def _execute(
        self, script: str, args_payload: dict[str, object], *, cwd: str | None = None
    ) -> WorkflowResult:
        errors = _validate_script(script)
        if errors:
            detail = "\n".join(f"  line {ln}: {msg}" for ln, msg in errors)
            raise ValueError(f"workflow script validation failed:\n{detail}")
        journal = _Journal.create(self._api.ctx.cwd, self._api.ctx.root_session_id)
        journal.prime()
        run = _WorkflowRun(
            api=self._api,
            journal=journal,
            budget=_BudgetTracker(total=self._budget_tokens),
            semaphore=asyncio.Semaphore(self._concurrency),
            default_scenario=self._default_scenario,
            max_agents=self._max_agents,
            args_payload=dict(args_payload),
            cwd_override=cwd,
        )
        t0 = time.monotonic()
        coro = _exec_script(script, _build_namespace(run))
        raw = await (
            asyncio.wait_for(coro, timeout=self._wall_clock_timeout)
            if self._wall_clock_timeout
            else coro
        )
        if run._bg_tasks:
            await asyncio.gather(*run._bg_tasks, return_exceptions=True)
        logger.info(
            "workflow: done in {:.1f}s, {} agents ({} ok, {} failed)",
            time.monotonic() - t0,
            run.agents_spawned,
            run.agents_succeeded,
            run.agents_failed,
        )
        return _auto_parse(_coerce(raw))

    async def handle_tool(self, args: dict[str, JsonValue]) -> ToolResult:
        parsed = _WorkflowToolArgs.model_validate(args)
        script = parsed.script
        if parsed.script_path and not script:
            try:
                script = expand_path_from_cwd(
                    parsed.script_path, self._api.ctx.cwd
                ).read_text(encoding="utf-8")
            except OSError as exc:
                return error_result(f"cannot read script: {exc}")
        if not script:
            return error_result("either 'script' or 'script_path' is required")
        try:
            result = await self._execute(script, parsed.args or {}, cwd=parsed.cwd)
        except Exception as exc:
            return error_result(f"workflow failed: {type(exc).__name__}: {exc}")
        return text_result(
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False, default=str)
        )


# ===== MANIFEST + tool registration ========================================


class WorkflowConfig(BaseModel):
    max_concurrency: int | None = None
    max_agents: int | None = None
    wall_clock_timeout_s: float | None = None
    default_scenario: str | None = None
    budget_tokens: int | None = Field(default=None, gt=0)


MANIFEST = ExtensionManifest(
    name="workflow",
    description="Run orchestration scripts with agent/parallel/pipeline and journal-based resume.",
    registers=(
        "tool:workflow",
        "tool:workflow_lineage",
        "tool:workflow_invalidate",
        "service:workflow_runner",
    ),
    config_schema=WorkflowConfig,
    requires=(),
    priority=AtomInstallPriority.TOOL,
)


class _WorkflowToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    script: str | None = Field(
        default=None, description="Inline async Python orchestration script."
    )
    script_path: str | None = Field(default=None, description="Path to script file.")
    args: dict[str, object] | None = Field(
        default=None, description="JSON payload exposed as args."
    )
    cwd: str | None = Field(default=None, description="Working directory override.")


class _LineageParams(BaseModel):
    key: str | None = Field(
        default=None, description="Scope to this key + ancestors. Omit for full graph."
    )


class _InvalidateParams(BaseModel):
    key: str = Field(description="Journal key of the wrong agent() result.")
    reason: str = Field(description="Why the result is wrong.")
    feedback: str | None = Field(default=None, description="Guidance for the re-run.")
    carry_previous: bool = Field(
        default=False, description="Include invalidated result in re-run prompt."
    )


class _WorkflowAtomRuntime:
    def __init__(self, api: AtomAPI, config: WorkflowConfig) -> None:
        self._api = api
        self._runner = WorkflowRunner(
            api,
            max_concurrency=config.max_concurrency or _default_concurrency(),
            max_agents=config.max_agents or _DEFAULT_MAX_AGENTS,
            default_scenario=config.default_scenario,
            budget_tokens=config.budget_tokens,
            wall_clock_timeout=config.wall_clock_timeout_s,
        )

    def install(self) -> None:
        self._api.services.register(_RUNNER_SERVICE, self._runner, scope="session")
        self._api.register_tool(
            FunctionTool(
                name="workflow",
                description="Run an async Python orchestration script. SDK: agent(prompt, *, schema=, scenario=, extra_extensions=, atom_config=, retry=, timeout=, max_turns=, label=), parallel([awaitables]), pipeline(items, *stages), budget, args, json, log(msg), phase(name). `return` becomes the result. Results are journal-cached for resume.",
                parameters=pydantic_to_tool_schema(_WorkflowToolArgs),
                fn=self._runner.handle_tool,
                metadata={"workflow": True},
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="workflow_lineage",
                description="Inspect the workflow journal as a dependency graph.",
                parameters=pydantic_to_tool_schema(_LineageParams),
                fn=self._lineage,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="workflow_invalidate",
                description="Flag a journaled agent() result as wrong so the next run redoes it and downstream nodes.",
                parameters=pydantic_to_tool_schema(_InvalidateParams),
                fn=self._invalidate,
            )
        )

    async def _lineage(self, args: dict[str, JsonValue]) -> ToolResult:
        try:
            params = _LineageParams.model_validate(args)
        except ValidationError as exc:
            return error_result(f"workflow_lineage: {exc}")
        journal = _Journal.create(self._api.ctx.cwd, self._api.ctx.root_session_id)
        entries = journal.load_entries()
        if not entries:
            return text_result("The workflow journal is empty.")
        graph = _derive_lineage(entries)
        selected = entries
        if params.key and params.key.strip():
            key = params.key.strip()
            if key not in {e.key for e in entries}:
                return error_result(
                    f"No entry with key '{key}'. Known: {', '.join(sorted(e.key for e in entries)[:10])}"
                )
            wanted = {key, *_ancestors(graph, key)}
            selected = [e for e in entries if e.key in wanted]
        selected_keys = {e.key for e in selected}
        payload = {
            "nodes": [
                {
                    "key": e.key,
                    "prompt_head": (e.prompt or "")[:_SNIPPET_CHARS],
                    "result_head": e.result[:_SNIPPET_CHARS],
                    "invalidated": e.invalidated,
                    "recorded_at": e.timestamp,
                }
                for e in selected
            ],
            "edges": [
                {"src": e.src, "dst": e.dst}
                for e in graph.edges
                if e.src in selected_keys and e.dst in selected_keys
            ],
            "order_candidates": {
                n: p for n, p in graph.order_candidates.items() if n in selected_keys
            },
        }
        return text_result(json.dumps(payload, ensure_ascii=False, indent=2))

    async def _invalidate(self, args: dict[str, JsonValue]) -> ToolResult:
        try:
            params = _InvalidateParams.model_validate(args)
        except ValidationError as exc:
            return error_result(f"workflow_invalidate: {exc}")
        key, reason = params.key.strip(), params.reason.strip()
        if not key or not reason:
            return error_result("Both key and reason are required.")
        journal = _Journal.create(self._api.ctx.cwd, self._api.ctx.root_session_id)
        if not journal.key_exists(key):
            return error_result(f"No journaled result with key '{key}'.")
        journal.write_invalidation(
            key,
            reason=reason,
            feedback=(params.feedback or "").strip() or None,
            carry_previous=params.carry_previous,
        )
        return text_result(
            f"Invalidated entry {key}. On next run this node and downstream nodes re-run."
        )


def install(api: AtomAPI, config: WorkflowConfig) -> None:
    if api.ctx.purpose == _WORKER_PURPOSE:
        return
    _WorkflowAtomRuntime(api, config).install()
