"""Workflow atom script validation, execution, and runner service."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import importlib
import importlib.util
import json
from loguru import logger
import sys
import textwrap
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    ARTIFACT_STORE_SERVICE,
    ExtensionAPI,
    ExtensionStaleError,
    TextContent,
    ToolResult,
)
from agentm.core.lib import expand_path_from_cwd
from agentm.extensions.builtin._workflow.sdk import (
    AgentMockMode,
    BudgetSnapshot,
    ModelResolver,
    RunSummary,
    WorkflowContext,
    WorkflowResult,
    _ArtifactStore,
    _Budget,
    _BudgetService,
    _Journal,
    _SAFE_BUILTINS,
    _WorkflowRun,
    _auto_parse,
    _empty_budget_snapshot,
    _empty_run_summary,
)

# ---------------------------------------------------------------------------
# Pre-execution validation
# ---------------------------------------------------------------------------


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


async def _call_module_run(module: Any, ctx: WorkflowContext) -> object:
    """Invoke a loaded module's ``run(ctx)`` entry point.

    Shared by both module-load branches (package import vs standalone file)
    so the entry-point contract is checked in one place."""
    run_fn = getattr(module, "run", None)
    if run_fn is None or not callable(run_fn):
        raise RuntimeError(
            "module-mode script must define `async def run(ctx: WorkflowContext)`"
        )
    run_callable = cast("Callable[[WorkflowContext], Awaitable[object]]", run_fn)
    return await run_callable(ctx)


def _module_loaded_from(module: Any, root: Path) -> bool:
    raw_file = getattr(module, "__file__", None)
    if not isinstance(raw_file, str) or not raw_file:
        return False
    try:
        module_file = Path(raw_file).resolve()
    except (OSError, RuntimeError):
        return False
    return module_file.is_relative_to(root)


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
        # Narrow sys.path mutation to the synchronous import phase so it
        # never spans an ``await`` — concurrent module-mode workflow
        # coroutines on the same event loop cannot see each other's path
        # entries (asyncio is cooperative; no yield ⇒ no interleaving).
        module_name = ".".join(parts)
        path_entry = str(current)
        pkg_root = parts[0]
        try:
            import_root = current.resolve()
        except (OSError, RuntimeError):
            import_root = current
        modules_before = set(sys.modules)
        added = path_entry not in sys.path
        if added:
            sys.path.insert(0, path_entry)
        try:
            module = importlib.import_module(module_name)
        finally:
            if added:
                try:
                    sys.path.remove(path_entry)
                except ValueError as exc:
                    # Another importer already removed our temporary entry.
                    logger.debug("workflow: sys.path entry already removed: {}", exc)
        try:
            return await _call_module_run(module, ctx)
        finally:
            stale = [
                k
                for k, loaded in list(sys.modules.items())
                if k == pkg_root
                or k.startswith(pkg_root + ".")
                or (
                    k not in modules_before and _module_loaded_from(loaded, import_root)
                )
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
            return await _call_module_run(module, ctx)
        finally:
            sys.modules.pop(module_name, None)


class _WorkflowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    script: str | None = Field(
        default=None,
        description=(
            "Async Python orchestration script. Available names: "
            "agent(prompt, *, schema=, scenario=, model=, isolation=, "
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
    )
    script_path: str | None = Field(
        default=None,
        description=(
            "Path to a pre-written async Python workflow script file. "
            "Mutually exclusive with script. The file is read and executed "
            "in the same curated namespace as an inline script."
        ),
    )
    args: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary JSON payload exposed to the script as args.",
    )
    cwd: str | None = Field(
        default=None,
        description=(
            "Working directory for child agent sessions spawned by "
            "the workflow. Defaults to the caller's cwd."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Model name or config.toml profile for child agent "
            "sessions. Defaults to the caller's model."
        ),
    )
    agent_mock: Literal["off", "mock"] | None = Field(
        default=None,
        description=(
            "Set to 'mock' to dry-run agent() calls. The workflow still "
            "executes, but ctx.agent/agent returns synthetic results and "
            "logs call parameters via workflow_phase events."
        ),
    )


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
        default_agent_timeout: float | None,
        agent_mock: AgentMockMode,
        model_resolver: ModelResolver | None,
    ) -> None:
        self._api = api
        self._concurrency = concurrency
        self._max_agents = max_agents
        self._wall_clock_timeout = wall_clock_timeout
        self._default_scenario = default_scenario
        self._budget_tokens = budget_tokens
        self._default_agent_timeout = default_agent_timeout
        self._agent_mock = agent_mock
        self._model_resolver = model_resolver
        self._last_run: _WorkflowRun | None = None
        self._last_wall_clock_s: float = 0.0

    @property
    def last_agents_spawned(self) -> int:
        return self._last_run.agents_spawned if self._last_run else 0

    @property
    def last_budget_snapshot(self) -> BudgetSnapshot:
        if self._last_run is None:
            return _empty_budget_snapshot()
        return self._last_run.budget_svc.snapshot()

    @property
    def last_progress(self) -> list[dict[str, str]]:
        return self._last_run.progress if self._last_run else []

    @property
    def last_run_summary(self) -> RunSummary:
        if self._last_run is None:
            return _empty_run_summary()
        return RunSummary(
            agents_spawned=self._last_run.agents_spawned,
            agents_mocked=self._last_run.agents_mocked,
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
            ARTIFACT_STORE_SERVICE,
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
        agent_mock: AgentMockMode | None = None,
    ) -> WorkflowResult:
        """Run a pre-written workflow script file."""
        sp = Path(path)
        if not sp.is_absolute():
            sp = expand_path_from_cwd(sp, self._api.cwd)
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
            agent_mock=agent_mock,
        )

    async def run_script(
        self,
        script: str,
        args: dict[str, Any] | None = None,
        *,
        cwd: str | None = None,
        provider: tuple[str, dict[str, Any]] | None = None,
        agent_mock: AgentMockMode | None = None,
    ) -> WorkflowResult:
        """Run an inline workflow script string."""
        if not script.strip():
            raise ValueError("workflow: empty script")
        return await self._execute(
            script,
            args or {},
            cwd=cwd,
            provider=provider,
            agent_mock=agent_mock,
        )

    async def _execute(
        self,
        script: str,
        args_payload: dict[str, Any],
        *,
        source_path: Path | None = None,
        cwd: str | None = None,
        provider: tuple[str, dict[str, Any]] | None = None,
        agent_mock: AgentMockMode | None = None,
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
                store=self._api.get_service(ARTIFACT_STORE_SERVICE),
            ),
            budget_svc=_BudgetService(
                total_budget_tokens=self._budget_tokens,
            ),
            semaphore=asyncio.Semaphore(self._concurrency),
            default_scenario=self._default_scenario,
            max_agents=self._max_agents,
            default_agent_timeout_s=self._default_agent_timeout,
            agent_mock=agent_mock or self._agent_mock,
            args_payload=dict(args_payload),
            cwd_override=cwd,
            provider_override=provider,
            model_resolver=self._model_resolver,
        )

        # One store probe up front: a fresh run then skips all per-agent
        # journal lookups (the common, non-resume path).
        await run.journal.prime()

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

            if run._bg_tasks:
                await asyncio.gather(*run._bg_tasks, return_exceptions=True)
        self._last_wall_clock_s = time.monotonic() - t0

        result = _auto_parse(_coerce_result(raw))

        await self._emit_delivery(
            run,
            result,
            source_path=source_path,
            args_payload=args_payload,
        )

        return result
