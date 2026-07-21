# code-health: ignore-file[AM025] -- policy events deserialize untyped runtime payloads
"""Policy engine atom — DSL-driven policy detection and enforcement.

Deterministic rule evaluation over agent events. LLM reasoning activates
only when a rule escalates. See .claude/designs/policy-engine.md for the
full design.
"""

from __future__ import annotations

import asyncio
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import time

from loguru import logger

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.abi import (
    AgentMessage as AgentMessage,
    AtomAPI,
    AtomInstallPriority,
    BASH_OPERATIONS_SERVICE,
    BashOperations,
    BeforeRunEvent,
    BusPriority,
    FunctionTool,
    ImageContent,
    Model,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResult,
)
from agentm.core.abi.events import (
    BeforeSendEvent,
    ChildSessionStartEvent,
    ContextEvent,
    DiagnosticEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCommittedEvent,
)
from agentm.core.lib.tokens import count_text_tokens
from agentm.extensions import ExtensionManifest

from .entity import EntityRegistry
from .evaluator import PolicyEvaluator
from .ifc import IFCEngine
from .loader import load_policy_bundle, resolve_policy_path
from .persistence import PolicyPersistence
from .paths import default_policy_db_path
from .repository_index import (
    REPOSITORY_INDEX_SERVICE,
    RepositoryIndex,
    RepositoryRefreshPlan,
    repository_refresh_plan,
)
from .state import PolicyState, SessionEntry
from .types import RuleInstance, ToolArgs, ToolLogEntry


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class PolicyEngineConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    policy_files: list[str] = []
    extraction_mode: str = "heuristic"
    db_path: str | None = None
    ifg_realtime: bool = True
    ifg_repository_scan: bool = True
    ifg_repository_scan_timeout: float = Field(default=120.0, gt=0)
    ifg_repository_refresh_timeout: float = Field(default=30.0, gt=0)
    ifg_repository_search_roots: list[str] = Field(default_factory=list)


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _PolicyExplainArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rule: str = Field(min_length=1, description="Rule name")


# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------


MANIFEST = ExtensionManifest(
    name="policy_engine",
    description="DSL-driven policy detection and enforcement over agent events.",
    registers=(
        "tool:reload_policies",
        "tool:policy_explain",
        "tool:policy_stats",
        f"service:{REPOSITORY_INDEX_SERVICE}",
    ),
    config_schema=PolicyEngineConfig,
    requires=("service:operations:bash",),
    priority=AtomInstallPriority.POLICY,
)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def install(session: AtomAPI, config: PolicyEngineConfig) -> None:
    engine = _PolicyEngineRuntime(session, config)
    engine.install()


def _invalid_tool_call(tool_name: str, exc: ValidationError) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=f"Invalid {tool_name} call: {exc}")],
        is_error=True,
    )


def _tool_event_metadata(event: ToolResultEvent) -> dict[str, object]:
    metadata: dict[str, object] = {}
    extras = _result_extras(event.result)
    runtime = extras.get("agentm_runtime")
    runtime_extras = runtime if isinstance(runtime, Mapping) else {}

    exit_code = _first_int(
        extras.get("exit_code"),
        extras.get("return_code"),
        extras.get("returncode"),
        event.exit_code,
    )
    if exit_code is not None:
        metadata["exit_code"] = exit_code

    duration_ms = _first_int(
        extras.get("duration_ms"),
        event.duration_ms,
        runtime_extras.get("duration_ms"),
    )
    if duration_ms is not None:
        metadata["duration_ms"] = duration_ms

    result_content_hash = _first_str(
        extras.get("result_content_hash"),
        runtime_extras.get("result_content_hash"),
        event.result_content_hash,
    )
    if result_content_hash is not None:
        metadata["result_content_hash"] = result_content_hash

    for key in (
        "args_hash",
        "content_hash",
        "previous_content_hash",
        "file_op",
        "path",
        "timed_out",
        "timeout_s",
        "bytes",
        "stdout_lines",
        "stderr_lines",
    ):
        value = extras.get(key, runtime_extras.get(key))
        if value is not None:
            metadata[key] = value
    if event.result is not None:
        metadata["result_length"] = _tool_result_length(event.result)
    return metadata


def _raw_tool_result(result: ToolResult | None) -> dict[str, object]:
    if result is None:
        return {}
    return {
        "is_error": result.is_error,
        "content": [_content_block_json(block) for block in result.content],
        "extras": _mapping_or_value(result.extras),
    }


def _processed_tool_result(
    event: ToolResultEvent,
    text: str | None,
    error: str | None,
    metadata: Mapping[str, object],
) -> dict[str, object | None]:
    processed: dict[str, object | None] = {
        "is_error": event.result.is_error if event.result else None,
        "text_length": len(text or ""),
        "error": error,
        "exit_code": metadata.get("exit_code"),
        "duration_ms": metadata.get("duration_ms"),
        "result_content_hash": metadata.get("result_content_hash"),
        "content_hash": metadata.get("content_hash"),
        "previous_content_hash": metadata.get("previous_content_hash"),
        "result_length": metadata.get("result_length"),
    }
    for key in (
        "args_hash",
        "file_op",
        "path",
        "timed_out",
        "timeout_s",
        "bytes",
        "stdout_lines",
        "stderr_lines",
    ):
        if key in metadata:
            processed[key] = metadata[key]
    return processed


def _content_block_json(block: TextContent | ImageContent) -> dict[str, object]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    return {
        "type": "image",
        "mime_type": block.mime_type,
        "byte_length": len(block.data),
        "data_base64": base64.b64encode(block.data).decode("ascii"),
    }


def _mapping_or_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _mapping_or_value(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mapping_or_value(item) for item in value]
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, bytearray):
        return {
            "encoding": "base64",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    return value


def _model_name(model: Model | None) -> str | None:
    return model.id if model is not None else None


def _result_extras(result: ToolResult | None) -> Mapping[str, object]:
    if result is None or not isinstance(result.extras, Mapping):
        return {}
    return result.extras


def _first_int(*values: object) -> int | None:
    for value in values:
        coerced = _coerce_int(value)
        if coerced is not None:
            return coerced
    return None


def _coerce_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _first_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _tool_result_length(result: ToolResult) -> int:
    total = 0
    for block in result.content:
        if isinstance(block, TextContent):
            total += len(block.text)
        elif isinstance(block, ImageContent):
            total += len(block.data)
    return total


def _content_stats(
    blocks: Sequence[object],
    *,
    model_name: str | None,
) -> tuple[int, int, int, int]:
    total_chars = 0
    total_tokens = 0
    text_blocks = 0
    image_blocks = 0
    for block in blocks:
        if isinstance(block, TextContent):
            total_chars += len(block.text)
            total_tokens += count_text_tokens(block.text, model=model_name)
            text_blocks += 1
            continue
        if isinstance(block, ImageContent):
            image_blocks += 1
            continue
        if isinstance(block, ToolCallBlock):
            rendered = json.dumps(
                {
                    "name": block.name,
                    "arguments": _mapping_or_value(block.arguments),
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            total_chars += len(rendered)
            total_tokens += count_text_tokens(rendered, model=model_name)
            continue
        if isinstance(block, ToolResultBlock):
            chars, tokens, text_count, image_count = _content_stats(
                block.content,
                model_name=model_name,
            )
            total_chars += chars
            total_tokens += tokens
            text_blocks += text_count
            image_blocks += image_count
    return total_chars, total_tokens, text_blocks, image_blocks


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class _PolicyEngineRuntime:
    """Owns policy engine lifecycle for one session."""

    def __init__(self, session: AtomAPI, config: PolicyEngineConfig) -> None:
        self._session = session
        self._config = config
        self._state = PolicyState(cwd=session.ctx.cwd)
        self._entity_registry = EntityRegistry()
        self._ifc: IFCEngine | None = None
        self._evaluator: PolicyEvaluator = None  # type: ignore[assignment]
        self._rules: list[RuleInstance] = []
        self._persistence: PolicyPersistence | None = None
        self._pending_diagnostics: list[str] = []
        self._file_mtimes: dict[str, float] = {}
        self._pending_tool_args: dict[str, ToolArgs] = {}  # call_id → args
        self._ifg_pending_tool_args: dict[str, ToolArgs] = {}
        self._repository_index: RepositoryIndex | None = None
        self._repository_scan_task: asyncio.Task[bool] | None = None
        self._repository_refresh_plans: dict[str, RepositoryRefreshPlan] = {}
        self._persisted_effect_count = 0

    def install(self) -> None:
        self._load_rules()
        self._setup_ifc()
        self._setup_persistence()
        self._setup_repository_index()

        self._evaluator = PolicyEvaluator(self._rules, self._state)
        self._evaluator.set_entity_evidence_fn(self._entity_registry.entity_evidence)
        if self._ifc:
            self._evaluator.set_ifc_engine(self._ifc)
        if self._persistence:
            self._evaluator.set_cross_session_fn(self._persistence.count_cross_session)
            self._evaluator.set_eval_error_fn(self._record_eval_error)

        s = self._session
        s.on(ToolCallEvent.CHANNEL, self._on_tool_call, priority=BusPriority.PRE)
        s.on(
            ToolCallEvent.CHANNEL,
            self._on_repository_tool_call,
            priority=BusPriority.NORMAL,
        )
        s.on(ToolResultEvent.CHANNEL, self._on_tool_result, priority=BusPriority.PRE)
        s.on(
            ToolResultEvent.CHANNEL,
            self._on_repository_tool_result,
            priority=BusPriority.NORMAL,
        )
        s.on(
            ToolResultEvent.CHANNEL, self._on_tool_result_ifg, priority=BusPriority.POST
        )
        s.on(TurnCommittedEvent.CHANNEL, self._on_turn_committed)
        s.on(SessionReadyEvent.CHANNEL, self._on_repository_session_ready)
        s.on(
            BeforeRunEvent.CHANNEL,
            self._on_before_run_repository,
            priority=BusPriority.PRE,
        )
        s.on(ChildSessionStartEvent.CHANNEL, self._on_child_session_start)
        s.on(ContextEvent.CHANNEL, self._on_context)
        s.on(BeforeSendEvent.CHANNEL, self._on_before_send)
        s.on(
            SessionShutdownEvent.CHANNEL,
            self._on_repository_shutdown,
            priority=BusPriority.PRE,
        )
        s.on(SessionShutdownEvent.CHANNEL, self._on_session_shutdown)

        s.register_tool(
            FunctionTool(
                name="reload_policies",
                description="Reload policy rules from disk.",
                parameters=_NoArgs,
                fn=self._tool_reload,
            )
        )
        s.register_tool(
            FunctionTool(
                name="policy_explain",
                description="Explain why a specific policy rule did or didn't fire.",
                parameters=_PolicyExplainArgs,
                fn=self._tool_explain,
            )
        )
        s.register_tool(
            FunctionTool(
                name="policy_stats",
                description="Show policy engine statistics for the current session.",
                parameters=_NoArgs,
                fn=self._tool_stats,
            )
        )

        logger.info(
            "policy_engine installed: {} rules ({} enforce, {} observe)",
            len(self._rules),
            sum(1 for r in self._rules if r.mode == "enforce"),
            sum(1 for r in self._rules if r.mode == "observe"),
        )

    # --- Rule loading ---

    def _load_rules(self) -> None:
        bundle = load_policy_bundle(
            self._config.policy_files,
            cwd=Path(self._session.ctx.cwd),
        )
        self._rules = bundle.rules
        self._ifc = bundle.ifc
        self._file_mtimes = bundle.file_mtimes

    def _resolve_policy_path(self, file_ref: str) -> Path | None:
        return resolve_policy_path(file_ref, cwd=Path(self._session.ctx.cwd))

    # --- IFC setup ---

    def _setup_ifc(self) -> None:
        # _load_rules loads labels alongside the composed policy bundle.
        return

    # --- Persistence setup ---

    def _setup_persistence(self) -> None:
        db_path = self._config.db_path
        if not db_path:
            db_path = str(default_policy_db_path())
        self._persistence = PolicyPersistence(Path(db_path))
        try:
            self._persistence.open()
        except Exception as e:
            logger.warning("policy persistence failed to open: {}", e)
            self._persistence = None

    def _setup_repository_index(self) -> None:
        if not self._config.ifg_realtime or not self._config.ifg_repository_scan:
            return
        bash = self._session.services.get(BASH_OPERATIONS_SERVICE, BashOperations)
        if bash is None:
            logger.warning("policy repository index disabled: bash operations missing")
            return
        self._repository_index = RepositoryIndex(
            root=self._session.ctx.cwd,
            bash=bash,
            scan_timeout=self._config.ifg_repository_scan_timeout,
            refresh_timeout=self._config.ifg_repository_refresh_timeout,
            search_roots=self._config.ifg_repository_search_roots,
        )
        self._state.ifg_investigation.configure_repository(
            contains_file=self._repository_index.contains_file,
            canonicalize=self._repository_index.canonical_path,
        )
        self._session.services.register(
            REPOSITORY_INDEX_SERVICE,
            self._repository_index,
            scope="session",
        )

    # --- Event handlers ---

    def _on_repository_session_ready(self, event: SessionReadyEvent) -> None:
        del event
        self._start_repository_scan()

    def _start_repository_scan(self) -> asyncio.Task[bool] | None:
        if self._repository_index is None:
            return None
        if self._repository_scan_task is None:
            self._repository_scan_task = asyncio.create_task(
                self._scan_repository_on_mount(),
                name=f"policy-repository-scan-{self._session.ctx.session_id}",
            )
        return self._repository_scan_task

    async def _scan_repository_on_mount(self) -> bool:
        repository_index = self._repository_index
        if repository_index is None:
            return False
        with self._session.track_background():
            success = await repository_index.scan_all()
        status = repository_index.status
        if success:
            logger.info(
                "policy repository index ready: {} files under {}",
                status.files,
                status.root,
            )
        else:
            logger.warning(
                "policy repository index scan failed for {}: {}",
                status.root,
                status.last_error,
            )
        return success

    async def _on_before_run_repository(self, event: BeforeRunEvent) -> None:
        del event
        await self._ensure_repository_scanned()

    def _on_repository_tool_call(self, event: ToolCallEvent) -> None:
        if self._repository_index is None:
            return
        plan = repository_refresh_plan(
            tool_name=event.tool_name,
            args=event.args,
            cwd=self._session.ctx.cwd,
        )
        if plan is not None:
            self._repository_refresh_plans[event.tool_call_id] = plan

    async def _on_repository_tool_result(self, event: ToolResultEvent) -> None:
        repository_index = self._repository_index
        plan = self._repository_refresh_plans.pop(event.tool_call_id, None)
        if repository_index is None or event.result is None or event.result.is_error:
            return
        if plan is None:
            plan = repository_refresh_plan(
                tool_name=event.tool_name,
                args=event.args,
                cwd=self._session.ctx.cwd,
            )
        if plan is None:
            return
        await self._ensure_repository_scanned()
        success = await repository_index.refresh(plan)
        if not success:
            logger.warning(
                "policy repository index refresh failed ({}): {}",
                plan.reason,
                repository_index.status.last_error,
            )

    async def _ensure_repository_scanned(self) -> bool:
        repository_index = self._repository_index
        if repository_index is None:
            return False
        task = self._start_repository_scan()
        return await asyncio.shield(task) if task is not None else False

    async def _on_repository_shutdown(self, event: SessionShutdownEvent) -> None:
        del event
        task = self._repository_scan_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._repository_refresh_plans.clear()

    def _on_tool_call(self, event: ToolCallEvent) -> dict[str, object] | None:
        """Handle tool_call_pre: run reactions + evaluate rules."""
        # Entity extraction (Layer 1: structural from args)
        self._entity_registry.extract_structural(
            event.tool_name, event.args, self._state.turn_count
        )

        # Cache args for tool_result phase
        self._pending_tool_args[event.tool_call_id] = event.args
        self._ifg_pending_tool_args[event.tool_call_id] = event.args

        # Evaluate rules on tool_call channel
        effects = self._evaluator.evaluate(
            channel="tool_call",
            tool_name=event.tool_name,
            args=event.args,
        )
        self._persist_new_effects()
        self._queue_tool_event_pre(event)
        self._queue_state_snapshot()
        self._flush_policy_persistence()
        self._persist_ifg_tool_event_pre(event)

        if not effects:
            return None

        for _rule_id, effect_type, diagnostic in effects:
            if effect_type == "abort":
                raise RuntimeError(f"[PolicyEngine:abort] {diagnostic}")
            if effect_type == "block":
                return {"block": True, "reason": diagnostic}
            # escalate/notify → queue for context injection
            self._pending_diagnostics.append(diagnostic)

        return None

    def _on_tool_result(self, event: ToolResultEvent) -> None:
        """Handle tool_call_post: record state + evaluate rules."""
        text: str | None = None
        error: str | None = None
        if event.result and event.result.content:
            text_parts = [
                c.text for c in event.result.content if isinstance(c, TextContent)
            ]
            text = "\n".join(text_parts) if text_parts else None
            if event.result.is_error:
                error = text or "unknown error"

        # Retrieve cached args from tool_call phase
        args = dict(event.args or self._pending_tool_args.pop(event.tool_call_id, {}))
        self._pending_tool_args.pop(event.tool_call_id, None)
        metadata = _tool_event_metadata(event)
        raw_result = _raw_tool_result(event.result)

        result_for_state: dict[str, object] = {}
        if text:
            result_for_state["text"] = text
        if error:
            result_for_state["error"] = error
        result_for_state.update(metadata)

        self._state.record_tool_call(
            tool_name=event.tool_name,
            args=args,
            result=result_for_state,
        )

        # Entity evidence from result
        if error:
            self._entity_registry.record_tool_failure(
                event.tool_name,
                args,
                self._state.turn_count,
                error,
            )
        else:
            self._entity_registry.record_tool_success(
                event.tool_name,
                args,
                self._state.turn_count,
            )

        # IFC: check tool result for taint sources
        if self._ifc and text:
            self._ifc.process_tool_result(event.tool_name, text)

        # Lexical extraction from result text (Layer 2)
        if text:
            self._entity_registry.extract_lexical(
                text, self._state.turn_count, "tool_result"
            )

        # Evaluate rules on tool_result channel
        result_for_eval: dict[str, str | None] = {"text": text, "error": error}
        effects = self._evaluator.evaluate(
            channel="tool_result",
            tool_name=event.tool_name,
            args=args,
            result=result_for_eval,
        )
        self._persist_new_effects()
        self._queue_tool_event_post(
            event,
            args,
            raw_result=raw_result,
            processed=_processed_tool_result(event, text, error, metadata),
        )
        self._queue_state_snapshot()
        self._queue_session_summary()
        self._flush_policy_persistence()

        for _rule_id, effect_type, diagnostic in effects:
            if effect_type == "abort":
                raise RuntimeError(f"[PolicyEngine:abort] {diagnostic}")
            self._pending_diagnostics.append(diagnostic)

    def _on_tool_result_ifg(self, event: ToolResultEvent) -> None:
        """Persist IFG facts after result handlers have had a chance to run."""
        if not self._persistence or not self._config.ifg_realtime:
            return
        text: str | None = None
        error: str | None = None
        if event.result and event.result.content:
            text_parts = [
                c.text for c in event.result.content if isinstance(c, TextContent)
            ]
            text = "\n".join(text_parts) if text_parts else None
            if event.result.is_error:
                error = text or "unknown error"
        args = dict(
            event.args or self._ifg_pending_tool_args.pop(event.tool_call_id, {})
        )
        self._ifg_pending_tool_args.pop(event.tool_call_id, None)
        metadata = _tool_event_metadata(event)
        raw_result = _raw_tool_result(event.result)
        processed = _processed_tool_result(event, text, error, metadata)
        entry = self._latest_tool_log_entry()
        if entry is not None:
            processed["error_fingerprint"] = entry.error_fingerprint
            processed["error_category"] = entry.error_category
        self._persist_ifg_tool_event_post(
            event,
            args,
            raw_result=raw_result,
            processed=processed,
        )

    def _on_turn_committed(self, event: TurnCommittedEvent) -> None:
        """Handle turn end: advance turn, record summary, evaluate rules, flush."""
        self._state.advance_turn()
        turn_idx = self._state.turn_count

        # Record turn_summary

        recent_entries = [
            e for e in self._state.tool_log.entries() if e.get("turn") == turn_idx
        ]
        turn_summary = {
            "turn_index": turn_idx,
            "tool_calls_count": len(recent_entries),
            "tool_names_set": {e.get("tool") for e in recent_entries},
            "error_count": sum(1 for e in recent_entries if e.get("error")),
            "files_modified": {
                e.get("path")
                for e in recent_entries
                if e.get("path") and e.get("tool") in ("edit", "write")
            },
        }
        self._state.turn_summary.record(turn_idx, turn_summary)

        # Evaluate rules
        effects = self._evaluator.evaluate(
            channel="turn_committed",
            tool_name="",
        )
        self._persist_new_effects()
        self._queue_turn_summary(turn_idx, turn_summary)
        self._queue_state_snapshot()
        self._flush_policy_persistence()

        for _rule_id, effect_type, diagnostic in effects:
            if effect_type == "abort":
                raise RuntimeError(f"[PolicyEngine:abort] {diagnostic}")
            self._pending_diagnostics.append(diagnostic)

        # Hot reload: check policy file mtime
        self._check_hot_reload()

    def _on_child_session_start(self, event: ChildSessionStartEvent) -> None:
        """Handle child session spawn."""
        self._state.session_tree.record_spawn(
            event.child_session_id,
            SessionEntry(
                session_id=event.child_session_id,
                parent_id=event.parent_session_id,
                purpose=event.purpose,
                concurrent_siblings=self._state.session_tree.concurrent_children(
                    event.parent_session_id
                ),
            ),
        )

        effects = self._evaluator.evaluate(
            channel="child_session_start",
            tool_name="",
        )
        self._persist_new_effects()
        self._queue_state_snapshot()
        self._flush_policy_persistence()
        for _rule_id, _effect_type, diagnostic in effects:
            self._pending_diagnostics.append(diagnostic)

    def _on_context(self, event: ContextEvent) -> list | None:
        """Inject pending diagnostics + record context_state."""
        # Record context_state

        context_state = self._context_state(
            event.turn_index,
            event.messages,
            model=None,
        )
        self._state.context_state.record(event.turn_index, context_state)

        if not self._pending_diagnostics:
            self._queue_context_state(event.turn_index, context_state)
            self._queue_state_snapshot()
            self._flush_policy_persistence()
            return None

        diagnostics = self._pending_diagnostics[:]
        self._pending_diagnostics.clear()

        # Dict recall (Layer 3) on the context messages
        for msg in event.messages:
            for block in msg.content:
                if isinstance(block, TextContent):
                    self._entity_registry.extract_dict_recall(
                        block.text, self._state.turn_count
                    )

        self._queue_context_state(event.turn_index, context_state)
        self._queue_state_snapshot()
        self._flush_policy_persistence()

        # Emit diagnostics via DiagnosticEvent for observability
        for diag in diagnostics:
            self._session.bus.emit_sync(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(level="warning", source="policy_engine", message=diag),
            )

        # Inject as additional context (appended to last message)
        from agentm.core.abi.messages import UserMessage

        combined = "\n\n".join(diagnostics)
        injection = UserMessage(
            role="user",
            content=(TextContent(type="text", text=f"\n---\n{combined}\n---"),),
            timestamp=0.0,
        )
        messages = list(event.messages) + [injection]
        return messages

    def _on_before_send(self, event: BeforeSendEvent) -> None:
        """Record final provider-facing context usage after context transforms."""
        context_state = self._context_state(
            event.turn_index,
            event.messages,
            model=event.model,
        )
        self._state.context_state.record(event.turn_index, context_state)
        self._queue_context_state(event.turn_index, context_state)
        self._flush_policy_persistence()

    def _on_session_shutdown(self, event: SessionShutdownEvent) -> None:
        """Persist a session-level aggregate snapshot on shutdown."""
        del event
        if not self._persistence:
            return
        self._persist_new_effects()
        self._queue_state_snapshot()
        self._queue_session_summary()
        self._persistence.flush()
        self._persistence.close()
        self._persistence = None

    def _context_state(
        self,
        turn_index: int,
        messages: tuple[AgentMessage, ...],
        *,
        model: Model | None,
    ) -> dict[str, object]:
        total_chars = 0
        total_tokens = 0
        text_blocks = 0
        image_blocks = 0
        hidden_messages = 0
        synthetic_messages = 0
        model_name = _model_name(model)
        for message in messages:
            if message.meta.visibility != "visible":
                hidden_messages += 1
            if message.meta.synthetic:
                synthetic_messages += 1
            chars, tokens, text_count, image_count = _content_stats(
                message.content,
                model_name=model_name,
            )
            total_chars += chars
            total_tokens += tokens
            text_blocks += text_count
            image_blocks += image_count
        context_window = model.context_window if model is not None else None
        max_output_tokens = model.max_output_tokens if model is not None else None
        input_budget = None
        context_usage_pct = None
        if context_window is not None:
            input_budget = context_window
            if max_output_tokens is not None:
                input_budget = max(1, context_window - max_output_tokens)
            context_usage_pct = round((total_tokens / input_budget) * 100, 2)
        return {
            "turn_index": turn_index,
            "total_context_tokens": total_tokens,
            "total_context_chars": total_chars,
            "context_usage_pct": context_usage_pct,
            "context_window": context_window,
            "max_output_tokens": max_output_tokens,
            "input_budget_tokens": input_budget,
            "compaction_happened": False,
            "turns_since_user_message": 0,
            "injected_content_count": len(self._pending_diagnostics),
            "message_count": len(messages),
            "text_block_count": text_blocks,
            "image_block_count": image_blocks,
            "hidden_message_count": hidden_messages,
            "synthetic_message_count": synthetic_messages,
            "model": model_name,
        }

    def _session_summary(self) -> dict[str, object]:
        tool_entries = list(self._state.tool_log.entries())
        effect_entries = list(self._state.effect_log.entries())
        context_entries = list(self._state.context_state.entries())
        file_entries = list(self._state.file_state.entries())
        tool_counts = Counter(entry.tool for entry in tool_entries)
        effect_rule_counts = Counter(entry.rule_id for entry in effect_entries)
        effect_type_counts = Counter(entry.effect for entry in effect_entries)
        exit_code_counts = Counter(
            str(entry.exit_code)
            for entry in tool_entries
            if entry.exit_code is not None
        )
        durations = [entry.duration_ms for entry in tool_entries]
        duration_total = sum(durations)
        max_context_tokens = (
            max(
                (_coerce_int(entry.get("total_context_tokens")) or 0)
                for entry in context_entries
            )
            if context_entries
            else 0
        )
        max_context_usage_pct = max(
            (
                float(value)
                for entry in context_entries
                if isinstance((value := entry.get("context_usage_pct")), (int, float))
            ),
            default=0.0,
        )
        return {
            "session_id": self._session.ctx.session_id,
            "turn_count": self._state.turn_count,
            "tool_call_count": len(tool_entries),
            "tool_error_count": sum(1 for entry in tool_entries if entry.error),
            "tool_log_entries": len(tool_entries),
            "effect_log_entries": len(effect_entries),
            "entities_tracked": len(self._entity_registry.entries()),
            "tool_counts": dict(tool_counts),
            "exit_code_counts": dict(exit_code_counts),
            "duration_ms_total": duration_total,
            "duration_ms_avg": (
                round(duration_total / len(durations), 2) if durations else 0.0
            ),
            "duration_ms_max": max(durations, default=0),
            "effect_rule_counts": dict(effect_rule_counts),
            "effect_type_counts": dict(effect_type_counts),
            "file_count": len(file_entries),
            "file_read_count": sum(entry.read_count for entry in file_entries),
            "file_write_count": sum(entry.write_count for entry in file_entries),
            "context_snapshot_count": len(context_entries),
            "max_context_tokens": max_context_tokens,
            "max_context_usage_pct": max_context_usage_pct,
            "error_count": sum(1 for entry in tool_entries if entry.error),
            "intervention": self._state.ifg_interventions.summary(),
            "investigation": self._state.ifg_investigation.summary(),
        }

    def _queue_session_summary(self) -> None:
        if self._persistence:
            self._persistence.queue_session_summary(
                self._session.ctx.session_id,
                self._session_summary(),
            )

    # --- Hot reload ---

    def _check_hot_reload(self) -> None:
        """Check if any policy file has been modified and reload if so."""
        for file_ref in self._config.policy_files:
            path = self._resolve_policy_path(file_ref)
            if not path or not path.exists():
                continue
            mtime = path.stat().st_mtime
            prev = self._file_mtimes.get(str(path))
            if prev is not None and mtime > prev:
                logger.info("policy file changed: {}, reloading", path)
                self._do_reload()
                return
            self._file_mtimes[str(path)] = mtime

    def _do_reload(self) -> None:
        """Reload all rules and rebuild evaluator."""
        self._rules.clear()
        self._load_rules()
        self._evaluator = PolicyEvaluator(self._rules, self._state)
        self._evaluator.set_entity_evidence_fn(self._entity_registry.entity_evidence)
        if self._ifc:
            self._evaluator.set_ifc_engine(self._ifc)
        if self._persistence:
            self._evaluator.set_cross_session_fn(self._persistence.count_cross_session)
            self._evaluator.set_eval_error_fn(self._record_eval_error)
        logger.info("policy engine reloaded: {} rules", len(self._rules))

    # --- Persistence helpers ---

    def _record_eval_error(
        self,
        rule_id: str,
        channel: str,
        tool_name: str,
        error: str,
    ) -> None:
        if not self._persistence:
            return
        self._persistence.queue_eval_error(
            session_id=self._session.ctx.session_id,
            turn=self._state.turn_count,
            rule_id=rule_id,
            channel=channel,
            tool_name=tool_name,
            error=error,
        )

    def _persist_new_effects(self) -> None:
        if not self._persistence:
            return
        records = list(self._state.effect_log.entries())
        for rec in records[self._persisted_effect_count :]:
            self._persistence.queue_effect(self._session.ctx.session_id, rec)
        self._persisted_effect_count = len(records)

    def _queue_tool_event_pre(self, event: ToolCallEvent) -> None:
        if not self._persistence:
            return
        self._persistence.queue_tool_event(
            session_id=self._session.ctx.session_id,
            turn=self._state.turn_count,
            phase="pre",
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            args=event.args,
            taint_labels=self._taint_labels(event.args),
            cwd=self._session.ctx.cwd,
        )

    def _queue_tool_event_post(
        self,
        event: ToolResultEvent,
        args: ToolArgs,
        *,
        raw_result: dict[str, object],
        processed: dict[str, object | None],
    ) -> None:
        if not self._persistence:
            return
        entry = self._latest_tool_log_entry()
        if entry is not None:
            processed["error_fingerprint"] = entry.error_fingerprint
            processed["error_category"] = entry.error_category
        self._persistence.queue_tool_event(
            session_id=self._session.ctx.session_id,
            turn=self._state.turn_count,
            phase="post",
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            args=args,
            entry=entry,
            result=raw_result,
            processed=processed,
            cwd=self._session.ctx.cwd,
        )

    def _queue_context_state(
        self,
        turn_index: int,
        context_state: dict[str, object],
    ) -> None:
        if self._persistence:
            self._persistence.queue_context_state(
                self._session.ctx.session_id,
                turn_index,
                context_state,
            )

    def _queue_turn_summary(
        self,
        turn_index: int,
        turn_summary: dict[str, object],
    ) -> None:
        if self._persistence:
            self._persistence.queue_turn_summary(
                self._session.ctx.session_id,
                turn_index,
                turn_summary,
            )

    def _queue_state_snapshot(self) -> None:
        if not self._persistence:
            return
        session_id = self._session.ctx.session_id
        self._persistence.queue_file_state_snapshot(
            session_id,
            self._state.file_state.entries(),
        )
        self._persistence.queue_entity_state_snapshot(
            session_id,
            self._entity_registry.entries(),
        )

    def _flush_policy_persistence(self) -> None:
        if self._persistence:
            self._persistence.flush()

    def _persist_ifg_tool_event_pre(self, event: ToolCallEvent) -> None:
        if not self._persistence or not self._config.ifg_realtime:
            return
        self._persist_ifg_tool_event(
            phase="pre",
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            args=event.args,
            result={},
            processed={},
            state={"taint_labels": list(self._taint_labels(event.args))},
            raw_evidence={
                "source": "ToolCallEvent",
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "args": event.args,
            },
        )

    def _persist_ifg_tool_event_post(
        self,
        event: ToolResultEvent,
        args: ToolArgs,
        *,
        raw_result: Mapping[str, object],
        processed: Mapping[str, object | None],
    ) -> None:
        if not self._persistence or not self._config.ifg_realtime:
            return
        self._persist_ifg_tool_event(
            phase="post",
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            args=args,
            result=raw_result,
            processed=processed,
            state={"processed": dict(processed)},
            raw_evidence={
                "source": "ToolResultEvent",
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "args": args,
                "result": raw_result,
                "processed": dict(processed),
            },
        )

    def _persist_ifg_tool_event(
        self,
        *,
        phase: str,
        tool_call_id: str,
        tool_name: str,
        args: Mapping[str, object],
        result: Mapping[str, object],
        processed: Mapping[str, object | None],
        state: Mapping[str, object],
        raw_evidence: Mapping[str, object],
    ) -> None:
        from .ifg import IfgToolEvent

        persistence = self._persistence
        if persistence is None:
            return

        event = IfgToolEvent(
            session_id=self._session.ctx.session_id,
            turn=self._state.turn_count,
            event_id=None,
            tool_call_id=tool_call_id,
            phase=phase,
            tool_name=tool_name,
            args=dict(args),
            result=dict(result),
            processed=dict(processed),
            state=dict(state),
            cwd=self._session.ctx.cwd,
            ts=time.time(),
            source="policy_runtime",
            raw_evidence=dict(raw_evidence),
        )
        persistence.persist_ifg_tool_events(
            (event,),
            repository_index=self._repository_index,
        )

    def _taint_labels(self, args: ToolArgs) -> tuple[str, ...]:
        if not self._ifc:
            return ()
        return tuple(sorted(self._ifc.check_event_taint(args)))

    def _latest_tool_log_entry(self) -> ToolLogEntry | None:
        entries = self._state.tool_log.entries()
        return entries[-1] if entries else None

    # --- Tools ---

    async def _tool_reload(self, args: dict[str, object]) -> ToolResult:
        try:
            _NoArgs.model_validate(args)
        except ValidationError as exc:
            return _invalid_tool_call("reload_policies", exc)
        self._do_reload()
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Reloaded {len(self._rules)} rules.",
                )
            ]
        )

    async def _tool_explain(self, args: dict[str, object]) -> ToolResult:
        try:
            parsed = _PolicyExplainArgs.model_validate(args)
        except ValidationError as exc:
            return _invalid_tool_call("policy_explain", exc)
        rule_name = parsed.rule
        for rule in self._rules:
            if rule.rule_id == rule_name:
                info = (
                    f"Rule: {rule.rule_id}\n"
                    f"Layer: L{rule.layer}\n"
                    f"Mode: {rule.mode}\n"
                    f"Channel: {rule.channel}\n"
                    f"Effect: {rule.effect.effect}\n"
                    f"Cooldown: {rule.cooldown_turns} turns\n"
                    f"Last fired: turn {rule.last_fired_turn}\n"
                    f"Guard: tools={rule.guard.tool_names}"
                )
                return ToolResult(content=[TextContent(type="text", text=info)])
        return ToolResult(
            content=[TextContent(type="text", text=f"Rule '{rule_name}' not found.")],
            is_error=True,
        )

    async def _tool_stats(self, args: dict[str, object]) -> ToolResult:
        try:
            _NoArgs.model_validate(args)
        except ValidationError as exc:
            return _invalid_tool_call("policy_stats", exc)
        lines = [
            f"Turn: {self._state.turn_count}",
            f"Rules: {len(self._rules)} ({sum(1 for r in self._rules if r.mode == 'enforce')} enforce, "
            f"{sum(1 for r in self._rules if r.mode == 'observe')} observe)",
            f"Tool log entries: {len(self._state.tool_log)}",
            f"Effect log entries: {len(self._state.effect_log.entries())}",
            f"Entities tracked: {len(self._entity_registry._entities)}",
        ]
        if self._repository_index is not None:
            status = self._repository_index.status
            lines.append(
                "Repository index: "
                f"{'ready' if status.ready else 'warming'} "
                f"({status.files} files, {status.scans} scans, "
                f"{status.refreshes} refreshes)"
            )
        lines.extend(("", "Recent firings:"))
        for rec in list(self._state.effect_log.entries())[-5:]:
            lines.append(
                f"  [{rec.mode}] {rec.rule_id} → {rec.effect} (turn {rec.turn})"
            )
        return ToolResult(content=[TextContent(type="text", text="\n".join(lines))])
