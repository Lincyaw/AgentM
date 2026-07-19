"""Policy engine atom — DSL-driven policy detection and enforcement.

Deterministic rule evaluation over agent events. LLM reasoning activates
only when a rule escalates. See .claude/designs/policy-engine.md for the
full design.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from pydantic import BaseModel, ConfigDict

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    BusPriority,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.core.abi.events import (
    ChildSessionStartEvent,
    ContextEvent,
    DiagnosticEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCommittedEvent,
)
from agentm.extensions import ExtensionManifest

from .compiler import compile_policy_file, compose_rules
from .entity import EntityRegistry
from .evaluator import PolicyEvaluator
from .ifc import IFCEngine, compile_labels
from .persistence import PolicyPersistence
from .state import PolicyState, SessionEntry
from .types import RuleInstance, ToolArgs



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class PolicyEngineConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    policy_files: list[str] = []
    extraction_mode: str = "heuristic"
    db_path: str | None = None


# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------


MANIFEST = ExtensionManifest(
    name="policy_engine",
    description="DSL-driven policy detection and enforcement over agent events.",
    registers=("tool:reload_policies", "tool:policy_explain", "tool:policy_stats"),
    config_schema=PolicyEngineConfig,
    requires=(),
    priority=AtomInstallPriority.POLICY,
)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def install(session: AtomAPI, config: PolicyEngineConfig) -> None:
    engine = _PolicyEngineRuntime(session, config)
    engine.install()


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class _PolicyEngineRuntime:
    """Owns policy engine lifecycle for one session."""

    def __init__(self, session: AtomAPI, config: PolicyEngineConfig) -> None:
        self._session = session
        self._config = config
        self._state = PolicyState()
        self._entity_registry = EntityRegistry()
        self._ifc: IFCEngine | None = None
        self._evaluator: PolicyEvaluator = None  # type: ignore[assignment]
        self._rules: list[RuleInstance] = []
        self._persistence: PolicyPersistence | None = None
        self._pending_diagnostics: list[str] = []
        self._file_mtimes: dict[str, float] = {}
        self._pending_tool_args: dict[str, ToolArgs] = {}  # call_id → args

    def install(self) -> None:
        self._load_rules()
        self._setup_ifc()
        self._setup_persistence()

        self._evaluator = PolicyEvaluator(self._rules, self._state)
        self._evaluator.set_entity_evidence_fn(self._entity_registry.entity_evidence)
        if self._ifc:
            self._evaluator.set_ifc_engine(self._ifc)
        if self._persistence:
            self._evaluator.set_cross_session_fn(
                self._persistence.count_cross_session
            )

        s = self._session
        s.on(ToolCallEvent.CHANNEL, self._on_tool_call, priority=BusPriority.PRE)
        s.on(ToolResultEvent.CHANNEL, self._on_tool_result, priority=BusPriority.PRE)
        s.on(TurnCommittedEvent.CHANNEL, self._on_turn_committed)
        s.on(ChildSessionStartEvent.CHANNEL, self._on_child_session_start)
        s.on(ContextEvent.CHANNEL, self._on_context)

        s.register_tool(FunctionTool(
            name="reload_policies",
            description="Reload policy rules from disk.",
            parameters={"type": "object", "properties": {}},
            fn=self._tool_reload,
        ))
        s.register_tool(FunctionTool(
            name="policy_explain",
            description="Explain why a specific policy rule did or didn't fire.",
            parameters={
                "type": "object",
                "properties": {"rule": {"type": "string", "description": "Rule name"}},
                "required": ["rule"],
            },
            fn=self._tool_explain,
        ))
        s.register_tool(FunctionTool(
            name="policy_stats",
            description="Show policy engine statistics for the current session.",
            parameters={"type": "object", "properties": {}},
            fn=self._tool_stats,
        ))

        logger.info(
            "policy_engine installed: {} rules ({} enforce, {} observe)",
            len(self._rules),
            sum(1 for r in self._rules if r.mode == "enforce"),
            sum(1 for r in self._rules if r.mode == "observe"),
        )

    # --- Rule loading ---

    def _load_rules(self) -> None:
        layers: list[tuple[list[RuleInstance], list[str]]] = []

        # Layer 0: base policy (ships with the engine)
        base_path = Path(__file__).parent / "base_policy.yaml"
        if base_path.exists():
            content = base_path.read_text(encoding="utf-8")
            rules, disabled = compile_policy_file(content)
            layers.append((rules, disabled))
            self._file_mtimes[str(base_path)] = base_path.stat().st_mtime

        # Layer 1+: user/scenario policy files
        for file_ref in self._config.policy_files:
            path = self._resolve_policy_path(file_ref)
            if not path or not path.exists():
                logger.warning("policy file not found: {}", file_ref)
                continue
            content = path.read_text(encoding="utf-8")
            rules, disabled = compile_policy_file(content)
            layers.append((rules, disabled))
            self._file_mtimes[str(path)] = path.stat().st_mtime
            logger.debug("loaded {} rules from {}", len(rules), path)

        self._rules = compose_rules(layers)

    def _resolve_policy_path(self, file_ref: str) -> Path | None:
        p = Path(file_ref)
        if p.is_absolute() and p.exists():
            return p
        cwd = Path(self._session.ctx.cwd)
        candidate = cwd / file_ref
        if candidate.exists():
            return candidate
        home_candidate = Path.home() / ".agentm" / "policies" / file_ref
        if home_candidate.exists():
            return home_candidate
        return None

    # --- IFC setup ---

    def _setup_ifc(self) -> None:
        for file_ref in self._config.policy_files:
            path = self._resolve_policy_path(file_ref)
            if not path or not path.exists():
                continue
            import yaml
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.debug("failed to parse policy YAML {}: {}", path, e)
                continue
            if isinstance(data, dict) and "labels" in data:
                labels = compile_labels(data["labels"])
                if labels:
                    self._ifc = IFCEngine(labels)
                    break

    # --- Persistence setup ---

    def _setup_persistence(self) -> None:
        db_path = self._config.db_path
        if not db_path:
            db_dir = Path.home() / ".agentm" / "policy_state"
            db_path = str(db_dir / "policy.db")
        self._persistence = PolicyPersistence(Path(db_path))
        try:
            self._persistence.open()
        except Exception as e:
            logger.warning("policy persistence failed to open: {}", e)
            self._persistence = None

    # --- Event handlers ---

    def _on_tool_call(self, event: ToolCallEvent) -> dict[str, object] | None:
        """Handle tool_call_pre: run reactions + evaluate rules."""
        # Entity extraction (Layer 1: structural from args)
        self._entity_registry.extract_structural(
            event.tool_name, event.args, self._state.turn_count
        )

        # Cache args for tool_result phase
        self._pending_tool_args[event.tool_call_id] = event.args

        # Evaluate rules on tool_call channel
        effects = self._evaluator.evaluate(
            channel="tool_call",
            tool_name=event.tool_name,
            args=event.args,
        )

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
            for c in event.result.content:
                if isinstance(c, TextContent):
                    text = c.text
            if event.result.is_error:
                error = text or "unknown error"

        # Retrieve cached args from tool_call phase
        args = self._pending_tool_args.pop(event.tool_call_id, {})

        result_for_state: dict[str, object] = {}
        if text:
            result_for_state["text"] = text
        if error:
            result_for_state["error"] = error

        self._state.record_tool_call(
            tool_name=event.tool_name,
            args=args,
            result=result_for_state,
        )

        # Entity evidence from result
        if error:
            self._entity_registry.record_tool_failure(
                event.tool_name, args, self._state.turn_count, error,
            )
        else:
            self._entity_registry.record_tool_success(
                event.tool_name, args, self._state.turn_count,
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
            result=result_for_eval,
        )

        for _rule_id, effect_type, diagnostic in effects:
            if effect_type == "abort":
                raise RuntimeError(f"[PolicyEngine:abort] {diagnostic}")
            self._pending_diagnostics.append(diagnostic)

    def _on_turn_committed(self, event: TurnCommittedEvent) -> None:
        """Handle turn end: advance turn, record summary, evaluate rules, flush."""
        self._state.advance_turn()
        turn_idx = self._state.turn_count

        # Record turn_summary
        
        recent_entries = [
            e for e in self._state.tool_log.entries()
            if e.get("turn") == turn_idx
        ]
        self._state.turn_summary.record(turn_idx, dict(
            turn_index=turn_idx,
            tool_calls_count=len(recent_entries),
            tool_names_set={e.get("tool") for e in recent_entries},
            error_count=sum(1 for e in recent_entries if e.get("error")),
            files_modified={
                e.get("path") for e in recent_entries
                if e.get("path") and e.get("tool") in ("edit", "write")
            },
        ))

        # Evaluate rules
        effects = self._evaluator.evaluate(
            channel="turn_committed",
            tool_name="",
        )

        for _rule_id, effect_type, diagnostic in effects:
            if effect_type == "abort":
                raise RuntimeError(f"[PolicyEngine:abort] {diagnostic}")
            self._pending_diagnostics.append(diagnostic)

        # Flush persistence at turn boundary
        if self._persistence:
            for rec in self._state.effect_log.entries():
                if rec.turn == turn_idx:
                    session_id = self._session.ctx.session_id
                    self._persistence.queue_effect(session_id, rec)
            self._persistence.flush()

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
        for _rule_id, _effect_type, diagnostic in effects:
            self._pending_diagnostics.append(diagnostic)

    def _on_context(self, event: ContextEvent) -> list | None:
        """Inject pending diagnostics + record context_state."""
        # Record context_state
        
        total_chars = sum(
            sum(len(c.text) for c in m.content if isinstance(c, TextContent))
            for m in event.messages
        )
        self._state.context_state.record(event.turn_index, dict(
            turn_index=event.turn_index,
            total_context_tokens=total_chars // 4,  # rough token estimate
            context_usage_pct=0.0,
            compaction_happened=False,
            turns_since_user_message=0,
            injected_content_count=len(self._pending_diagnostics),
        ))

        if not self._pending_diagnostics:
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
            self._evaluator.set_cross_session_fn(
                self._persistence.count_cross_session
            )
        logger.info("policy engine reloaded: {} rules", len(self._rules))

    # --- Tools ---

    async def _tool_reload(self, args: dict[str, object]) -> ToolResult:
        self._do_reload()
        return ToolResult(content=[TextContent(
            type="text",
            text=f"Reloaded {len(self._rules)} rules.",
        )])

    async def _tool_explain(self, args: dict[str, object]) -> ToolResult:
        rule_name = args.get("rule", "")
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
        lines = [
            f"Turn: {self._state.turn_count}",
            f"Rules: {len(self._rules)} ({sum(1 for r in self._rules if r.mode == 'enforce')} enforce, "
            f"{sum(1 for r in self._rules if r.mode == 'observe')} observe)",
            f"Tool log entries: {len(self._state.tool_log)}",
            f"Effect log entries: {len(self._state.effect_log.entries())}",
            f"Entities tracked: {len(self._entity_registry._entities)}",
            "",
            "Recent firings:",
        ]
        for rec in list(self._state.effect_log.entries())[-5:]:
            lines.append(f"  [{rec.mode}] {rec.rule_id} → {rec.effect} (turn {rec.turn})")
        return ToolResult(content=[TextContent(type="text", text="\n".join(lines))])
