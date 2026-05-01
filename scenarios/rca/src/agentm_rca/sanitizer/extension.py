"""RCA investigation sanitizer extension.

Ports the old ``SanitizerMiddleware`` state machine onto the AgentM extension
bus. The tracker lives for the full session; event handlers observe tool
traffic, schedule sanitizer checks at round boundaries, and inject sanitizer
feedback into the next LLM call as a synthetic user message.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from agentm.core.kernel import (
    AssistantMessage,
    TextContent,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)
from agentm.extensions import ExtensionManifest
from agentm.harness.events import SessionShutdownEvent
from agentm.harness.extension import ExtensionAPI

from agentm_rca.sanitizer.code_sanitizer import CodeSanitizer
from agentm_rca.sanitizer.critic_sanitizer import CriticSanitizer
from agentm_rca.sanitizer.models import SanitizerContext, SanitizerFinding, Severity
from agentm_rca.sanitizer.tracker import InvestigationTracker
from agentm_rca.stores import HypothesisStore, ServiceProfileStore

_FINALIZE_RE = re.compile(r"<decision>\s*finalize\s*</decision>", re.IGNORECASE)

MANIFEST = ExtensionManifest(
    name="sanitizer",
    description="Track RCA investigation progress and inject sanitizer feedback between rounds.",
    registers=(
        "event:turn_start",
        "event:turn_end",
        "event:tool_call",
        "event:tool_result",
        "event:context",
        "event:session_shutdown",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "hypothesis_store": {"type": "object"},
            "profile_store": {"type": "object"},
            "severity_map": {"type": "object"},
            "disabled": {"type": "array", "items": {"type": "string"}},
            "drift_window": {"type": "integer", "minimum": 1},
            "periodic_interval": {"type": "integer", "minimum": 0},
            "max_block_retries": {"type": "integer", "minimum": 0},
            "tool_call_budget": {"type": ["integer", "null"], "minimum": 0},
            "max_steps": {"type": ["integer", "null"], "minimum": 1},
            "critic_model": {},
            "critic_disabled": {"type": "array", "items": {"type": "string"}},
            "critic_severity_map": {"type": "object"},
        },
        "required": ["hypothesis_store", "profile_store"],
        "additionalProperties": False,
    },
)


@dataclass(slots=True)
class _DispatchMeta:
    task_type: str
    hypothesis_id: str


class _SanitizerExtension:
    def __init__(
        self,
        *,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        code_sanitizer: CodeSanitizer,
        critic_sanitizer: CriticSanitizer | None,
        periodic_interval: int,
        max_block_retries: int,
        tool_call_budget: int | None,
        max_steps: int | None,
    ) -> None:
        self._hypothesis_store = hypothesis_store
        self._profile_store = profile_store
        self._code_sanitizer = code_sanitizer
        self._critic_sanitizer = critic_sanitizer
        self._tracker = InvestigationTracker()
        self._periodic_interval = periodic_interval
        self._max_block_retries = max_block_retries
        self._tool_call_budget = tool_call_budget
        self._max_steps = max_steps

        self._pending_findings: list[SanitizerFinding] = []
        self._pending_block_message: str | None = None
        self._finalize_block_counts: dict[str, int] = {}
        self._hypothesis_changed = False
        self._finalize_checked_this_round = False
        self._seen_finding_keys: set[tuple[str, str]] = set()
        self._dispatch_meta: dict[str, _DispatchMeta] = {}
        self._current_round = 0
        self._completed_round = 0
        self._last_processed_round = 0
        self._tool_call_count = 0
        self._pending_finalize_check = False

    def on_turn_start(self, event: TurnStartEvent) -> None:
        self._current_round = event.turn_index + 1

    def on_turn_end(self, event: TurnEndEvent) -> None:
        self._completed_round = event.turn_index + 1
        self._pending_finalize_check = _assistant_contains_finalize(event.message)

    def on_tool_call(self, event: ToolCallEvent) -> None:
        tool_name = event.tool_name
        tool_args = event.args
        current_round = max(self._current_round, 1)

        if tool_name == "dispatch_agent":
            task_type = str(tool_args.get("task_type", ""))
            hypothesis_id = _extract_hypothesis_id(str(tool_args.get("task", "")))
            self._dispatch_meta[event.tool_call_id] = _DispatchMeta(
                task_type=task_type,
                hypothesis_id=hypothesis_id,
            )
            self._tracker.record(
                round=current_round,
                event_type="dispatch",
                data={
                    "agent_id": str(tool_args.get("agent_id", "")),
                    "task_type": task_type,
                    "hypothesis_id": hypothesis_id,
                    "target_services": _extract_services(tool_args),
                    "instruction": str(tool_args.get("task", "")),
                },
            )
        elif tool_name == "update_hypothesis":
            self._tracker.record(
                round=current_round,
                event_type="hypothesis_change",
                data={
                    "hypothesis_id": str(tool_args.get("id", "")),
                    "new_status": str(tool_args.get("status", "")),
                },
            )
            self._hypothesis_changed = True
        elif tool_name == "remove_hypothesis":
            self._tracker.record(
                round=current_round,
                event_type="hypothesis_change",
                data={
                    "hypothesis_id": str(tool_args.get("id", "")),
                    "new_status": "removed",
                },
            )
            self._hypothesis_changed = True
        elif tool_name == "update_service_profile":
            self._tracker.record(
                round=current_round,
                event_type="tool_call",
                data={
                    "tool_name": "update_service_profile",
                    "service_name": str(tool_args.get("service_name", "")),
                },
            )
        elif tool_name == "query_service_profile":
            self._tracker.record(
                round=current_round,
                event_type="tool_call",
                data={
                    "tool_name": "query_service_profile",
                    "service_name": _extract_profile_query_service_name(tool_args),
                },
            )

    def on_tool_result(self, event: ToolResultEvent) -> ToolResult | None:
        self._tool_call_count += 1
        current_round = max(self._current_round, 1)
        result_text = _tool_result_text(event.result)

        if event.tool_name == "dispatch_agent":
            meta = self._dispatch_meta.pop(event.tool_call_id, _DispatchMeta("", ""))
            _try_record_completion(
                result_text,
                current_round,
                self._tracker,
                task_type=meta.task_type,
                hypothesis_id=meta.hypothesis_id,
            )
            if self._critic_sanitizer is not None:
                ctx = self._build_ctx(current_round)
                _schedule_async_check(
                    self._critic_sanitizer,
                    self._hypothesis_store,
                    self._profile_store,
                    self._tracker,
                    ctx,
                )
        elif event.tool_name == "check_tasks":
            _try_record_completion(result_text, current_round, self._tracker)

        return None

    async def on_context(self, event: Any) -> None:
        await self._process_completed_round_if_ready()
        self._drain_async_critic()
        injected = self._build_injection_message()
        if injected is None:
            return
        event.messages.append(injected)

    def on_session_shutdown(self, event: SessionShutdownEvent) -> None:
        del event
        if self._critic_sanitizer is not None:
            self._critic_sanitizer.cancel_pending()

    async def _process_completed_round_if_ready(self) -> None:
        if self._completed_round <= self._last_processed_round:
            return

        round_number = self._completed_round
        self._last_processed_round = round_number
        ctx = self._build_ctx(round_number)
        self._finalize_checked_this_round = False

        if self._pending_finalize_check:
            self._finalize_checked_this_round = True
            code_findings = self._code_sanitizer.check(
                "pre_finalize",
                self._hypothesis_store,
                self._profile_store,
                self._tracker,
                ctx,
            )
            if self._critic_sanitizer is not None:
                code_findings.extend(
                    await self._critic_sanitizer.check(
                        "pre_finalize",
                        self._hypothesis_store,
                        self._profile_store,
                        self._tracker,
                        ctx,
                    )
                )
            code_findings = self._apply_budget_degradation(code_findings, ctx)
            code_findings = self._apply_retry_degradation(code_findings)
            block_findings = [f for f in code_findings if f.severity == Severity.BLOCK]
            if block_findings:
                self._pending_block_message = self._format_finalize_blocked(block_findings)
                for finding in code_findings:
                    if finding.severity != Severity.BLOCK:
                        self._add_finding(finding)
            else:
                self._finalize_block_counts.clear()
                for finding in code_findings:
                    self._add_finding(finding)

        if not self._finalize_checked_this_round:
            every_round_findings = self._code_sanitizer.check(
                "every_round",
                self._hypothesis_store,
                self._profile_store,
                self._tracker,
                ctx,
            )
            for finding in every_round_findings:
                self._add_finding(finding)

            if self._hypothesis_changed:
                hyp_findings = self._code_sanitizer.check(
                    "hypothesis_change",
                    self._hypothesis_store,
                    self._profile_store,
                    self._tracker,
                    ctx,
                )
                hyp_findings = [finding for finding in hyp_findings if finding.code != "C1"]
                for finding in hyp_findings:
                    self._add_finding(finding)
                self._hypothesis_changed = False

            if self._periodic_interval > 0 and round_number % self._periodic_interval == 0:
                periodic_findings = self._code_sanitizer.check(
                    "periodic",
                    self._hypothesis_store,
                    self._profile_store,
                    self._tracker,
                    ctx,
                )
                for finding in periodic_findings:
                    self._add_finding(finding)
        else:
            self._hypothesis_changed = False

        self._pending_finalize_check = False

    def _build_ctx(self, round_number: int) -> SanitizerContext:
        return SanitizerContext(
            agent_id="orchestrator",
            step=max(round_number - 1, 0),
            max_steps=self._max_steps,
            tool_call_count=self._tool_call_count,
            metadata={},
        )

    def _drain_async_critic(self) -> None:
        if self._critic_sanitizer is None:
            return
        async_findings = self._critic_sanitizer.collect_async_results()
        for finding in async_findings:
            self._add_finding(finding)

    def _build_injection_message(self) -> UserMessage | None:
        parts: list[str] = []
        if self._pending_block_message is not None:
            parts.append(self._pending_block_message)
            self._pending_block_message = None
        if self._pending_findings:
            parts.append(self._format_findings(self._pending_findings))
            self._pending_findings = []
        self._seen_finding_keys.clear()
        if not parts:
            return None
        return UserMessage(
            role="user",
            content=[TextContent(type="text", text="\n".join(parts))],
            timestamp=time.time(),
        )

    def _add_finding(self, finding: SanitizerFinding) -> None:
        key = (finding.code, _extract_target_key(finding))
        if key not in self._seen_finding_keys:
            self._seen_finding_keys.add(key)
            self._pending_findings.append(finding)

    def _format_findings(self, findings: list[SanitizerFinding]) -> str:
        lines = [
            f'<finding code="{finding.code}" severity="{finding.severity.value}">\n'
            f"{finding.message}\n"
            f"</finding>"
            for finding in findings
        ]
        return "<sanitizer_report>\n" + "\n".join(lines) + "\n</sanitizer_report>"

    def _format_finalize_blocked(self, block_findings: list[SanitizerFinding]) -> str:
        items = "\n".join(
            f"{index + 1}. [{finding.code}] {finding.message}"
            for index, finding in enumerate(block_findings)
        )
        return (
            f'<finalize_blocked reason="{len(block_findings)} BLOCK findings">\n'
            "You attempted to finalize but the following conditions are not met:\n"
            f"{items}\n"
            "Address these before attempting to finalize again.\n"
            "</finalize_blocked>"
        )

    def _apply_budget_degradation(
        self,
        findings: list[SanitizerFinding],
        ctx: SanitizerContext,
    ) -> list[SanitizerFinding]:
        if self._tool_call_budget is None:
            return findings
        if ctx.tool_call_count < self._tool_call_budget:
            return findings

        result: list[SanitizerFinding] = []
        for finding in findings:
            if finding.severity == Severity.BLOCK and finding.code[0] in ("E", "J"):
                result.append(
                    SanitizerFinding(
                        code=finding.code,
                        severity=Severity.WARN,
                        message=f"{finding.message} [budget_exhausted]",
                        details=finding.details,
                    )
                )
            else:
                result.append(finding)
        return result

    def _apply_retry_degradation(
        self,
        findings: list[SanitizerFinding],
    ) -> list[SanitizerFinding]:
        result: list[SanitizerFinding] = []
        for finding in findings:
            if finding.severity != Severity.BLOCK:
                result.append(finding)
                continue

            code = finding.code
            count = self._finalize_block_counts.get(code, 0)
            if count >= self._max_block_retries:
                result.append(
                    SanitizerFinding(
                        code=finding.code,
                        severity=Severity.WARN,
                        message=(
                            f"{finding.message} "
                            f"[DEGRADED: unresolved after {count} attempts]"
                        ),
                        details=finding.details,
                    )
                )
            else:
                self._finalize_block_counts[code] = count + 1
                result.append(finding)
        return result


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    hypothesis_store = _expect_hypothesis_store(config)
    profile_store = _expect_profile_store(config)
    code_sanitizer = CodeSanitizer(
        severity_map=_string_map(config.get("severity_map")),
        disabled=_string_set(config.get("disabled")),
        drift_window=int(config.get("drift_window", 3)),
    )

    critic_model = config.get("critic_model")
    critic_sanitizer = (
        CriticSanitizer(
            critic_model,
            severity_map=_string_map(config.get("critic_severity_map")),
            disabled=_string_set(config.get("critic_disabled")),
        )
        if critic_model is not None
        else None
    )

    extension = _SanitizerExtension(
        hypothesis_store=hypothesis_store,
        profile_store=profile_store,
        code_sanitizer=code_sanitizer,
        critic_sanitizer=critic_sanitizer,
        periodic_interval=int(config.get("periodic_interval", 5)),
        max_block_retries=int(config.get("max_block_retries", 3)),
        tool_call_budget=_maybe_int(config.get("tool_call_budget")),
        max_steps=_maybe_int(config.get("max_steps")),
    )

    api.on("turn_start", extension.on_turn_start)
    api.on("turn_end", extension.on_turn_end)
    api.on("tool_call", extension.on_tool_call)
    api.on("tool_result", extension.on_tool_result)
    api.on("context", extension.on_context)
    api.on("session_shutdown", extension.on_session_shutdown)


def _expect_hypothesis_store(config: dict[str, Any]) -> HypothesisStore:
    store = config.get("hypothesis_store")
    if not isinstance(store, HypothesisStore):
        raise TypeError("sanitizer.install requires config['hypothesis_store']=HypothesisStore")
    return store


def _expect_profile_store(config: dict[str, Any]) -> ServiceProfileStore:
    store = config.get("profile_store")
    if not isinstance(store, ServiceProfileStore):
        raise TypeError("sanitizer.install requires config['profile_store']=ServiceProfileStore")
    return store


def _string_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if not isinstance(value, (list, set, tuple)):
        raise TypeError(f"Expected iterable[str], got {type(value).__name__}")
    return {str(item) for item in value}


def _string_map(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict[str, Any], got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _assistant_contains_finalize(message: AssistantMessage) -> bool:
    return _FINALIZE_RE.search(_assistant_text(message)) is not None


def _assistant_text(message: AssistantMessage) -> str:
    blocks: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            blocks.append(block.text)
    return "".join(blocks).strip()


def _tool_result_text(result: ToolResult) -> str:
    return "".join(
        block.text for block in result.content if isinstance(block, TextContent)
    )


def _extract_services(tool_args: dict[str, Any]) -> list[str]:
    services = tool_args.get("target_services")
    if isinstance(services, list):
        return [str(service) for service in services]

    task_text = str(tool_args.get("task", ""))
    if not task_text:
        return []
    return re.findall(r"`([^`]+)`", task_text)


def _extract_profile_query_service_name(tool_args: dict[str, Any]) -> str:
    service_names = tool_args.get("service_names")
    if isinstance(service_names, str):
        first = service_names.split(",", 1)[0].strip()
        if first:
            return first
    return ""


def _extract_hypothesis_id(text: str) -> str:
    match = re.search(r"\bH(\d+)\b", text)
    return match.group(0) if match else ""


def _try_record_completion(
    result: str,
    current_round: int,
    tracker: InvestigationTracker,
    *,
    task_type: str = "",
    hypothesis_id: str = "",
) -> None:
    for verdict in ("SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"):
        if re.search(r"\b" + verdict + r"\b", result):
            tracker.record(
                round=current_round,
                event_type="task_complete",
                data={
                    "verdict": verdict,
                    "task_type": task_type,
                    "hypothesis_id": hypothesis_id,
                },
            )
            break


def _extract_target_key(finding: SanitizerFinding) -> str:
    details = finding.details
    for key in (
        "hypothesis_id",
        "service",
        "service_name",
        "missing_service",
        "upstream_service",
    ):
        value = details.get(key)
        if value:
            return str(value)
    return finding.message


def _schedule_async_check(
    critic_sanitizer: CriticSanitizer,
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: SanitizerContext,
) -> None:
    import asyncio

    async def _run() -> None:
        await critic_sanitizer.check_async(
            "dispatch",
            hypothesis_store,
            profile_store,
            tracker,
            ctx,
        )

    asyncio.create_task(_run())
