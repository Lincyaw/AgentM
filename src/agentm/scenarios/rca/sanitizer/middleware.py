"""SanitizerMiddleware — wires CodeSanitizer and CriticSanitizer into the orchestrator loop.

Observes tool calls to track investigation events, gates finalize decisions
on sanitizer findings, and injects feedback messages into the LLM prompt.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from agentm.harness.middleware import MiddlewareBase
from agentm.harness.scenario import TrajectorySlot
from agentm.harness.types import LoopContext, Message
from agentm.scenarios.rca.hypothesis_store import HypothesisStore
from agentm.scenarios.rca.sanitizer.code_sanitizer import CodeSanitizer
from agentm.scenarios.rca.sanitizer.critic_sanitizer import CriticSanitizer
from agentm.scenarios.rca.sanitizer.models import SanitizerFinding, Severity
from agentm.scenarios.rca.sanitizer.tracker import InvestigationTracker
from agentm.scenarios.rca.service_profile import ServiceProfileStore

logger = logging.getLogger(__name__)

_FINALIZE_RE = re.compile(r"<decision>\s*finalize\s*</decision>", re.IGNORECASE)


class SanitizerMiddleware(MiddlewareBase):
    """Middleware that runs CodeSanitizer and CriticSanitizer checks
    during the orchestrator loop, gating finalize and injecting feedback.
    """

    def __init__(
        self,
        code_sanitizer: CodeSanitizer,
        critic_sanitizer: CriticSanitizer | None,
        tracker: InvestigationTracker,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        traj_slot: TrajectorySlot | None = None,
        periodic_interval: int = 5,
        max_block_retries: int = 3,
        tool_call_budget: int | None = None,
    ) -> None:
        self._code_sanitizer = code_sanitizer
        self._critic_sanitizer = critic_sanitizer
        self._tracker = tracker
        self._hypothesis_store = hypothesis_store
        self._profile_store = profile_store
        self._traj_slot = traj_slot
        self._periodic_interval = periodic_interval
        self._max_block_retries = max_block_retries
        self._tool_call_budget = tool_call_budget

        # Internal state
        self._pending_findings: list[SanitizerFinding] = []
        self._pending_block_message: str | None = None
        self._block_attempts: dict[tuple[str, str], int] = {}
        self._hypothesis_changed: bool = False

    # ------------------------------------------------------------------
    # on_tool_call — observe and record events
    # ------------------------------------------------------------------

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Callable[[str, dict[str, Any]], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        current_round = ctx.step

        if tool_name == "dispatch_agent":
            task_type = tool_args.get("task_type", "")
            # Extract hypothesis_id from task text (best effort: look for H-prefixed IDs)
            hypothesis_id = _extract_hypothesis_id(tool_args.get("task", ""))
            self._tracker.record(
                round=current_round,
                event_type="dispatch",
                data={
                    "agent_id": tool_args.get("agent_id", ""),
                    "task_type": task_type,
                    "hypothesis_id": hypothesis_id,
                    "target_services": _extract_services(tool_args),
                    "instruction": tool_args.get("task", ""),
                },
            )
            if self._critic_sanitizer is not None:
                await self._critic_sanitizer.check_async(
                    "dispatch",
                    self._hypothesis_store,
                    self._profile_store,
                    self._tracker,
                    ctx,
                )

        elif tool_name == "update_hypothesis":
            self._tracker.record(
                round=current_round,
                event_type="hypothesis_change",
                data={
                    "hypothesis_id": tool_args.get("id", ""),
                    "new_status": tool_args.get("status", ""),
                },
            )
            self._hypothesis_changed = True

        elif tool_name == "remove_hypothesis":
            self._tracker.record(
                round=current_round,
                event_type="hypothesis_change",
                data={
                    "hypothesis_id": tool_args.get("id", ""),
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
                    "service_name": tool_args.get("service_name", ""),
                },
            )

        elif tool_name == "query_service_profile":
            self._tracker.record(
                round=current_round,
                event_type="tool_call",
                data={
                    "tool_name": "query_service_profile",
                    "service_name": tool_args.get("service_name", ""),
                },
            )

        # Always execute the actual tool
        result = await call_next(tool_name, tool_args)

        # Record task completion with dispatch context from tool_args
        if tool_name == "dispatch_agent":
            _try_record_completion(
                result, current_round, self._tracker,
                task_type=tool_args.get("task_type", ""),
                hypothesis_id=_extract_hypothesis_id(tool_args.get("task", "")),
            )
        elif tool_name == "check_tasks":
            _try_record_completion(result, current_round, self._tracker)

        return result

    # ------------------------------------------------------------------
    # on_llm_end — run checks and gate finalize
    # ------------------------------------------------------------------

    async def on_llm_end(self, response: object, ctx: LoopContext) -> object:
        current_round = ctx.step + 1  # step is 0-based, rounds are 1-based

        content = getattr(response, "content", "") or ""
        content_modified = False

        # Check for finalize decision
        if _FINALIZE_RE.search(content):
            # Run pre_finalize checks
            code_findings = self._code_sanitizer.check(
                "pre_finalize",
                self._hypothesis_store,
                self._profile_store,
                self._tracker,
                ctx,
            )

            if self._critic_sanitizer is not None:
                critic_findings = await self._critic_sanitizer.check(
                    "pre_finalize",
                    self._hypothesis_store,
                    self._profile_store,
                    self._tracker,
                    ctx,
                )
                code_findings.extend(critic_findings)

            # Apply budget-aware degradation
            code_findings = self._apply_budget_degradation(code_findings, ctx)

            # Apply retry degradation
            code_findings = self._apply_retry_degradation(code_findings)

            # Record to trajectory
            self._record_findings("pre_finalize", code_findings)

            # Check for remaining BLOCKs
            block_findings = [f for f in code_findings if f.severity == Severity.BLOCK]
            if block_findings:
                # Strip finalize tag
                content = _FINALIZE_RE.sub("", content)
                content_modified = True
                # Store block message separately from findings
                self._pending_block_message = self._format_finalize_blocked(
                    block_findings
                )
                # Add non-block findings as regular findings
                self._pending_findings.extend(
                    f for f in code_findings if f.severity != Severity.BLOCK
                )
            else:
                # Finalize allowed — reset block attempts for previously tracked codes
                self._block_attempts.clear()
                # Still report non-block findings
                self._pending_findings.extend(code_findings)

        # Every round checks
        every_round_findings = self._code_sanitizer.check(
            "every_round",
            self._hypothesis_store,
            self._profile_store,
            self._tracker,
            ctx,
        )
        if every_round_findings:
            self._pending_findings.extend(every_round_findings)
            self._record_findings("every_round", every_round_findings)

        # Hypothesis change checks
        if self._hypothesis_changed:
            hyp_findings = self._code_sanitizer.check(
                "hypothesis_change",
                self._hypothesis_store,
                self._profile_store,
                self._tracker,
                ctx,
            )
            if hyp_findings:
                self._pending_findings.extend(hyp_findings)
                self._record_findings("hypothesis_change", hyp_findings)
            self._hypothesis_changed = False

        # Periodic checks
        if (
            self._periodic_interval > 0
            and current_round % self._periodic_interval == 0
        ):
            periodic_findings = self._code_sanitizer.check(
                "periodic",
                self._hypothesis_store,
                self._profile_store,
                self._tracker,
                ctx,
            )
            if periodic_findings:
                self._pending_findings.extend(periodic_findings)
                self._record_findings("periodic", periodic_findings)

        # Apply content modification if needed
        if content_modified:
            response.content = content  # type: ignore[attr-defined]

        return response

    # ------------------------------------------------------------------
    # on_llm_start — inject pending findings
    # ------------------------------------------------------------------

    async def on_llm_start(
        self, messages: list[Message], ctx: LoopContext
    ) -> list[Message]:
        # Collect async critic results
        if self._critic_sanitizer is not None:
            async_findings = self._critic_sanitizer.collect_async_results()
            if async_findings:
                self._pending_findings.extend(async_findings)
                self._record_findings("async_critic", async_findings)

        # Build injection message from block message + regular findings
        parts: list[str] = []
        if self._pending_block_message is not None:
            parts.append(self._pending_block_message)
            self._pending_block_message = None
        if self._pending_findings:
            parts.append(self._format_findings(self._pending_findings))
            self._pending_findings = []

        if parts:
            return [*messages, {"role": "human", "content": "\n".join(parts)}]

        return messages

    # ------------------------------------------------------------------
    # Finding formatting
    # ------------------------------------------------------------------

    def _format_findings(self, findings: list[SanitizerFinding]) -> str:
        """Format findings as XML sanitizer_report for injection."""
        lines = [
            f'<finding code="{f.code}" severity="{f.severity.value}">\n'
            f"{f.message}\n"
            f"</finding>"
            for f in findings
        ]
        return (
            "<sanitizer_report>\n"
            + "\n".join(lines)
            + "\n</sanitizer_report>"
        )

    def _format_finalize_blocked(
        self, block_findings: list[SanitizerFinding]
    ) -> str:
        """Format the finalize_blocked XML message."""
        items = "\n".join(
            f"{i + 1}. [{f.code}] {f.message}"
            for i, f in enumerate(block_findings)
        )
        return (
            f'<finalize_blocked reason="{len(block_findings)} BLOCK findings">\n'
            f"You attempted to finalize but the following conditions are not met:\n"
            f"{items}\n"
            f"Address these before attempting to finalize again.\n"
            f"</finalize_blocked>"
        )

    # ------------------------------------------------------------------
    # Degradation helpers
    # ------------------------------------------------------------------

    def _apply_budget_degradation(
        self,
        findings: list[SanitizerFinding],
        ctx: LoopContext,
    ) -> list[SanitizerFinding]:
        """Degrade E-codes and J-codes from BLOCK to WARN when budget exhausted."""
        if self._tool_call_budget is None:
            return findings
        if ctx.tool_call_count < self._tool_call_budget:
            return findings

        result: list[SanitizerFinding] = []
        for f in findings:
            if f.severity == Severity.BLOCK and f.code[0] in ("E", "J"):
                result.append(
                    SanitizerFinding(
                        code=f.code,
                        severity=Severity.WARN,
                        message=f"{f.message} [budget_exhausted]",
                        details=f.details,
                    )
                )
            else:
                result.append(f)
        return result

    def _apply_retry_degradation(
        self,
        findings: list[SanitizerFinding],
    ) -> list[SanitizerFinding]:
        """Degrade BLOCK findings after max_block_retries consecutive blocks."""
        result: list[SanitizerFinding] = []
        for f in findings:
            if f.severity != Severity.BLOCK:
                result.append(f)
                continue

            target_key = _extract_target_key(f)
            key = (f.code, target_key)
            count = self._block_attempts.get(key, 0)

            if count >= self._max_block_retries:
                result.append(
                    SanitizerFinding(
                        code=f.code,
                        severity=Severity.WARN,
                        message=(
                            f"{f.message} "
                            f"[DEGRADED: unresolved after {count} attempts]"
                        ),
                        details=f.details,
                    )
                )
            else:
                self._block_attempts[key] = count + 1
                result.append(f)

        return result

    # ------------------------------------------------------------------
    # Trajectory recording
    # ------------------------------------------------------------------

    def _record_findings(
        self,
        trigger: str,
        findings: list[SanitizerFinding],
    ) -> None:
        """Record sanitizer findings to trajectory if available."""
        trajectory = self._traj_slot.value if self._traj_slot is not None else None
        if trajectory is None or not findings:
            return
        trajectory.record_sync(
            event_type="sanitizer",
            agent_path=["orchestrator"],
            data={
                "trigger": trigger,
                "findings": [
                    {
                        "code": f.code,
                        "severity": f.severity.value,
                        "message": f.message,
                    }
                    for f in findings
                ],
                "blocked": any(f.severity == Severity.BLOCK for f in findings),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_services(tool_args: dict[str, Any]) -> list[str]:
    """Best-effort extraction of target service names from dispatch args."""
    # Check for explicit target_services field
    services = tool_args.get("target_services")
    if isinstance(services, list):
        return [str(s) for s in services]

    # Try to parse from task text
    task_text = str(tool_args.get("task", ""))
    if not task_text:
        return []

    # Look for backtick-quoted service names
    return re.findall(r"`([^`]+)`", task_text)


def _extract_hypothesis_id(text: str) -> str:
    """Best-effort extraction of hypothesis ID (e.g. 'H1', 'H12') from text."""
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
    """Record task completion if the result contains a verdict keyword.

    *task_type* and *hypothesis_id* come from the original dispatch_agent
    tool_args so that downstream checks (C1, C2, P1) can match completions
    to specific hypotheses and task types.
    """
    for verdict in ("SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"):
        if verdict in result:
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
    """Extract a target key from finding details for retry tracking."""
    details = finding.details
    # Try hypothesis_id first
    hyp_id = details.get("hypothesis_id")
    if hyp_id:
        return str(hyp_id)
    # Try service name
    svc = details.get("service") or details.get("service_name")
    if svc:
        return str(svc)
    # Try service list
    services = details.get("target_services")
    if isinstance(services, list):
        return ",".join(str(s) for s in sorted(services))
    # Fallback to empty string
    return ""
