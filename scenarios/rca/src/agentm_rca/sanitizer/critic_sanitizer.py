"""CriticSanitizer — LLM-based semantic checks for RCA investigations.

Uses a cheap model to perform focused checks on investigation quality:
- C3: Causal direction verification
- J1: Signal misread detection
- J4: Symptom-as-cause detection
- P2: Dispatch quality assessment
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel

from agentm_rca.stores import HypothesisEntry, HypothesisStore
from agentm_rca.sanitizer.models import SanitizerContext, SanitizerFinding, Severity
from agentm_rca.sanitizer.tracker import InvestigationTracker
from agentm_rca.stores import ServiceProfile, ServiceProfileStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------


class CausalDirectionResult(BaseModel):
    """LLM response for causal direction check (C3)."""

    proven: bool
    missing_evidence: str = ""


class SignalMisreadResult(BaseModel):
    """LLM response for signal misread check (J1)."""

    misread: bool
    reasoning: str = ""


class SymptomAsCauseResult(BaseModel):
    """LLM response for symptom-as-cause check (J4)."""

    likely_symptom: bool
    suspect_upstream: list[str] = []


class DispatchQualityResult(BaseModel):
    """LLM response for dispatch quality check (P2)."""

    quality: Literal["good", "incomplete", "poor"]
    missing: list[str] = []


# ---------------------------------------------------------------------------
# Default severity map
# ---------------------------------------------------------------------------

_DEFAULT_SEVERITY: dict[str, Severity] = {
    "C3": Severity.BLOCK,
    "J1": Severity.WARN,
    "J4": Severity.WARN,
    "P2": Severity.INFO,
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_causal_direction_prompt(
    hypothesis: HypothesisEntry,
    deep_analyze_findings: list[str],
) -> str:
    """Build prompt for C3: causal direction verification."""
    findings_text = "\n".join(f"- {f}" for f in deep_analyze_findings) or "(none)"
    return (
        "You are reviewing whether a root-cause hypothesis has been proven "
        "with sufficient causal direction evidence.\n\n"
        f"Confirmed hypothesis: {hypothesis.description}\n"
        f"Evidence collected: {'; '.join(hypothesis.evidence)}\n\n"
        f"Deep-analyze task findings:\n{findings_text}\n\n"
        "Does the evidence include:\n"
        "1. Internal-time attribution (proving the suspect service's "
        "latency/errors started BEFORE downstream impact)\n"
        "2. Independent-anomaly analysis (ruling out that the suspect "
        "service was merely reacting to another upstream failure)\n\n"
        "Respond with proven=true only if BOTH criteria are met."
    )


def _build_signal_misread_prompt(
    hypothesis: HypothesisEntry,
    evidence: tuple[str, ...],
    rejection_event: dict[str, Any],
) -> str:
    """Build prompt for J1: signal misread detection."""
    evidence_text = "\n".join(f"- {e}" for e in evidence)
    return (
        "You are reviewing whether a hypothesis was incorrectly rejected "
        "despite having strong supporting evidence.\n\n"
        f"Hypothesis: {hypothesis.description}\n"
        f"Status at rejection: {hypothesis.status}\n"
        f"Evidence collected ({len(evidence)} items):\n{evidence_text}\n\n"
        f"Rejection event data: {rejection_event}\n\n"
        "Compare the strength of the supporting evidence against the "
        "rejection reason. Was the evidence misread or dismissed too "
        "quickly?\n\n"
        "Respond with misread=true if the evidence appears stronger than "
        "the rejection justification."
    )


def _build_symptom_as_cause_prompt(
    service_name: str,
    service_profile: ServiceProfile,
    upstream_profiles: list[ServiceProfile],
) -> str:
    """Build prompt for J4: symptom-as-cause detection."""
    upstream_text = "\n".join(
        f"- {p.service_name}: anomaly={p.is_anomalous}, "
        f"summary={p.anomaly_summary or 'N/A'}"
        for p in upstream_profiles
    )
    return (
        "You are checking whether the confirmed root-cause service might "
        "actually be a victim (symptom) of an upstream failure.\n\n"
        f"Confirmed root-cause service: {service_name}\n"
        f"Service anomaly summary: {service_profile.anomaly_summary or 'N/A'}\n\n"
        f"Upstream services with anomalies:\n{upstream_text}\n\n"
        "If any upstream anomaly could explain the confirmed service's "
        "issues, the confirmed service might be a symptom, not the cause.\n\n"
        "Respond with likely_symptom=true if upstream anomalies suggest "
        "the confirmed service is a victim. List the suspect upstream "
        "service names in suspect_upstream."
    )


def _build_dispatch_quality_prompt(task_instruction: str) -> str:
    """Build prompt for P2: dispatch quality assessment."""
    return (
        "You are evaluating the quality of a task dispatch instruction "
        "sent to an investigation agent.\n\n"
        f"Task instruction:\n{task_instruction}\n\n"
        "A good dispatch should include:\n"
        "1. Target service names to investigate\n"
        "2. Specific metric values or thresholds to look for\n"
        "3. The hypothesis being tested\n"
        "4. Forward predictions (what we expect to find if the "
        "hypothesis is correct)\n\n"
        "Rate quality as 'good' (all 4), 'incomplete' (2-3), or "
        "'poor' (0-1). List what is missing."
    )


# ---------------------------------------------------------------------------
# CriticSanitizer
# ---------------------------------------------------------------------------


class CriticSanitizer:
    """LLM-based semantic sanitizer for RCA investigations.

    Runs focused prompts through a cheap model to detect investigation
    quality issues that rule-based checks cannot catch.
    """

    def __init__(
        self,
        model: Any,  # ModelProtocol
        severity_map: dict[str, Severity] | None = None,
        disabled: set[str] | None = None,
    ) -> None:
        self._model = model
        self._severity: dict[str, Severity] = {
            **_DEFAULT_SEVERITY,
            **(severity_map or {}),
        }
        self._disabled: set[str] = disabled or set()
        self._async_tasks: list[asyncio.Task[list[SanitizerFinding]]] = []

    def _get_severity(self, code: str) -> Severity:
        """Look up severity for a check code, falling back to WARN."""
        return self._severity.get(code, Severity.WARN)

    def _is_enabled(self, code: str) -> bool:
        """Return True if the check is not disabled."""
        return code not in self._disabled

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def check(
        self,
        trigger: str,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        tracker: InvestigationTracker,
        ctx: SanitizerContext,
    ) -> list[SanitizerFinding]:
        """Run LLM checks for pre_finalize trigger concurrently.

        Returns findings from C3, J1, and J4 checks.
        """
        if trigger != "pre_finalize":
            return []

        # Run all three checks in parallel — they are independent
        results = await asyncio.gather(
            self._check_causal_direction(hypothesis_store, profile_store, tracker),
            self._check_signal_misread(hypothesis_store, profile_store, tracker),
            self._check_symptom_as_cause(hypothesis_store, profile_store, tracker),
            return_exceptions=True,
        )

        findings: list[SanitizerFinding] = []
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("Critic check failed during pre_finalize", exc_info=r)
            else:
                findings.extend(r)

        return findings

    async def check_async(
        self,
        trigger: str,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        tracker: InvestigationTracker,
        ctx: SanitizerContext,
    ) -> None:
        """Fire-and-forget async checks for dispatch trigger.

        Stores asyncio.Tasks internally; call collect_async_results()
        to drain completed results.
        """
        if trigger != "dispatch":
            return

        if not self._is_enabled("P2"):
            return

        # Get latest dispatch event
        dispatches = tracker.dispatches()
        if not dispatches:
            return

        latest = dispatches[-1]
        instruction = latest.data.get("instruction", "")
        if not instruction:
            return

        task = asyncio.create_task(self._check_dispatch_quality(str(instruction)))
        self._async_tasks.append(task)

    def collect_async_results(self) -> list[SanitizerFinding]:
        """Drain completed async tasks and return their findings."""
        findings: list[SanitizerFinding] = []
        pending: list[asyncio.Task[list[SanitizerFinding]]] = []

        for task in self._async_tasks:
            if task.done():
                try:
                    findings.extend(task.result())
                except Exception:
                    logger.warning("Async sanitizer task failed", exc_info=True)
            else:
                pending.append(task)

        self._async_tasks = pending
        return findings

    def cancel_pending(self) -> None:
        """Cancel all pending async tasks. Call when the investigation ends."""
        for task in self._async_tasks:
            if not task.done():
                task.cancel()
        self._async_tasks = []

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    async def _check_causal_direction(
        self,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        tracker: InvestigationTracker,
    ) -> list[SanitizerFinding]:
        """C3: Verify causal direction of confirmed hypothesis."""
        if not self._is_enabled("C3"):
            return []

        confirmed_id = hypothesis_store.confirmed_id
        if confirmed_id is None:
            return []

        hypothesis = hypothesis_store.get(confirmed_id)
        if hypothesis is None:
            return []

        # Gather deep_analyze task completion summaries
        completions = tracker.task_completions()
        deep_findings = [
            str(c.data.get("summary", ""))
            for c in completions
            if c.data.get("task_type") == "deep_analyze"
        ]

        prompt = _build_causal_direction_prompt(hypothesis, deep_findings)

        try:
            structured = self._model.with_structured_output(
                CausalDirectionResult, method="function_calling"
            )
            result: CausalDirectionResult = await structured.ainvoke(
                [{"role": "human", "content": prompt}]
            )
        except Exception:
            logger.warning("C3 causal direction check failed", exc_info=True)
            return []

        if not result.proven:
            return [
                SanitizerFinding(
                    code="C3",
                    severity=self._get_severity("C3"),
                    message=f"Causal direction not proven: {result.missing_evidence}",
                )
            ]
        return []

    async def _check_signal_misread(
        self,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        tracker: InvestigationTracker,
    ) -> list[SanitizerFinding]:
        """J1: Detect potential signal misreads in rejected hypotheses."""
        if not self._is_enabled("J1"):
            return []

        all_hypotheses = hypothesis_store.get_all()
        # Find rejected hypotheses with >= 3 evidence items
        rejected_with_evidence = [
            h for h in all_hypotheses.values()
            if h.status == "rejected" and len(h.evidence) >= 3
        ]

        if not rejected_with_evidence:
            return []

        # Find hypothesis_change events for rejections
        changes = tracker.hypothesis_changes()

        findings: list[SanitizerFinding] = []
        for hypothesis in rejected_with_evidence:
            # Find the rejection event for this hypothesis
            rejection_event: dict[str, Any] = {}
            for change in changes:
                if (
                    change.data.get("hypothesis_id") == hypothesis.id
                    and change.data.get("new_status") == "rejected"
                ):
                    rejection_event = change.data
                    break

            prompt = _build_signal_misread_prompt(
                hypothesis, hypothesis.evidence, rejection_event
            )

            try:
                structured = self._model.with_structured_output(
                    SignalMisreadResult, method="function_calling"
                )
                result: SignalMisreadResult = await structured.ainvoke(
                    [{"role": "human", "content": prompt}]
                )
            except Exception:
                logger.warning(
                    "J1 signal misread check failed for %s",
                    hypothesis.id,
                    exc_info=True,
                )
                continue

            if result.misread:
                findings.append(
                    SanitizerFinding(
                        code="J1",
                        severity=self._get_severity("J1"),
                        message=(
                            f"Potential signal misread for hypothesis "
                            f"'{hypothesis.id}': {result.reasoning}"
                        ),
                    )
                )

        return findings

    async def _check_symptom_as_cause(
        self,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        tracker: InvestigationTracker,
    ) -> list[SanitizerFinding]:
        """J4: Check if confirmed cause is actually a symptom."""
        if not self._is_enabled("J4"):
            return []

        confirmed_id = hypothesis_store.confirmed_id
        if confirmed_id is None:
            return []

        hypothesis = hypothesis_store.get(confirmed_id)
        if hypothesis is None:
            return []

        # Extract first backtick-quoted service name from description
        match = re.search(r"`([^`]+)`", hypothesis.description)
        if match is None:
            return []

        service_name = match.group(1)
        service_profile = profile_store.get(service_name)
        if service_profile is None:
            return []

        # Get upstream services and check for anomalous ones
        if not service_profile.upstream_services:
            return []

        upstream_profiles: list[ServiceProfile] = []
        for upstream_name in service_profile.upstream_services:
            up_profile = profile_store.get(upstream_name)
            if up_profile is not None and up_profile.is_anomalous:
                upstream_profiles.append(up_profile)

        if not upstream_profiles:
            return []

        prompt = _build_symptom_as_cause_prompt(
            service_name, service_profile, upstream_profiles
        )

        try:
            structured = self._model.with_structured_output(
                SymptomAsCauseResult, method="function_calling"
            )
            result: SymptomAsCauseResult = await structured.ainvoke(
                [{"role": "human", "content": prompt}]
            )
        except Exception:
            logger.warning("J4 symptom-as-cause check failed", exc_info=True)
            return []

        if result.likely_symptom:
            suspects = ", ".join(result.suspect_upstream) if result.suspect_upstream else "unknown"
            return [
                SanitizerFinding(
                    code="J4",
                    severity=self._get_severity("J4"),
                    message=(
                        f"Confirmed service '{service_name}' may be a "
                        f"symptom, not the cause. Suspect upstream: {suspects}"
                    ),
                )
            ]
        return []

    async def _check_dispatch_quality(
        self,
        task_instruction: str,
    ) -> list[SanitizerFinding]:
        """P2: Assess dispatch instruction quality."""
        prompt = _build_dispatch_quality_prompt(task_instruction)

        try:
            structured = self._model.with_structured_output(
                DispatchQualityResult, method="function_calling"
            )
            result: DispatchQualityResult = await structured.ainvoke(
                [{"role": "human", "content": prompt}]
            )
        except Exception:
            logger.warning("P2 dispatch quality check failed", exc_info=True)
            return []

        if result.quality != "good":
            missing_text = ", ".join(result.missing) if result.missing else "unspecified"
            return [
                SanitizerFinding(
                    code="P2",
                    severity=self._get_severity("P2"),
                    message=(
                        f"Dispatch quality '{result.quality}': "
                        f"missing {missing_text}"
                    ),
                )
            ]
        return []
