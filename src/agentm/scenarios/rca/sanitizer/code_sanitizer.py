"""CodeSanitizer — deterministic code-based checks for RCA investigations.

Implements 11 check functions covering evidence gaps, confirmation hygiene,
investigation judgement, and process adherence. The CodeSanitizer class
routes triggers to the appropriate subset of checks.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from agentm.harness.types import LoopContext
from agentm.scenarios.rca.hypothesis_store import HypothesisStore
from agentm.scenarios.rca.sanitizer.models import SanitizerFinding, Severity
from agentm.scenarios.rca.sanitizer.tracker import InvestigationTracker
from agentm.scenarios.rca.service_profile import ServiceProfileStore

# ---------------------------------------------------------------------------
# Default dimension mapping for E2
# ---------------------------------------------------------------------------

DEFAULT_DIMENSION_MAP: dict[str, str] = {
    "latency": "latency",
    "p99": "latency",
    "avg_duration": "latency",
    "error": "error_rate",
    "status_code": "error_rate",
    "call_volume": "call_volume",
    "call_count": "call_volume",
    "request_count": "call_volume",
    "resource": "resources",
    "cpu": "resources",
    "memory": "resources",
    "gc": "resources",
    "metric": "resources",
}

REQUIRED_DIMENSIONS: frozenset[str] = frozenset(
    {"latency", "error_rate", "call_volume", "resources"}
)

# ---------------------------------------------------------------------------
# Default severity for each check code
# ---------------------------------------------------------------------------

_DEFAULT_SEVERITIES: dict[str, Severity] = {
    "E1": Severity.WARN,
    "E2": Severity.WARN,
    "E3": Severity.WARN,
    "E4": Severity.WARN,
    "C1": Severity.BLOCK,
    "C2": Severity.BLOCK,
    "C4": Severity.WARN,
    "J2": Severity.WARN,
    "J3": Severity.BLOCK,
    "P1": Severity.WARN,
    "P3": Severity.INFO,
}


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def _make_finding(
    code: str,
    message: str,
    severity_map: dict[str, Severity],
    details: dict[str, Any] | None = None,
) -> SanitizerFinding:
    """Helper to create a finding with the correct severity."""
    severity = severity_map.get(code, _DEFAULT_SEVERITIES[code])
    return SanitizerFinding(
        code=code,
        severity=severity,
        message=message,
        details=details or {},
    )


def check_anchoring_bias(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
) -> list[SanitizerFinding]:
    """E1: Check if upstream services of anomalous services lack observations."""
    smap = severity_map or {}
    findings: list[SanitizerFinding] = []
    anomalous = profile_store.query(anomalous_only=True)

    for profile in anomalous:
        for upstream in profile.upstream_services:
            up_profile = profile_store.get(upstream)
            if up_profile is None or not up_profile.observations:
                findings.append(
                    _make_finding(
                        "E1",
                        f"Upstream service '{upstream}' of anomalous "
                        f"'{profile.service_name}' has no observations",
                        smap,
                        details={
                            "anomalous_service": profile.service_name,
                            "upstream_service": upstream,
                        },
                    )
                )
    return findings


def check_dimension_gap(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
    dimension_map: dict[str, str] | None = None,
) -> list[SanitizerFinding]:
    """E2: Check if anomalous services are missing data dimension categories."""
    smap = severity_map or {}
    dmap = dimension_map or DEFAULT_DIMENSION_MAP
    findings: list[SanitizerFinding] = []
    anomalous = profile_store.query(anomalous_only=True)

    for profile in anomalous:
        covered: set[str] = set()
        for ds in profile.data_sources_queried:
            ds_lower = ds.lower()
            for keyword, category in dmap.items():
                if keyword in ds_lower:
                    covered.add(category)
        missing = REQUIRED_DIMENSIONS - covered
        if missing:
            findings.append(
                _make_finding(
                    "E2",
                    f"Service '{profile.service_name}' missing dimension "
                    f"categories: {sorted(missing)}",
                    smap,
                    details={
                        "service": profile.service_name,
                        "missing_dimensions": sorted(missing),
                        "covered_dimensions": sorted(covered),
                    },
                )
            )
    return findings


def check_coverage_gap(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
) -> list[SanitizerFinding]:
    """E3: Check if services mentioned in topology are missing from the store."""
    smap = severity_map or {}
    findings: list[SanitizerFinding] = []
    all_profiles = profile_store.get_all()

    # Collect all services mentioned in any profile's topology
    mentioned: set[str] = set()
    for profile in all_profiles.values():
        mentioned.update(profile.upstream_services)
        mentioned.update(profile.downstream_services)

    for svc in sorted(mentioned):
        if svc not in all_profiles:
            findings.append(
                _make_finding(
                    "E3",
                    f"Service '{svc}' mentioned in topology but has no profile",
                    smap,
                    details={"missing_service": svc},
                )
            )
    return findings


def check_premature_termination(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
    evidence_findings: list[SanitizerFinding] | None = None,
) -> list[SanitizerFinding]:
    """E4: Warn if finalizing with budget remaining and open evidence gaps."""
    smap = severity_map or {}
    max_steps = ctx.max_steps or 0
    if max_steps <= 0:
        return []

    remaining_ratio = max(0, max_steps - ctx.step) / max_steps
    if remaining_ratio <= 0.3:
        return []

    # Check for any E1/E2/E3 findings
    has_evidence_gaps = bool(evidence_findings)
    if not has_evidence_gaps:
        return []

    return [
        _make_finding(
            "E4",
            f"Finalizing with {remaining_ratio:.0%} budget remaining "
            f"and open evidence gaps",
            smap,
            details={"remaining_ratio": remaining_ratio},
        )
    ]


def check_skipped_verify(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
) -> list[SanitizerFinding]:
    """C1: Check if confirmed hypotheses have a verify task completion."""
    smap = severity_map or {}
    findings: list[SanitizerFinding] = []
    completions = tracker.task_completions()

    for h in hypothesis_store.get_all().values():
        if h.status != "confirmed":
            continue
        has_verify = any(
            tc.data.get("task_type") == "verify" and tc.data.get("hypothesis_id") == h.id
            for tc in completions
        )
        if not has_verify:
            findings.append(
                _make_finding(
                    "C1",
                    f"Hypothesis '{h.id}' confirmed without verify task",
                    smap,
                    details={"hypothesis_id": h.id},
                )
            )
    return findings


def check_unresolved_contradiction(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
) -> list[SanitizerFinding]:
    """C2: Check if contradicted hypotheses have follow-up dispatches."""
    smap = severity_map or {}
    findings: list[SanitizerFinding] = []
    completions = tracker.task_completions()
    dispatches = tracker.dispatches()

    for tc in completions:
        if tc.data.get("verdict") != "CONTRADICTED":
            continue
        hyp_id = tc.data.get("hypothesis_id")
        if not hyp_id:
            continue
        # Check for any dispatch after this round referencing the same hypothesis
        has_followup = any(
            d.round > tc.round and d.data.get("hypothesis_id") == hyp_id
            for d in dispatches
        )
        if not has_followup:
            findings.append(
                _make_finding(
                    "C2",
                    f"Contradiction for hypothesis '{hyp_id}' at round "
                    f"{tc.round} has no follow-up dispatch",
                    smap,
                    details={"hypothesis_id": hyp_id, "contradiction_round": tc.round},
                )
            )
    return findings


def check_no_alternative(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
) -> list[SanitizerFinding]:
    """C4: Warn if only one hypothesis exists and it is confirmed."""
    smap = severity_map or {}
    active_statuses = {"investigating", "confirmed", "rejected", "refined", "inconclusive"}
    all_hyps = hypothesis_store.get_all()

    active = [h for h in all_hyps.values() if h.status in active_statuses]
    has_confirmed = any(h.status == "confirmed" for h in active)

    if len(active) <= 1 and has_confirmed:
        return [
            _make_finding(
                "C4",
                "Root cause confirmed with no alternative hypotheses explored",
                smap,
            )
        ]
    return []


def check_investigation_drift(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
    drift_window: int = 3,
) -> list[SanitizerFinding]:
    """J2: Warn if the last N dispatches all target the same services and hypothesis."""
    smap = severity_map or {}
    dispatches = tracker.dispatches()

    if len(dispatches) < drift_window:
        return []

    recent = dispatches[-drift_window:]
    # Extract hypothesis IDs and target services
    hyp_ids = {d.data.get("hypothesis_id") for d in recent}
    service_sets = [frozenset(d.data.get("target_services", [])) for d in recent]

    # All same hypothesis and same services?
    if len(hyp_ids) == 1 and hyp_ids != {None} and len(set(service_sets)) == 1:
        return [
            _make_finding(
                "J2",
                f"Last {drift_window} dispatches all target the same "
                f"hypothesis and services",
                smap,
                details={
                    "hypothesis_id": next(iter(hyp_ids)),
                    "target_services": sorted(next(iter(service_sets))),
                },
            )
        ]
    return []


def check_incomplete_chain(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
) -> list[SanitizerFinding]:
    """J3: Check if all anomalous services are covered by the confirmed hypothesis."""
    smap = severity_map or {}
    confirmed_id = hypothesis_store.confirmed_id
    if not confirmed_id:
        return []

    confirmed = hypothesis_store.get(confirmed_id)
    if not confirmed:
        return []

    anomalous = profile_store.query(anomalous_only=True)
    if not anomalous:
        return []

    # Build the text to search: description + all evidence
    search_text = confirmed.description.lower()
    for ev in confirmed.evidence:
        search_text += " " + ev.lower()

    findings: list[SanitizerFinding] = []
    for profile in anomalous:
        svc = profile.service_name.lower()
        # Use word-boundary match to avoid "a" matching inside "database"
        if not re.search(r"\b" + re.escape(svc) + r"\b", search_text):
            findings.append(
                _make_finding(
                    "J3",
                    f"Anomalous service '{profile.service_name}' not mentioned "
                    f"in confirmed hypothesis",
                    smap,
                    details={
                        "service": profile.service_name,
                        "hypothesis_id": confirmed_id,
                    },
                )
            )
    return findings


def check_hypothesis_before_scout(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
) -> list[SanitizerFinding]:
    """P1: Warn if a hypothesis was formed before any scout task completed."""
    smap = severity_map or {}
    hyp_changes = tracker.hypothesis_changes()
    completions = tracker.task_completions()

    if not hyp_changes:
        return []

    first_hyp_round = hyp_changes[0].round

    scout_completions = [
        tc for tc in completions if tc.data.get("task_type") == "scout"
    ]
    if not scout_completions:
        # No scout at all — if there's a hypothesis, that's a problem
        return [
            _make_finding(
                "P1",
                "Hypothesis formed before any scout task completed",
                smap,
                details={"first_hypothesis_round": first_hyp_round},
            )
        ]

    first_scout_round = scout_completions[0].round
    if first_hyp_round < first_scout_round:
        return [
            _make_finding(
                "P1",
                f"Hypothesis formed at round {first_hyp_round} before "
                f"first scout completed at round {first_scout_round}",
                smap,
                details={
                    "first_hypothesis_round": first_hyp_round,
                    "first_scout_round": first_scout_round,
                },
            )
        ]
    return []


def check_profile_write_without_read(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    ctx: LoopContext,
    *,
    severity_map: dict[str, Severity] | None = None,
) -> list[SanitizerFinding]:
    """P3: Check if profile updates happened without a prior read for that service."""
    smap = severity_map or {}
    findings: list[SanitizerFinding] = []

    writes = tracker.tool_calls_for("update_service_profile")
    reads = tracker.tool_calls_for("query_service_profile")

    for write in writes:
        write_svc = write.data.get("service_name", "")
        if not write_svc:
            continue
        # Check if there's any prior read referencing this service
        has_prior_read = any(
            r.round <= write.round and r.data.get("service_name") == write_svc
            for r in reads
        )
        if not has_prior_read:
            findings.append(
                _make_finding(
                    "P3",
                    f"Profile update for '{write_svc}' without prior query",
                    smap,
                    details={"service_name": write_svc, "write_round": write.round},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Trigger routing
# ---------------------------------------------------------------------------

_TRIGGER_MAP: dict[str, list[str]] = {
    "every_round": ["J2", "P3"],
    "periodic": ["E1", "E2", "E3"],
    "hypothesis_change": ["C1", "C2", "C4", "P1"],
    "pre_finalize": [
        "E1", "E2", "E3", "E4", "C1", "C2", "C4", "J2", "J3", "P1", "P3",
    ],
}

# Map from check code to function
_CHECK_REGISTRY: dict[str, Callable[..., list[SanitizerFinding]]] = {
    "E1": check_anchoring_bias,
    "E2": check_dimension_gap,
    "E3": check_coverage_gap,
    "E4": check_premature_termination,
    "C1": check_skipped_verify,
    "C2": check_unresolved_contradiction,
    "C4": check_no_alternative,
    "J2": check_investigation_drift,
    "J3": check_incomplete_chain,
    "P1": check_hypothesis_before_scout,
    "P3": check_profile_write_without_read,
}


class CodeSanitizer:
    """Deterministic code-based sanitizer for RCA investigations.

    Routes triggers to the appropriate checks, applies severity overrides,
    and filters disabled checks.
    """

    def __init__(
        self,
        severity_map: dict[str, Severity] | None = None,
        disabled: set[str] | None = None,
        drift_window: int = 3,
    ) -> None:
        self._severity_map = severity_map or {}
        self._disabled = disabled or set()
        self._drift_window = drift_window

    def check(
        self,
        trigger: str,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        tracker: InvestigationTracker,
        ctx: LoopContext,
    ) -> list[SanitizerFinding]:
        """Run checks for the given trigger and return findings."""
        codes = _TRIGGER_MAP.get(trigger, [])
        active_codes = [c for c in codes if c not in self._disabled]

        findings: list[SanitizerFinding] = []

        # For E4, we need evidence findings from E1/E2/E3
        evidence_findings: list[SanitizerFinding] = []

        for code in active_codes:
            if code == "E4":
                # E4 needs evidence findings collected from E1/E2/E3
                continue  # handled below

            fn = _CHECK_REGISTRY[code]
            kwargs: dict[str, object] = {"severity_map": self._severity_map}

            if code == "J2":
                kwargs["drift_window"] = self._drift_window

            result = fn(
                hypothesis_store, profile_store, tracker, ctx, **kwargs
            )
            findings.extend(result)

            # Collect evidence findings for E4
            if code in ("E1", "E2", "E3"):
                evidence_findings.extend(result)

        # Run E4 if active
        if "E4" in active_codes:
            e4_result = check_premature_termination(
                hypothesis_store,
                profile_store,
                tracker,
                ctx,
                severity_map=self._severity_map,
                evidence_findings=evidence_findings,
            )
            findings.extend(e4_result)

        return findings
