"""Tests for CodeSanitizer — deterministic RCA investigation checks."""

from __future__ import annotations

import pytest

from agentm.harness.types import LoopContext
from agentm.scenarios.rca.hypothesis_store import HypothesisStore
from agentm.scenarios.rca.sanitizer.code_sanitizer import (
    CodeSanitizer,
    check_anchoring_bias,
    check_coverage_gap,
    check_dimension_gap,
    check_hypothesis_before_scout,
    check_incomplete_chain,
    check_investigation_drift,
    check_no_alternative,
    check_premature_termination,
    check_profile_write_without_read,
    check_skipped_verify,
    check_unresolved_contradiction,
)
from agentm.scenarios.rca.sanitizer.models import Severity
from agentm.scenarios.rca.sanitizer.tracker import InvestigationTracker
from agentm.scenarios.rca.service_profile import ServiceProfileStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(step: int = 5, max_steps: int = 10) -> LoopContext:
    return LoopContext(
        agent_id="test",
        step=step,
        max_steps=max_steps,
        tool_call_count=0,
        metadata={},
    )


def _empty_stores() -> tuple[HypothesisStore, ServiceProfileStore, InvestigationTracker]:
    return HypothesisStore(), ServiceProfileStore(), InvestigationTracker()


# ===========================================================================
# E1: check_anchoring_bias
# ===========================================================================

class TestE1AnchoringBias:
    def test_fires_when_upstream_has_no_observations(self) -> None:
        hs, ps, tr = _empty_stores()
        ps.update("svc-a", is_anomalous=True, upstream_services=["svc-b"])
        # svc-b not in store at all
        findings = check_anchoring_bias(hs, ps, tr, _ctx())
        assert len(findings) == 1
        assert findings[0].code == "E1"
        assert findings[0].severity == Severity.WARN

    def test_clean_when_upstream_has_observations(self) -> None:
        hs, ps, tr = _empty_stores()
        ps.update("svc-a", is_anomalous=True, upstream_services=["svc-b"])
        ps.update("svc-b", key_observation="looks fine")
        findings = check_anchoring_bias(hs, ps, tr, _ctx())
        assert findings == []


# ===========================================================================
# E2: check_dimension_gap
# ===========================================================================

class TestE2DimensionGap:
    def test_fires_when_dimensions_missing(self) -> None:
        hs, ps, tr = _empty_stores()
        ps.update("svc-a", is_anomalous=True, data_sources_queried=["latency_p99"])
        findings = check_dimension_gap(hs, ps, tr, _ctx())
        assert len(findings) == 1
        assert findings[0].code == "E2"
        assert "error_rate" in findings[0].details["missing_dimensions"]

    def test_clean_when_all_dimensions_covered(self) -> None:
        hs, ps, tr = _empty_stores()
        ps.update(
            "svc-a",
            is_anomalous=True,
            data_sources_queried=[
                "latency_p99", "error_rate_5xx", "call_volume_total", "cpu_resource"
            ],
        )
        findings = check_dimension_gap(hs, ps, tr, _ctx())
        assert findings == []

    def test_keyword_matching_is_case_insensitive(self) -> None:
        hs, ps, tr = _empty_stores()
        ps.update(
            "svc-a",
            is_anomalous=True,
            data_sources_queried=["Latency_P99", "ERROR_count", "Call_Volume", "CPU_metric"],
        )
        findings = check_dimension_gap(hs, ps, tr, _ctx())
        assert findings == []


# ===========================================================================
# E3: check_coverage_gap
# ===========================================================================

class TestE3CoverageGap:
    def test_fires_when_topology_service_missing(self) -> None:
        hs, ps, tr = _empty_stores()
        ps.update("svc-a", downstream_services=["svc-b"])
        # svc-b not in store
        findings = check_coverage_gap(hs, ps, tr, _ctx())
        assert len(findings) == 1
        assert findings[0].code == "E3"
        assert findings[0].details["missing_service"] == "svc-b"

    def test_clean_when_all_topology_services_exist(self) -> None:
        hs, ps, tr = _empty_stores()
        ps.update("svc-a", downstream_services=["svc-b"])
        ps.update("svc-b")
        findings = check_coverage_gap(hs, ps, tr, _ctx())
        assert findings == []


# ===========================================================================
# E4: check_premature_termination
# ===========================================================================

class TestE4PrematureTermination:
    def test_fires_with_budget_remaining_and_gaps(self) -> None:
        hs, ps, tr = _empty_stores()
        from agentm.scenarios.rca.sanitizer.models import SanitizerFinding
        evidence = [SanitizerFinding(code="E1", severity=Severity.WARN, message="gap")]
        # step=2, max_steps=10 → remaining=0.8 > 0.3
        findings = check_premature_termination(
            hs, ps, tr, _ctx(step=2, max_steps=10), evidence_findings=evidence
        )
        assert len(findings) == 1
        assert findings[0].code == "E4"

    def test_clean_when_budget_low(self) -> None:
        hs, ps, tr = _empty_stores()
        from agentm.scenarios.rca.sanitizer.models import SanitizerFinding
        evidence = [SanitizerFinding(code="E1", severity=Severity.WARN, message="gap")]
        # step=8, max_steps=10 → remaining=0.2 ≤ 0.3
        findings = check_premature_termination(
            hs, ps, tr, _ctx(step=8, max_steps=10), evidence_findings=evidence
        )
        assert findings == []

    def test_clean_when_no_evidence_gaps(self) -> None:
        hs, ps, tr = _empty_stores()
        findings = check_premature_termination(
            hs, ps, tr, _ctx(step=2, max_steps=10), evidence_findings=[]
        )
        assert findings == []


# ===========================================================================
# C1: check_skipped_verify
# ===========================================================================

class TestC1SkippedVerify:
    def test_fires_when_confirmed_without_verify(self) -> None:
        hs, ps, tr = _empty_stores()
        hs.update("h1", "some hypothesis", status="confirmed")
        findings = check_skipped_verify(hs, ps, tr, _ctx())
        assert len(findings) == 1
        assert findings[0].code == "C1"
        assert findings[0].severity == Severity.BLOCK

    def test_clean_when_verify_exists(self) -> None:
        hs, ps, tr = _empty_stores()
        hs.update("h1", "some hypothesis", status="confirmed")
        tr.record(3, "task_complete", {"task_type": "verify", "hypothesis_id": "h1"})
        findings = check_skipped_verify(hs, ps, tr, _ctx())
        assert findings == []


# ===========================================================================
# C2: check_unresolved_contradiction
# ===========================================================================

class TestC2UnresolvedContradiction:
    def test_fires_when_no_followup_dispatch(self) -> None:
        hs, ps, tr = _empty_stores()
        tr.record(3, "task_complete", {"verdict": "CONTRADICTED", "hypothesis_id": "h1"})
        findings = check_unresolved_contradiction(hs, ps, tr, _ctx())
        assert len(findings) == 1
        assert findings[0].code == "C2"
        assert findings[0].severity == Severity.BLOCK

    def test_clean_when_followup_exists(self) -> None:
        hs, ps, tr = _empty_stores()
        tr.record(3, "task_complete", {"verdict": "CONTRADICTED", "hypothesis_id": "h1"})
        tr.record(4, "dispatch", {"hypothesis_id": "h1"})
        findings = check_unresolved_contradiction(hs, ps, tr, _ctx())
        assert findings == []


# ===========================================================================
# C4: check_no_alternative
# ===========================================================================

class TestC4NoAlternative:
    def test_fires_when_single_confirmed(self) -> None:
        hs, ps, tr = _empty_stores()
        hs.update("h1", "root cause", status="confirmed")
        findings = check_no_alternative(hs, ps, tr, _ctx())
        assert len(findings) == 1
        assert findings[0].code == "C4"

    def test_clean_when_multiple_hypotheses(self) -> None:
        hs, ps, tr = _empty_stores()
        hs.update("h1", "root cause", status="confirmed")
        hs.update("h2", "alternative", status="rejected")
        findings = check_no_alternative(hs, ps, tr, _ctx())
        assert findings == []


# ===========================================================================
# J2: check_investigation_drift
# ===========================================================================

class TestJ2InvestigationDrift:
    def test_fires_when_dispatches_repeat(self) -> None:
        hs, ps, tr = _empty_stores()
        for i in range(3):
            tr.record(
                i + 1,
                "dispatch",
                {"hypothesis_id": "h1", "target_services": ["svc-a"]},
            )
        findings = check_investigation_drift(hs, ps, tr, _ctx())
        assert len(findings) == 1
        assert findings[0].code == "J2"

    def test_clean_when_dispatches_vary(self) -> None:
        hs, ps, tr = _empty_stores()
        tr.record(1, "dispatch", {"hypothesis_id": "h1", "target_services": ["svc-a"]})
        tr.record(2, "dispatch", {"hypothesis_id": "h1", "target_services": ["svc-b"]})
        tr.record(3, "dispatch", {"hypothesis_id": "h2", "target_services": ["svc-a"]})
        findings = check_investigation_drift(hs, ps, tr, _ctx())
        assert findings == []


# ===========================================================================
# J3: check_incomplete_chain
# ===========================================================================

class TestJ3IncompleteChain:
    def test_fires_when_anomalous_not_in_hypothesis(self) -> None:
        hs, ps, tr = _empty_stores()
        hs.update("h1", "issue in svc-a caused timeout", status="confirmed")
        ps.update("svc-a", is_anomalous=True)
        ps.update("svc-b", is_anomalous=True)
        findings = check_incomplete_chain(hs, ps, tr, _ctx())
        assert len(findings) == 1
        assert findings[0].code == "J3"
        assert findings[0].details["service"] == "svc-b"

    def test_clean_when_all_anomalous_mentioned(self) -> None:
        hs, ps, tr = _empty_stores()
        hs.update("h1", "issue in svc-a and svc-b", status="confirmed")
        ps.update("svc-a", is_anomalous=True)
        ps.update("svc-b", is_anomalous=True)
        findings = check_incomplete_chain(hs, ps, tr, _ctx())
        assert findings == []


# ===========================================================================
# P1: check_hypothesis_before_scout
# ===========================================================================

class TestP1HypothesisBeforeScout:
    def test_fires_when_hypothesis_before_scout(self) -> None:
        hs, ps, tr = _empty_stores()
        tr.record(1, "hypothesis_change", {"hypothesis_id": "h1"})
        tr.record(3, "task_complete", {"task_type": "scout"})
        findings = check_hypothesis_before_scout(hs, ps, tr, _ctx())
        assert len(findings) == 1
        assert findings[0].code == "P1"

    def test_clean_when_scout_comes_first(self) -> None:
        hs, ps, tr = _empty_stores()
        tr.record(1, "task_complete", {"task_type": "scout"})
        tr.record(2, "hypothesis_change", {"hypothesis_id": "h1"})
        findings = check_hypothesis_before_scout(hs, ps, tr, _ctx())
        assert findings == []


# ===========================================================================
# P3: check_profile_write_without_read
# ===========================================================================

class TestP3ProfileWriteWithoutRead:
    def test_fires_when_write_without_prior_read(self) -> None:
        hs, ps, tr = _empty_stores()
        tr.record(2, "tool_call", {"tool_name": "update_service_profile", "service_name": "svc-a"})
        findings = check_profile_write_without_read(hs, ps, tr, _ctx())
        assert len(findings) == 1
        assert findings[0].code == "P3"
        assert findings[0].severity == Severity.INFO

    def test_clean_when_read_before_write(self) -> None:
        hs, ps, tr = _empty_stores()
        tr.record(1, "tool_call", {"tool_name": "query_service_profile", "service_name": "svc-a"})
        tr.record(2, "tool_call", {"tool_name": "update_service_profile", "service_name": "svc-a"})
        findings = check_profile_write_without_read(hs, ps, tr, _ctx())
        assert findings == []


# ===========================================================================
# CodeSanitizer — routing, overrides, disabled
# ===========================================================================

class TestCodeSanitizerRouting:
    def test_every_round_only_returns_j2_p3(self) -> None:
        """every_round trigger should only run J2 and P3 checks."""
        hs, ps, tr = _empty_stores()
        # Set up state that would trigger E1 (upstream without obs)
        ps.update("svc-a", is_anomalous=True, upstream_services=["svc-b"])
        # Set up state that would trigger P3
        tr.record(2, "tool_call", {"tool_name": "update_service_profile", "service_name": "svc-x"})

        sanitizer = CodeSanitizer()
        findings = sanitizer.check("every_round", hs, ps, tr, _ctx())
        codes = {f.code for f in findings}
        # E1 should NOT appear (not in every_round trigger)
        assert "E1" not in codes
        # P3 should appear
        assert "P3" in codes

    def test_severity_override(self) -> None:
        """Severity override should change C1 from BLOCK to WARN."""
        hs, ps, tr = _empty_stores()
        hs.update("h1", "some hypothesis", status="confirmed")
        sanitizer = CodeSanitizer(severity_map={"C1": Severity.WARN})
        findings = sanitizer.check("hypothesis_change", hs, ps, tr, _ctx())
        c1 = [f for f in findings if f.code == "C1"]
        assert len(c1) == 1
        assert c1[0].severity == Severity.WARN

    def test_disabled_check(self) -> None:
        """Disabled check should produce no findings even when it would fire."""
        hs, ps, tr = _empty_stores()
        ps.update("svc-a", is_anomalous=True, upstream_services=["svc-b"])
        sanitizer = CodeSanitizer(disabled={"E1"})
        findings = sanitizer.check("periodic", hs, ps, tr, _ctx())
        codes = {f.code for f in findings}
        assert "E1" not in codes

    def test_pre_finalize_runs_all_checks(self) -> None:
        """pre_finalize trigger should run all check codes."""
        hs, ps, tr = _empty_stores()
        sanitizer = CodeSanitizer()
        # Just verify it runs without error with empty stores
        findings = sanitizer.check("pre_finalize", hs, ps, tr, _ctx())
        # With empty stores, most checks return nothing
        assert isinstance(findings, list)
