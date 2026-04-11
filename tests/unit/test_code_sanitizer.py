"""Focused regression tests for key CodeSanitizer checks and routing."""

from __future__ import annotations

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
from agentm.scenarios.rca.sanitizer.models import SanitizerFinding, Severity
from agentm.scenarios.rca.sanitizer.tracker import InvestigationTracker
from agentm.scenarios.rca.service_profile import ServiceProfileStore


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


def test_e1_anchoring_bias_fires_without_upstream_observations() -> None:
    hs, ps, tr = _empty_stores()
    ps.update("svc-a", is_anomalous=True, upstream_services=["svc-b"])
    findings = check_anchoring_bias(hs, ps, tr, _ctx())
    assert [f.code for f in findings] == ["E1"]


def test_e2_dimension_gap_accepts_case_insensitive_dimension_keywords() -> None:
    hs, ps, tr = _empty_stores()
    ps.update(
        "svc-a",
        is_anomalous=True,
        data_sources_queried=["Latency_P99", "ERROR_count", "Call_Volume", "CPU_metric"],
    )
    assert check_dimension_gap(hs, ps, tr, _ctx()) == []


def test_e3_coverage_gap_fires_for_missing_topology_service() -> None:
    hs, ps, tr = _empty_stores()
    ps.update("svc-a", downstream_services=["svc-b"])
    findings = check_coverage_gap(hs, ps, tr, _ctx())
    assert [f.code for f in findings] == ["E3"]


def test_e4_premature_termination_fires_when_budget_remaining_and_gaps_exist() -> None:
    hs, ps, tr = _empty_stores()
    evidence = [SanitizerFinding(code="E1", severity=Severity.WARN, message="gap")]
    findings = check_premature_termination(hs, ps, tr, _ctx(step=2, max_steps=10), evidence_findings=evidence)
    assert [f.code for f in findings] == ["E4"]


def test_c1_c2_c4_checks_cover_critical_hypothesis_safety_guards() -> None:
    hs, ps, tr = _empty_stores()
    hs.update("h1", "some hypothesis", status="confirmed")
    c1 = check_skipped_verify(hs, ps, tr, _ctx())
    assert [f.code for f in c1] == ["C1"]

    hs2, ps2, tr2 = _empty_stores()
    tr2.record(3, "task_complete", {"verdict": "CONTRADICTED", "hypothesis_id": "h1"})
    c2 = check_unresolved_contradiction(hs2, ps2, tr2, _ctx())
    assert [f.code for f in c2] == ["C2"]

    hs3, ps3, tr3 = _empty_stores()
    hs3.update("h1", "root cause", status="confirmed")
    c4 = check_no_alternative(hs3, ps3, tr3, _ctx())
    assert [f.code for f in c4] == ["C4"]


def test_j2_j3_checks_cover_drift_and_incomplete_chain() -> None:
    hs, ps, tr = _empty_stores()
    for i in range(3):
        tr.record(i + 1, "dispatch", {"hypothesis_id": "h1", "target_services": ["svc-a"]})
    assert [f.code for f in check_investigation_drift(hs, ps, tr, _ctx())] == ["J2"]

    hs2, ps2, tr2 = _empty_stores()
    hs2.update("h1", "issue in svc-a caused timeout", status="confirmed")
    ps2.update("svc-a", is_anomalous=True)
    ps2.update("svc-b", is_anomalous=True)
    assert [f.code for f in check_incomplete_chain(hs2, ps2, tr2, _ctx())] == ["J3"]


def test_p1_and_p3_checks_cover_ordering_and_profile_read_before_write() -> None:
    hs, ps, tr = _empty_stores()
    tr.record(1, "hypothesis_change", {"hypothesis_id": "h1"})
    tr.record(3, "task_complete", {"task_type": "scout"})
    assert [f.code for f in check_hypothesis_before_scout(hs, ps, tr, _ctx())] == ["P1"]

    hs2, ps2, tr2 = _empty_stores()
    tr2.record(2, "tool_call", {"tool_name": "update_service_profile", "service_name": "svc-a"})
    assert [f.code for f in check_profile_write_without_read(hs2, ps2, tr2, _ctx())] == ["P3"]


def test_code_sanitizer_routing_overrides_and_disable_controls() -> None:
    hs, ps, tr = _empty_stores()
    ps.update("svc-a", is_anomalous=True, upstream_services=["svc-b"])
    tr.record(2, "tool_call", {"tool_name": "update_service_profile", "service_name": "svc-x"})

    findings = CodeSanitizer().check("every_round", hs, ps, tr, _ctx())
    codes = {f.code for f in findings}
    assert "P3" in codes
    assert "E1" not in codes

    hs2, ps2, tr2 = _empty_stores()
    hs2.update("h1", "confirmed", status="confirmed")
    overridden = CodeSanitizer(severity_map={"C1": Severity.WARN}).check("hypothesis_change", hs2, ps2, tr2, _ctx())
    assert [f.severity for f in overridden if f.code == "C1"] == [Severity.WARN]

    hs3, ps3, tr3 = _empty_stores()
    ps3.update("svc-a", is_anomalous=True, upstream_services=["svc-b"])
    disabled = CodeSanitizer(disabled={"E1"}).check("periodic", hs3, ps3, tr3, _ctx())
    assert "E1" not in {f.code for f in disabled}
