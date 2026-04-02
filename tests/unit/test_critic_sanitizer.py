"""Tests for CriticSanitizer — LLM-based semantic checks."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentm.harness.types import LoopContext
from agentm.scenarios.rca.hypothesis_store import HypothesisStore
from agentm.scenarios.rca.sanitizer.critic_sanitizer import (
    CausalDirectionResult,
    CriticSanitizer,
    DispatchQualityResult,
    SignalMisreadResult,
    SymptomAsCauseResult,
)
from agentm.scenarios.rca.sanitizer.models import Severity
from agentm.scenarios.rca.sanitizer.tracker import InvestigationTracker
from agentm.scenarios.rca.service_profile import ServiceProfileStore


# ---------------------------------------------------------------------------
# Mock model helpers
# ---------------------------------------------------------------------------


class MockStructuredModel:
    """Mock structured output model that returns a preconfigured response."""

    def __init__(self, response: Any) -> None:
        self._response = response

    async def ainvoke(self, messages: Any) -> Any:
        return self._response


class MockModel:
    """Mock model that returns preconfigured structured outputs per schema.

    Pass a dict mapping schema classes to responses, or a single response
    that will be returned for all schemas.
    """

    def __init__(self, response: Any | dict[type, Any]) -> None:
        self._response = response
        self.called = False

    def with_structured_output(self, schema: type, method: str = "function_calling") -> MockStructuredModel:
        self.called = True
        if isinstance(self._response, dict):
            return MockStructuredModel(self._response.get(schema))
        return MockStructuredModel(self._response)

    async def ainvoke(self, messages: Any) -> Any:
        return type("R", (), {"content": ""})()


class ExplodingStructuredModel:
    """Mock that raises on ainvoke."""

    async def ainvoke(self, messages: Any) -> Any:
        raise RuntimeError("LLM exploded")


class ExplodingModel:
    """Mock model whose structured output always raises."""

    def with_structured_output(self, schema: type, method: str = "function_calling") -> ExplodingStructuredModel:
        return ExplodingStructuredModel()

    async def ainvoke(self, messages: Any) -> Any:
        raise RuntimeError("LLM exploded")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ctx() -> LoopContext:
    return LoopContext(agent_id="test", step=5, max_steps=10, tool_call_count=3, metadata={})


def _make_hypothesis_store_with_confirmed() -> HypothesisStore:
    """Create a store with a confirmed hypothesis referencing `payment-svc`."""
    store = HypothesisStore()
    store.update("H1", "Root cause is `payment-svc` timeout", status="confirmed", evidence_summary="e1")
    # Add more evidence
    store.update("H1", "Root cause is `payment-svc` timeout", status="confirmed", evidence_summary="e2")
    return store


def _make_profile_store_with_upstream() -> ServiceProfileStore:
    """Create a store where payment-svc has anomalous upstream."""
    store = ServiceProfileStore()
    store.update("payment-svc", is_anomalous=True, anomaly_summary="high latency", upstream_services=["db-svc"])
    store.update("db-svc", is_anomalous=True, anomaly_summary="connection pool exhausted")
    return store


def _make_tracker_with_deep_analyze() -> InvestigationTracker:
    """Create a tracker with deep_analyze task completions."""
    tracker = InvestigationTracker()
    tracker.record(1, "task_complete", {"task_type": "deep_analyze", "summary": "Found latency spike"})
    return tracker


# ---------------------------------------------------------------------------
# C3 tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c3_fires_when_not_proven() -> None:
    """C3 fires BLOCK finding when causal direction is not proven."""
    model = MockModel({
        CausalDirectionResult: CausalDirectionResult(proven=False, missing_evidence="no time attribution"),
        SignalMisreadResult: SignalMisreadResult(misread=False),
        SymptomAsCauseResult: SymptomAsCauseResult(likely_symptom=False),
    })
    sanitizer = CriticSanitizer(model)

    findings = await sanitizer.check(
        "pre_finalize",
        _make_hypothesis_store_with_confirmed(),
        _make_profile_store_with_upstream(),
        _make_tracker_with_deep_analyze(),
        _make_ctx(),
    )

    assert len(findings) == 1
    assert findings[0].code == "C3"
    assert findings[0].severity == Severity.BLOCK
    assert "no time attribution" in findings[0].message


@pytest.mark.asyncio
async def test_c3_clean_when_proven() -> None:
    """C3 produces no findings when causal direction is proven."""
    model = MockModel({
        CausalDirectionResult: CausalDirectionResult(proven=True),
        SignalMisreadResult: SignalMisreadResult(misread=False),
        SymptomAsCauseResult: SymptomAsCauseResult(likely_symptom=False),
    })
    sanitizer = CriticSanitizer(model)

    findings = await sanitizer.check(
        "pre_finalize",
        _make_hypothesis_store_with_confirmed(),
        _make_profile_store_with_upstream(),
        _make_tracker_with_deep_analyze(),
        _make_ctx(),
    )

    assert len(findings) == 0


@pytest.mark.asyncio
async def test_c3_skip_no_confirmed_hypothesis() -> None:
    """C3 skips when no hypothesis is confirmed, no LLM call made."""
    model = MockModel(CausalDirectionResult(proven=False, missing_evidence="should not appear"))
    sanitizer = CriticSanitizer(model)

    empty_store = HypothesisStore()
    findings = await sanitizer.check(
        "pre_finalize",
        empty_store,
        ServiceProfileStore(),
        InvestigationTracker(),
        _make_ctx(),
    )

    c3_findings = [f for f in findings if f.code == "C3"]
    assert len(c3_findings) == 0


# ---------------------------------------------------------------------------
# J1 tests
# ---------------------------------------------------------------------------


def _make_store_with_rejected_hypothesis() -> HypothesisStore:
    """Create a store with a rejected hypothesis that had 3+ evidence items."""
    store = HypothesisStore()
    store.update("H1", "Some hypothesis", status="investigating", evidence_summary="ev1")
    store.update("H1", "Some hypothesis", status="investigating", evidence_summary="ev2")
    store.update("H1", "Some hypothesis", status="rejected", evidence_summary="ev3")
    return store


def _make_tracker_with_rejection() -> InvestigationTracker:
    tracker = InvestigationTracker()
    tracker.record(2, "hypothesis_change", {
        "hypothesis_id": "H1",
        "new_status": "rejected",
        "reason": "conflicting metric",
    })
    return tracker


@pytest.mark.asyncio
async def test_j1_fires_on_misread() -> None:
    """J1 fires when a rejected hypothesis with 3+ evidence is flagged as misread."""
    model = MockModel(SignalMisreadResult(misread=True, reasoning="evidence was strong"))
    sanitizer = CriticSanitizer(model)

    findings = await sanitizer.check(
        "pre_finalize",
        _make_store_with_rejected_hypothesis(),
        ServiceProfileStore(),
        _make_tracker_with_rejection(),
        _make_ctx(),
    )

    j1_findings = [f for f in findings if f.code == "J1"]
    assert len(j1_findings) == 1
    assert j1_findings[0].severity == Severity.WARN
    assert "evidence was strong" in j1_findings[0].message


@pytest.mark.asyncio
async def test_j1_skip_no_rejected_with_evidence() -> None:
    """J1 skips when no rejected hypothesis has >= 3 evidence items."""
    model = MockModel(SignalMisreadResult(misread=True, reasoning="should not appear"))
    sanitizer = CriticSanitizer(model)

    # Only 1 evidence item
    store = HypothesisStore()
    store.update("H1", "Some hypothesis", status="rejected", evidence_summary="ev1")

    findings = await sanitizer.check(
        "pre_finalize",
        store,
        ServiceProfileStore(),
        InvestigationTracker(),
        _make_ctx(),
    )

    j1_findings = [f for f in findings if f.code == "J1"]
    assert len(j1_findings) == 0


# ---------------------------------------------------------------------------
# J4 tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_j4_fires_when_symptom_detected() -> None:
    """J4 fires when confirmed service likely a symptom of upstream failure."""
    model = MockModel({
        CausalDirectionResult: CausalDirectionResult(proven=True),
        SignalMisreadResult: SignalMisreadResult(misread=False),
        SymptomAsCauseResult: SymptomAsCauseResult(likely_symptom=True, suspect_upstream=["db-svc"]),
    })
    sanitizer = CriticSanitizer(model)

    findings = await sanitizer.check(
        "pre_finalize",
        _make_hypothesis_store_with_confirmed(),
        _make_profile_store_with_upstream(),
        InvestigationTracker(),
        _make_ctx(),
    )

    j4_findings = [f for f in findings if f.code == "J4"]
    assert len(j4_findings) == 1
    assert "db-svc" in j4_findings[0].message
    assert j4_findings[0].severity == Severity.WARN


@pytest.mark.asyncio
async def test_j4_skip_no_anomalous_upstream() -> None:
    """J4 skips when confirmed service has no anomalous upstream."""
    model = MockModel({
        CausalDirectionResult: CausalDirectionResult(proven=True),
        SignalMisreadResult: SignalMisreadResult(misread=False),
        SymptomAsCauseResult: SymptomAsCauseResult(likely_symptom=True, suspect_upstream=["db-svc"]),
    })
    sanitizer = CriticSanitizer(model)

    store = ServiceProfileStore()
    store.update("payment-svc", is_anomalous=True, upstream_services=["db-svc"])
    store.update("db-svc", is_anomalous=False)  # healthy upstream

    findings = await sanitizer.check(
        "pre_finalize",
        _make_hypothesis_store_with_confirmed(),
        store,
        InvestigationTracker(),
        _make_ctx(),
    )

    j4_findings = [f for f in findings if f.code == "J4"]
    assert len(j4_findings) == 0


# ---------------------------------------------------------------------------
# P2 tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p2_async_fires() -> None:
    """P2 fires asynchronously on dispatch with poor quality instruction."""
    model = MockModel(DispatchQualityResult(quality="poor", missing=["service names", "metrics"]))
    sanitizer = CriticSanitizer(model)

    tracker = InvestigationTracker()
    tracker.record(1, "dispatch", {"instruction": "go investigate stuff"})

    await sanitizer.check_async(
        "dispatch",
        HypothesisStore(),
        ServiceProfileStore(),
        tracker,
        _make_ctx(),
    )

    # Let the task complete
    await asyncio.sleep(0.05)

    findings = sanitizer.collect_async_results()
    assert len(findings) == 1
    assert findings[0].code == "P2"
    assert findings[0].severity == Severity.INFO
    assert "service names" in findings[0].message


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_failure_graceful_degradation() -> None:
    """LLM exceptions are caught — no crash, no findings."""
    sanitizer = CriticSanitizer(ExplodingModel())

    findings = await sanitizer.check(
        "pre_finalize",
        _make_hypothesis_store_with_confirmed(),
        _make_profile_store_with_upstream(),
        _make_tracker_with_deep_analyze(),
        _make_ctx(),
    )

    # All checks should have failed gracefully
    assert findings == []


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_severity_override() -> None:
    """Custom severity map overrides default for J4."""
    model = MockModel({
        CausalDirectionResult: CausalDirectionResult(proven=True),
        SignalMisreadResult: SignalMisreadResult(misread=False),
        SymptomAsCauseResult: SymptomAsCauseResult(likely_symptom=True, suspect_upstream=["db-svc"]),
    })
    sanitizer = CriticSanitizer(model, severity_map={"J4": Severity.INFO})

    findings = await sanitizer.check(
        "pre_finalize",
        _make_hypothesis_store_with_confirmed(),
        _make_profile_store_with_upstream(),
        InvestigationTracker(),
        _make_ctx(),
    )

    j4_findings = [f for f in findings if f.code == "J4"]
    assert len(j4_findings) == 1
    assert j4_findings[0].severity == Severity.INFO


@pytest.mark.asyncio
async def test_disabled_check_skipped() -> None:
    """Disabled check C3 produces no findings even when it would fire."""
    model = MockModel({
        CausalDirectionResult: CausalDirectionResult(proven=False, missing_evidence="would fire"),
        SignalMisreadResult: SignalMisreadResult(misread=False),
        SymptomAsCauseResult: SymptomAsCauseResult(likely_symptom=False),
    })
    sanitizer = CriticSanitizer(model, disabled={"C3"})

    findings = await sanitizer.check(
        "pre_finalize",
        _make_hypothesis_store_with_confirmed(),
        _make_profile_store_with_upstream(),
        _make_tracker_with_deep_analyze(),
        _make_ctx(),
    )

    c3_findings = [f for f in findings if f.code == "C3"]
    assert len(c3_findings) == 0
