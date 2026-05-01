from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentm_rca.sanitizer.critic_sanitizer import (
    CausalDirectionResult,
    CriticSanitizer,
    DispatchQualityResult,
    SignalMisreadResult,
    SymptomAsCauseResult,
)
from agentm_rca.sanitizer.models import SanitizerContext, Severity
from agentm_rca.sanitizer.tracker import InvestigationTracker
from agentm_rca.stores import HypothesisStore, ServiceProfileStore


class MockStructuredModel:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def ainvoke(self, messages: Any) -> Any:
        return self._response


class MockModel:
    def __init__(self, response: Any | dict[type, Any]) -> None:
        self._response = response

    def with_structured_output(
        self,
        schema: type,
        method: str = "function_calling",
    ) -> MockStructuredModel:
        if isinstance(self._response, dict):
            return MockStructuredModel(self._response.get(schema))
        return MockStructuredModel(self._response)

    async def ainvoke(self, messages: Any) -> Any:
        del messages
        return type("R", (), {"content": ""})()


class ExplodingStructuredModel:
    async def ainvoke(self, messages: Any) -> Any:
        del messages
        raise RuntimeError("LLM exploded")


class ExplodingModel:
    def with_structured_output(
        self,
        schema: type,
        method: str = "function_calling",
    ) -> ExplodingStructuredModel:
        del schema, method
        return ExplodingStructuredModel()



def _make_ctx() -> SanitizerContext:
    return SanitizerContext(
        agent_id="test",
        step=5,
        max_steps=10,
        tool_call_count=3,
        metadata={},
    )



def _confirmed_store() -> HypothesisStore:
    store = HypothesisStore()
    store.update("H1", "Root cause is `payment-svc` timeout", status="confirmed", evidence_summary="e1")
    store.update("H1", "Root cause is `payment-svc` timeout", status="confirmed", evidence_summary="e2")
    return store



def _profile_with_upstream() -> ServiceProfileStore:
    store = ServiceProfileStore()
    store.update("payment-svc", is_anomalous=True, anomaly_summary="high latency", upstream_services=["db-svc"])
    store.update("db-svc", is_anomalous=True, anomaly_summary="connection pool exhausted")
    return store



def _store_with_rejected_hypothesis() -> HypothesisStore:
    store = HypothesisStore()
    store.update("H1", "Some hypothesis", status="investigating", evidence_summary="ev1")
    store.update("H1", "Some hypothesis", status="investigating", evidence_summary="ev2")
    store.update("H1", "Some hypothesis", status="rejected", evidence_summary="ev3")
    return store



def _tracker_with_rejection() -> InvestigationTracker:
    tracker = InvestigationTracker()
    tracker.record(
        2,
        "hypothesis_change",
        {
            "hypothesis_id": "H1",
            "new_status": "rejected",
            "reason": "conflicting metric",
        },
    )
    return tracker


@pytest.mark.asyncio
async def test_c3_blocks_finalize_when_causal_direction_not_proven() -> None:
    sanitizer = CriticSanitizer(
        MockModel(
            {
                CausalDirectionResult: CausalDirectionResult(
                    proven=False,
                    missing_evidence="no time attribution",
                ),
                SignalMisreadResult: SignalMisreadResult(misread=False),
                SymptomAsCauseResult: SymptomAsCauseResult(likely_symptom=False),
            }
        )
    )

    findings = await sanitizer.check(
        "pre_finalize",
        _confirmed_store(),
        _profile_with_upstream(),
        InvestigationTracker(),
        _make_ctx(),
    )

    c3 = [f for f in findings if f.code == "C3"]
    assert len(c3) == 1
    assert c3[0].severity == Severity.BLOCK


@pytest.mark.asyncio
async def test_c3_is_skipped_when_no_confirmed_hypothesis() -> None:
    sanitizer = CriticSanitizer(
        MockModel(CausalDirectionResult(proven=False, missing_evidence="unused"))
    )

    findings = await sanitizer.check(
        "pre_finalize",
        HypothesisStore(),
        ServiceProfileStore(),
        InvestigationTracker(),
        _make_ctx(),
    )

    assert [f for f in findings if f.code == "C3"] == []


@pytest.mark.asyncio
async def test_j1_warns_on_misread_rejection_with_sufficient_evidence() -> None:
    sanitizer = CriticSanitizer(
        MockModel(SignalMisreadResult(misread=True, reasoning="evidence was strong"))
    )

    findings = await sanitizer.check(
        "pre_finalize",
        _store_with_rejected_hypothesis(),
        ServiceProfileStore(),
        _tracker_with_rejection(),
        _make_ctx(),
    )

    j1 = [f for f in findings if f.code == "J1"]
    assert len(j1) == 1
    assert j1[0].severity == Severity.WARN


@pytest.mark.asyncio
async def test_j4_warns_when_confirmed_service_looks_like_upstream_symptom() -> None:
    sanitizer = CriticSanitizer(
        MockModel(
            {
                CausalDirectionResult: CausalDirectionResult(proven=True),
                SignalMisreadResult: SignalMisreadResult(misread=False),
                SymptomAsCauseResult: SymptomAsCauseResult(
                    likely_symptom=True,
                    suspect_upstream=["db-svc"],
                ),
            }
        )
    )

    findings = await sanitizer.check(
        "pre_finalize",
        _confirmed_store(),
        _profile_with_upstream(),
        InvestigationTracker(),
        _make_ctx(),
    )

    j4 = [f for f in findings if f.code == "J4"]
    assert len(j4) == 1
    assert j4[0].severity == Severity.WARN
    assert "db-svc" in j4[0].message


@pytest.mark.asyncio
async def test_p2_async_check_reports_dispatch_quality_issue() -> None:
    sanitizer = CriticSanitizer(
        MockModel(
            DispatchQualityResult(
                quality="poor",
                missing=["service names", "metrics"],
            )
        )
    )

    tracker = InvestigationTracker()
    tracker.record(1, "dispatch", {"instruction": "go investigate stuff"})

    await sanitizer.check_async(
        "dispatch",
        HypothesisStore(),
        ServiceProfileStore(),
        tracker,
        _make_ctx(),
    )
    await asyncio.sleep(0.05)

    findings = sanitizer.collect_async_results()
    assert len(findings) == 1
    assert findings[0].code == "P2"


@pytest.mark.asyncio
async def test_llm_failure_degrades_gracefully_without_throwing() -> None:
    sanitizer = CriticSanitizer(ExplodingModel())

    findings = await sanitizer.check(
        "pre_finalize",
        _confirmed_store(),
        _profile_with_upstream(),
        InvestigationTracker(),
        _make_ctx(),
    )

    assert findings == []


@pytest.mark.asyncio
async def test_config_controls_severity_and_disabled_checks() -> None:
    sanitizer = CriticSanitizer(
        MockModel(
            {
                CausalDirectionResult: CausalDirectionResult(
                    proven=False,
                    missing_evidence="would block",
                ),
                SignalMisreadResult: SignalMisreadResult(misread=False),
                SymptomAsCauseResult: SymptomAsCauseResult(
                    likely_symptom=True,
                    suspect_upstream=["db-svc"],
                ),
            }
        ),
        severity_map={"J4": Severity.INFO},
        disabled={"C3"},
    )

    findings = await sanitizer.check(
        "pre_finalize",
        _confirmed_store(),
        _profile_with_upstream(),
        InvestigationTracker(),
        _make_ctx(),
    )

    assert [f for f in findings if f.code == "C3"] == []
    j4 = [f for f in findings if f.code == "J4"]
    assert len(j4) == 1
    assert j4[0].severity == Severity.INFO
