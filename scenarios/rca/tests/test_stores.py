from __future__ import annotations

from agentm_rca.stores import HypothesisStore, ServiceProfileStore


def test_hypothesis_store_tracks_evidence_and_confirmed_root_cause() -> None:
    store = HypothesisStore()

    formed = store.update("H1", "database overload")
    investigating = store.update(
        "H1",
        "database overload",
        status="investigating",
        evidence_summary="p99 increased after failover",
    )
    confirmed = store.update(
        "H1",
        "database overload",
        status="confirmed",
        evidence_summary="error rate dropped after limiting traffic",
    )

    assert formed.status == "formed"
    assert investigating.evidence == ("p99 increased after failover",)
    assert confirmed.evidence == (
        "p99 increased after failover",
        "error rate dropped after limiting traffic",
    )
    assert store.confirmed_id == "H1"
    assert "Confirmed Root Cause: H1" in store.format_for_llm()

    assert store.remove("H1") is True
    assert store.confirmed_id is None
    assert store.get("H1") is None


def test_service_profile_store_merges_topology_and_observations() -> None:
    store = ServiceProfileStore()

    first = store.update(
        "payments",
        agent_id="scout-1",
        task_type="scout",
        is_anomalous=True,
        anomaly_summary="p99 latency spiked",
        upstream_services=["frontend"],
        downstream_services=["db"],
        data_sources_queried=["metrics"],
        key_observation="latency rose before retries",
        related_hypothesis_id="H1",
    )
    merged = store.update(
        "payments",
        agent_id="verify-1",
        task_type="verify",
        upstream_services=["frontend", "edge"],
        downstream_services=["db", "queue"],
        data_sources_queried=["traces"],
        key_observation="queue spans stayed healthy",
    )

    assert first.is_anomalous is True
    assert merged.upstream_services == ("frontend", "edge")
    assert merged.downstream_services == ("db", "queue")
    assert merged.data_sources_queried == ("metrics", "traces")
    assert len(merged.observations) == 2
    assert store.query(anomalous_only=True)[0].service_name == "payments"
    formatted = store.format_profile("payments")
    assert "ANOMALOUS: p99 latency spiked" in formatted
    assert "[verify-1/verify] queue spans stayed healthy" in formatted
