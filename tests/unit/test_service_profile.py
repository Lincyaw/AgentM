"""Focused regression tests for ServiceProfileStore merge/query/format contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from agentm.scenarios.rca.service_profile import ServiceObservation, ServiceProfileStore


def test_basic_crud_get_get_all() -> None:
    store = ServiceProfileStore()
    assert store.get("missing") is None
    store.update("svc-a", agent_id="s1", is_anomalous=False)
    store.update("svc-b", agent_id="s1", is_anomalous=True)
    assert store.get("svc-a") is not None
    all_profiles = store.get_all()
    assert set(all_profiles) == {"svc-a", "svc-b"}


def test_merge_logic_unions_topology_and_appends_observations() -> None:
    store = ServiceProfileStore()
    store.update("svc-a", upstream_services=["u1"], downstream_services=["d1"], agent_id="w1", key_observation="obs-1")
    store.update("svc-a", upstream_services=["u2"], downstream_services=["d1", "d2"], agent_id="w2", key_observation="obs-2")

    profile = store.get("svc-a")
    assert profile is not None
    assert set(profile.upstream_services) == {"u1", "u2"}
    assert set(profile.downstream_services) == {"d1", "d2"}
    assert [o.observation for o in profile.observations] == ["obs-1", "obs-2"]


def test_anomaly_flag_only_upgrades_and_summary_updates_on_nonempty() -> None:
    store = ServiceProfileStore()
    store.update("svc-a", is_anomalous=True, anomaly_summary="p99 60s")
    store.update("svc-a", is_anomalous=False, anomaly_summary="")

    profile = store.get("svc-a")
    assert profile is not None
    assert profile.is_anomalous is True
    assert profile.anomaly_summary == "p99 60s"

    store.update("svc-a", anomaly_summary="p99 120s")
    assert store.get("svc-a").anomaly_summary == "p99 120s"  # type: ignore[union-attr]


def test_query_combined_filters_return_expected_subset() -> None:
    store = ServiceProfileStore()
    store.update("svc-a", is_anomalous=True, upstream_services=["x"])
    store.update("svc-b", is_anomalous=False, upstream_services=["x"])
    store.update("svc-c", is_anomalous=True, upstream_services=["y"])

    results = store.query(anomalous_only=True, related_to="x")
    assert {p.service_name for p in results} == {"svc-a"}


def test_format_for_llm_groups_anomalous_and_healthy() -> None:
    store = ServiceProfileStore()
    store.update("sick", is_anomalous=True, anomaly_summary="high error")
    store.update("healthy", is_anomalous=False)

    output = store.format_for_llm()
    assert "Anomalous Services" in output
    assert "Healthy Services" in output
    assert "sick" in output
    assert "healthy" in output


def test_concurrent_updates_do_not_lose_observations() -> None:
    store = ServiceProfileStore()
    n_workers = 8
    obs_per_worker = 4

    def worker_update(worker_id: int) -> None:
        for i in range(obs_per_worker):
            store.update(
                "shared",
                agent_id=f"w-{worker_id}",
                task_type="scout",
                key_observation=f"obs-{worker_id}-{i}",
                is_anomalous=(worker_id % 2 == 0),
            )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(worker_update, i) for i in range(n_workers)]
        for future in as_completed(futures):
            future.result()

    profile = store.get("shared")
    assert profile is not None
    assert len(profile.observations) == n_workers * obs_per_worker
    assert profile.is_anomalous is True


def test_models_are_frozen() -> None:
    store = ServiceProfileStore()
    profile = store.update("svc-a", is_anomalous=True)
    try:
        profile.is_anomalous = False  # type: ignore[misc]
        assert False, "profile should be frozen"
    except AttributeError:
        pass

    obs = ServiceObservation(
        source_agent_id="a",
        source_task_type="scout",
        timestamp="2024-01-01",
        observation="x",
    )
    try:
        obs.observation = "mutated"  # type: ignore[misc]
        assert False, "observation should be frozen"
    except AttributeError:
        pass
