"""Tests for ServiceProfileStore — shared cross-agent service knowledge store.

Each test answers "what bug does this prevent?":
- Merge logic: topology must union, observations must append, anomaly must upgrade
- Thread safety: concurrent updates must not lose data
- Formatting: LLM output must separate anomalous from healthy
- Query: filters must work correctly
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from agentm.scenarios.rca.service_profile import (
    ServiceObservation,
    ServiceProfile,
    ServiceProfileStore,
)


class TestServiceProfileStoreBasicCRUD:
    """Bug: basic get/update contract violations."""

    def test_get_nonexistent_returns_none(self) -> None:
        store = ServiceProfileStore()
        assert store.get("nonexistent") is None

    def test_update_creates_new_profile(self) -> None:
        store = ServiceProfileStore()
        profile = store.update(
            "ts-order-service",
            agent_id="scout-1",
            task_type="scout",
            is_anomalous=True,
            anomaly_summary="p99 60s vs 4s",
        )
        assert profile.service_name == "ts-order-service"
        assert profile.is_anomalous is True
        assert profile.anomaly_summary == "p99 60s vs 4s"
        assert profile.first_seen_at != ""

    def test_get_returns_updated_profile(self) -> None:
        store = ServiceProfileStore()
        store.update("svc-a", agent_id="s1", is_anomalous=False)
        result = store.get("svc-a")
        assert result is not None
        assert result.service_name == "svc-a"

    def test_get_all_returns_snapshot(self) -> None:
        store = ServiceProfileStore()
        store.update("svc-a", agent_id="s1", is_anomalous=False)
        store.update("svc-b", agent_id="s1", is_anomalous=True)
        all_profiles = store.get_all()
        assert len(all_profiles) == 2
        assert "svc-a" in all_profiles
        assert "svc-b" in all_profiles


class TestServiceProfileStoreMergeLogic:
    """Bug: subsequent updates must merge, not overwrite."""

    def test_topology_union(self) -> None:
        """Upstream/downstream must be merged via set union, not overwritten."""
        store = ServiceProfileStore()
        store.update("svc-a", upstream_services=["svc-x"], downstream_services=["svc-y"])
        store.update("svc-a", upstream_services=["svc-z"], downstream_services=["svc-y", "svc-w"])

        profile = store.get("svc-a")
        assert profile is not None
        assert set(profile.upstream_services) == {"svc-x", "svc-z"}
        assert set(profile.downstream_services) == {"svc-y", "svc-w"}

    def test_observations_append(self) -> None:
        """Observations must accumulate, not replace."""
        store = ServiceProfileStore()
        store.update("svc-a", agent_id="s1", key_observation="first finding")
        store.update("svc-a", agent_id="s2", key_observation="second finding")

        profile = store.get("svc-a")
        assert profile is not None
        assert len(profile.observations) == 2
        assert profile.observations[0].observation == "first finding"
        assert profile.observations[1].observation == "second finding"

    def test_anomaly_upgrade_only(self) -> None:
        """is_anomalous can go False→True but never True→False."""
        store = ServiceProfileStore()
        store.update("svc-a", is_anomalous=True, anomaly_summary="high latency")
        store.update("svc-a", is_anomalous=False)

        profile = store.get("svc-a")
        assert profile is not None
        assert profile.is_anomalous is True

    def test_anomaly_summary_updated_when_nonempty(self) -> None:
        """Non-empty anomaly_summary overwrites previous; empty preserves."""
        store = ServiceProfileStore()
        store.update("svc-a", is_anomalous=True, anomaly_summary="p99 60s")
        store.update("svc-a", anomaly_summary="")
        assert store.get("svc-a").anomaly_summary == "p99 60s"  # type: ignore[union-attr]

        store.update("svc-a", anomaly_summary="p99 120s")
        assert store.get("svc-a").anomaly_summary == "p99 120s"  # type: ignore[union-attr]

    def test_data_sources_queried_union(self) -> None:
        """data_sources_queried must merge without duplicates."""
        store = ServiceProfileStore()
        store.update("svc-a", data_sources_queried=["logs", "metrics"])
        store.update("svc-a", data_sources_queried=["metrics", "traces"])

        profile = store.get("svc-a")
        assert profile is not None
        assert set(profile.data_sources_queried) == {"logs", "metrics", "traces"}

    def test_related_hypothesis_ids_union(self) -> None:
        store = ServiceProfileStore()
        store.update("svc-a", related_hypothesis_id="H1")
        store.update("svc-a", related_hypothesis_id="H2")
        store.update("svc-a", related_hypothesis_id="H1")  # duplicate

        profile = store.get("svc-a")
        assert profile is not None
        assert set(profile.related_hypothesis_ids) == {"H1", "H2"}

    def test_update_without_observation_preserves_existing(self) -> None:
        """Update with empty key_observation must not add empty observation."""
        store = ServiceProfileStore()
        store.update("svc-a", agent_id="s1", key_observation="finding")
        store.update("svc-a", agent_id="s2", key_observation="")

        profile = store.get("svc-a")
        assert profile is not None
        assert len(profile.observations) == 1


class TestServiceProfileStoreQuery:
    """Bug: query filters return wrong results."""

    def test_query_anomalous_only(self) -> None:
        store = ServiceProfileStore()
        store.update("healthy", is_anomalous=False)
        store.update("sick", is_anomalous=True)
        store.update("also-sick", is_anomalous=True)

        results = store.query(anomalous_only=True)
        names = {p.service_name for p in results}
        assert names == {"sick", "also-sick"}

    def test_query_related_to(self) -> None:
        store = ServiceProfileStore()
        store.update("svc-a", upstream_services=["svc-x"])
        store.update("svc-b", downstream_services=["svc-x"])
        store.update("svc-c", upstream_services=["svc-y"])

        results = store.query(related_to="svc-x")
        names = {p.service_name for p in results}
        assert names == {"svc-a", "svc-b"}

    def test_query_combined_filters(self) -> None:
        store = ServiceProfileStore()
        store.update("svc-a", is_anomalous=True, upstream_services=["svc-x"])
        store.update("svc-b", is_anomalous=False, upstream_services=["svc-x"])
        store.update("svc-c", is_anomalous=True, upstream_services=["svc-y"])

        results = store.query(anomalous_only=True, related_to="svc-x")
        names = {p.service_name for p in results}
        assert names == {"svc-a"}


class TestServiceProfileStoreFormat:
    """Bug: LLM formatting must group anomalous vs healthy correctly."""

    def test_format_for_llm_empty_returns_empty_string(self) -> None:
        store = ServiceProfileStore()
        assert store.format_for_llm() == ""

    def test_format_for_llm_groups_anomalous_and_healthy(self) -> None:
        store = ServiceProfileStore()
        store.update("sick-svc", is_anomalous=True, anomaly_summary="high error rate")
        store.update("healthy-svc", is_anomalous=False)

        output = store.format_for_llm()
        assert "Anomalous Services" in output
        assert "Healthy Services" in output
        assert "sick-svc" in output
        assert "healthy-svc" in output

    def test_format_profile_nonexistent(self) -> None:
        store = ServiceProfileStore()
        result = store.format_profile("ghost")
        assert "No profile found" in result

    def test_format_profile_includes_observations(self) -> None:
        store = ServiceProfileStore()
        store.update(
            "svc-a",
            agent_id="scout-1",
            task_type="scout",
            key_observation="DB connection pool saturated",
        )
        output = store.format_profile("svc-a")
        assert "DB connection pool saturated" in output
        assert "scout-1" in output

    def test_format_profile_includes_topology(self) -> None:
        store = ServiceProfileStore()
        store.update(
            "svc-a",
            upstream_services=["gateway"],
            downstream_services=["db-service"],
        )
        output = store.format_profile("svc-a")
        assert "gateway" in output
        assert "db-service" in output


class TestServiceProfileStoreThreadSafety:
    """Bug: concurrent updates from multiple workers must not lose data."""

    def test_concurrent_updates_no_data_loss(self) -> None:
        store = ServiceProfileStore()
        n_workers = 10
        observations_per_worker = 5

        def worker_update(worker_id: int) -> None:
            for i in range(observations_per_worker):
                store.update(
                    "shared-service",
                    agent_id=f"worker-{worker_id}",
                    task_type="scout",
                    key_observation=f"obs-{worker_id}-{i}",
                    is_anomalous=(worker_id % 2 == 0),
                )

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker_update, i) for i in range(n_workers)]
            for f in as_completed(futures):
                f.result()  # raise any exceptions

        profile = store.get("shared-service")
        assert profile is not None
        assert len(profile.observations) == n_workers * observations_per_worker
        # At least one even-numbered worker set is_anomalous=True
        assert profile.is_anomalous is True

    def test_concurrent_topology_updates(self) -> None:
        store = ServiceProfileStore()
        n_workers = 8

        def worker_topology(worker_id: int) -> None:
            store.update(
                "hub-service",
                upstream_services=[f"upstream-{worker_id}"],
                downstream_services=[f"downstream-{worker_id}"],
            )

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker_topology, i) for i in range(n_workers)]
            for f in as_completed(futures):
                f.result()

        profile = store.get("hub-service")
        assert profile is not None
        assert len(profile.upstream_services) == n_workers
        assert len(profile.downstream_services) == n_workers


class TestServiceProfileImmutability:
    """Bug: frozen dataclasses must not be mutated after retrieval."""

    def test_service_profile_is_frozen(self) -> None:
        store = ServiceProfileStore()
        profile = store.update("svc-a", is_anomalous=True)
        try:
            profile.is_anomalous = False  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass

    def test_service_observation_is_frozen(self) -> None:
        obs = ServiceObservation(
            source_agent_id="s1",
            source_task_type="scout",
            timestamp="2024-01-01",
            observation="test",
        )
        try:
            obs.observation = "mutated"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass
