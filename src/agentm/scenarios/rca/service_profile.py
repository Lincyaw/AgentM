"""Service Profile — shared cross-agent service knowledge store.

Provides a thread-safe, run-scoped in-memory store that both orchestrator
and worker agents can read/write via closure injection.  Each service gets
a frozen ``ServiceProfile`` built from merge-style updates (topology union,
observations append, anomaly upgrade).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone


@dataclass(frozen=True)
class ServiceObservation:
    """A single factual observation recorded by an agent."""

    source_agent_id: str
    source_task_type: str  # "scout" | "deep_analyze" | "verify"
    timestamp: str
    observation: str  # concise factual statement
    data_sources_used: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServiceProfile:
    """Accumulated knowledge about a single service."""

    service_name: str
    first_seen_at: str
    is_anomalous: bool = False
    anomaly_summary: str = ""
    upstream_services: tuple[str, ...] = ()
    downstream_services: tuple[str, ...] = ()
    observations: tuple[ServiceObservation, ...] = ()
    data_sources_queried: tuple[str, ...] = ()
    related_hypothesis_ids: tuple[str, ...] = ()


def _unique_tuple(existing: tuple[str, ...], new: list[str] | None) -> tuple[str, ...]:
    """Return a tuple that is the union of *existing* and *new*, preserving order."""
    if not new:
        return existing
    seen = set(existing)
    merged = list(existing)
    for item in new:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return tuple(merged)


class ServiceProfileStore:
    """Thread-safe, run-scoped shared service profile store.

    Accessible by both orchestrator and all workers via closure injection.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ServiceProfile] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update(
        self,
        service_name: str,
        *,
        agent_id: str = "",
        task_type: str = "scout",
        is_anomalous: bool = False,
        anomaly_summary: str = "",
        upstream_services: list[str] | None = None,
        downstream_services: list[str] | None = None,
        data_sources_queried: list[str] | None = None,
        key_observation: str = "",
        related_hypothesis_id: str | None = None,
    ) -> ServiceProfile:
        """Merge-update a service profile.

        * Topology fields (upstream/downstream) are merged via set union.
        * ``observations`` are appended.
        * ``is_anomalous`` can only upgrade (False → True), never downgrade.
        * ``anomaly_summary`` is overwritten if a new non-empty value is given.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            existing = self._profiles.get(service_name)

            if existing is None:
                obs: tuple[ServiceObservation, ...] = ()
                if key_observation:
                    obs = (
                        ServiceObservation(
                            source_agent_id=agent_id,
                            source_task_type=task_type,
                            timestamp=now,
                            observation=key_observation,
                            data_sources_used=tuple(data_sources_queried or []),
                        ),
                    )
                profile = ServiceProfile(
                    service_name=service_name,
                    first_seen_at=now,
                    is_anomalous=is_anomalous,
                    anomaly_summary=anomaly_summary,
                    upstream_services=tuple(upstream_services or []),
                    downstream_services=tuple(downstream_services or []),
                    observations=obs,
                    data_sources_queried=tuple(data_sources_queried or []),
                    related_hypothesis_ids=(
                        (related_hypothesis_id,) if related_hypothesis_id else ()
                    ),
                )
            else:
                obs = existing.observations
                if key_observation:
                    obs = (
                        *existing.observations,
                        ServiceObservation(
                            source_agent_id=agent_id,
                            source_task_type=task_type,
                            timestamp=now,
                            observation=key_observation,
                            data_sources_used=tuple(data_sources_queried or []),
                        ),
                    )
                profile = replace(
                    existing,
                    is_anomalous=existing.is_anomalous or is_anomalous,
                    anomaly_summary=anomaly_summary if anomaly_summary else existing.anomaly_summary,
                    upstream_services=_unique_tuple(existing.upstream_services, upstream_services),
                    downstream_services=_unique_tuple(existing.downstream_services, downstream_services),
                    observations=obs,
                    data_sources_queried=_unique_tuple(
                        existing.data_sources_queried, data_sources_queried
                    ),
                    related_hypothesis_ids=_unique_tuple(
                        existing.related_hypothesis_ids,
                        [related_hypothesis_id] if related_hypothesis_id else None,
                    ),
                )

            self._profiles[service_name] = profile
            return profile

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, service_name: str) -> ServiceProfile | None:
        """Return the profile for *service_name*, or ``None``."""
        with self._lock:
            return self._profiles.get(service_name)

    def get_all(self) -> dict[str, ServiceProfile]:
        """Return a snapshot of all profiles."""
        with self._lock:
            return dict(self._profiles)

    def query(
        self,
        *,
        anomalous_only: bool = False,
        related_to: str | None = None,
    ) -> list[ServiceProfile]:
        """Filter profiles by anomaly status or topology relationship."""
        with self._lock:
            profiles = list(self._profiles.values())

        if anomalous_only:
            profiles = [p for p in profiles if p.is_anomalous]
        if related_to:
            profiles = [
                p
                for p in profiles
                if related_to in p.upstream_services
                or related_to in p.downstream_services
            ]
        return profiles

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_llm(self) -> str:
        """Format all profiles for inclusion in the orchestrator system prompt.

        Groups profiles into anomalous and healthy sections.
        """
        with self._lock:
            profiles = list(self._profiles.values())

        if not profiles:
            return ""

        anomalous = [p for p in profiles if p.is_anomalous]
        healthy = [p for p in profiles if not p.is_anomalous]

        lines: list[str] = ["## Service Profiles"]

        if anomalous:
            lines.append("")
            lines.append("### Anomalous Services")
            for p in sorted(anomalous, key=lambda x: x.service_name):
                lines.append(self._format_single(p))

        if healthy:
            lines.append("")
            lines.append("### Healthy Services")
            for p in sorted(healthy, key=lambda x: x.service_name):
                lines.append(f"- `{p.service_name}` — no anomaly detected")

        return "\n".join(lines)

    def format_profile(self, service_name: str) -> str:
        """Format a single profile for tool output."""
        with self._lock:
            profile = self._profiles.get(service_name)
        if profile is None:
            return f"No profile found for '{service_name}'"
        return self._format_single(profile)

    @staticmethod
    def _format_single(p: ServiceProfile) -> str:
        """Render one profile as a compact markdown block."""
        parts = [f"- **`{p.service_name}`**"]
        if p.is_anomalous and p.anomaly_summary:
            parts.append(f"  ANOMALOUS: {p.anomaly_summary}")
        if p.upstream_services:
            parts.append(f"  upstream: {', '.join(p.upstream_services)}")
        if p.downstream_services:
            parts.append(f"  downstream: {', '.join(p.downstream_services)}")
        if p.data_sources_queried:
            parts.append(f"  data queried: {', '.join(p.data_sources_queried)}")
        if p.observations:
            parts.append("  observations:")
            for obs in p.observations:
                parts.append(f"    - [{obs.source_agent_id}/{obs.source_task_type}] {obs.observation}")
        return "\n".join(parts)
