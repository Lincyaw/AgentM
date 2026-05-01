"""Service Profile -- shared cross-agent service knowledge store."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone

from ._threadsafe import ThreadSafeStore


@dataclass(frozen=True)
class ServiceObservation:
    """A single factual observation recorded by an agent."""

    source_agent_id: str
    source_task_type: str
    timestamp: str
    observation: str
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


class ServiceProfileStore(ThreadSafeStore[str, ServiceProfile]):
    """Thread-safe, run-scoped shared service profile store."""

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
        """Merge-update a service profile."""
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            existing = self._data.get(service_name)

            if existing is None:
                observations: tuple[ServiceObservation, ...] = ()
                if key_observation:
                    observations = (
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
                    observations=observations,
                    data_sources_queried=tuple(data_sources_queried or []),
                    related_hypothesis_ids=(
                        (related_hypothesis_id,) if related_hypothesis_id else ()
                    ),
                )
            else:
                observations = existing.observations
                if key_observation:
                    observations = (
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
                    anomaly_summary=(
                        anomaly_summary if anomaly_summary else existing.anomaly_summary
                    ),
                    upstream_services=_unique_tuple(
                        existing.upstream_services, upstream_services
                    ),
                    downstream_services=_unique_tuple(
                        existing.downstream_services, downstream_services
                    ),
                    observations=observations,
                    data_sources_queried=_unique_tuple(
                        existing.data_sources_queried, data_sources_queried
                    ),
                    related_hypothesis_ids=_unique_tuple(
                        existing.related_hypothesis_ids,
                        [related_hypothesis_id] if related_hypothesis_id else None,
                    ),
                )

            self._data[service_name] = profile
            return profile

    def get(self, service_name: str) -> ServiceProfile | None:
        return super().get(service_name)

    def get_all(self) -> dict[str, ServiceProfile]:
        return super().get_all()

    def query(
        self,
        *,
        anomalous_only: bool = False,
        related_to: str | None = None,
    ) -> list[ServiceProfile]:
        """Filter profiles by anomaly status or topology relationship."""
        with self._lock:
            profiles = list(self._data.values())

        if anomalous_only:
            profiles = [profile for profile in profiles if profile.is_anomalous]
        if related_to:
            profiles = [
                profile
                for profile in profiles
                if related_to in profile.upstream_services
                or related_to in profile.downstream_services
            ]
        return profiles

    def format_for_llm(self) -> str:
        """Format all profiles for inclusion in the RCA context message."""
        with self._lock:
            profiles = list(self._data.values())

        if not profiles:
            return ""

        anomalous = [profile for profile in profiles if profile.is_anomalous]
        healthy = [profile for profile in profiles if not profile.is_anomalous]

        lines: list[str] = ["## Service Profiles"]

        if anomalous:
            lines.append("")
            lines.append("### Anomalous Services")
            for profile in sorted(anomalous, key=lambda item: item.service_name):
                lines.append(self._format_single(profile))

        if healthy:
            lines.append("")
            lines.append("### Healthy Services")
            for profile in sorted(healthy, key=lambda item: item.service_name):
                lines.append(f"- `{profile.service_name}` -- no anomaly detected")

        return "\n".join(lines)

    def format_profile(self, service_name: str) -> str:
        """Format a single profile for tool output."""
        with self._lock:
            profile = self._data.get(service_name)
        if profile is None:
            return f"No profile found for '{service_name}'"
        return self._format_single(profile)

    @staticmethod
    def _format_single(profile: ServiceProfile) -> str:
        parts = [f"- **`{profile.service_name}`**"]
        if profile.is_anomalous and profile.anomaly_summary:
            parts.append(f"  ANOMALOUS: {profile.anomaly_summary}")
        if profile.upstream_services:
            parts.append(f"  upstream: {', '.join(profile.upstream_services)}")
        if profile.downstream_services:
            parts.append(f"  downstream: {', '.join(profile.downstream_services)}")
        if profile.data_sources_queried:
            parts.append(f"  data queried: {', '.join(profile.data_sources_queried)}")
        if profile.observations:
            parts.append("  observations:")
            for observation in profile.observations:
                parts.append(
                    "    - "
                    f"[{observation.source_agent_id}/{observation.source_task_type}] "
                    f"{observation.observation}"
                )
        return "\n".join(parts)
