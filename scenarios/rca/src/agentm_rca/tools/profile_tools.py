"""RCA service-profile tool atoms."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from agentm.core.kernel import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

from agentm_rca.stores import ServiceProfileStore

MANIFEST = ExtensionManifest(
    name="profile_tools",
    description="Register RCA service profile tool atoms.",
    registers=("tool:query_service_profile", "tool:update_service_profile"),
)

_QUERY_PARAMETERS = {
    "type": "object",
    "properties": {
        "request": {"type": "string"},
        "service_names": {"type": "string", "default": ""},
        "anomalous_only": {"type": "boolean", "default": False},
    },
    "required": ["request"],
    "additionalProperties": False,
}

_UPDATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "service_name": {"type": "string"},
        "is_anomalous": {"type": "boolean"},
        "anomaly_summary": {"type": "string", "default": ""},
        "upstream_services": {"type": "array", "items": {"type": "string"}},
        "downstream_services": {"type": "array", "items": {"type": "string"}},
        "data_sources_queried": {"type": "array", "items": {"type": "string"}},
        "key_observation": {"type": "string", "default": ""},
        "source_agent_id": {"type": "string", "default": ""},
        "source_task_type": {"type": "string", "default": "scout"},
        "related_hypothesis_id": {"type": ["string", "null"]},
    },
    "required": ["service_name", "is_anomalous"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    store = _expect_store(config)

    async def _query(args: dict[str, Any]) -> ToolResult:
        del args["request"]
        text = _do_query(
            store,
            service_names=str(args.get("service_names", "")),
            anomalous_only=bool(args.get("anomalous_only", False)),
        )
        return _ok(text)

    async def _update(args: dict[str, Any]) -> ToolResult:
        existing = store.get(str(args["service_name"]))
        existing_obs_count = len(existing.observations) if existing is not None else 0
        profile = store.update(
            str(args["service_name"]),
            agent_id=str(args.get("source_agent_id", "")),
            task_type=str(args.get("source_task_type", "scout")),
            is_anomalous=bool(args["is_anomalous"]),
            anomaly_summary=str(args.get("anomaly_summary", "")),
            upstream_services=_string_list(args.get("upstream_services")),
            downstream_services=_string_list(args.get("downstream_services")),
            data_sources_queried=_string_list(args.get("data_sources_queried")),
            key_observation=str(args.get("key_observation", "")),
            related_hypothesis_id=_maybe_str(args.get("related_hypothesis_id")),
        )
        api.session.append_entry("service_profile", asdict(profile))

        text = store.format_profile(profile.service_name)
        if existing_obs_count > 0:
            text = (
                f"NOTE: This profile already had {existing_obs_count} observation(s) from "
                "other agents. Use query_service_profile to review existing data before "
                "adding more updates.\n\n"
            ) + text
        return _ok(text, details=asdict(profile))

    api.register_tool(
        FunctionTool(
            name="query_service_profile",
            description="Query the shared RCA service profile store.",
            parameters=_QUERY_PARAMETERS,
            fn=_query,
        )
    )
    api.register_tool(
        FunctionTool(
            name="update_service_profile",
            description="Merge new RCA observations into a shared service profile.",
            parameters=_UPDATE_PARAMETERS,
            fn=_update,
        )
    )


def _expect_store(config: dict[str, Any]) -> ServiceProfileStore:
    store = config.get("store")
    if not isinstance(store, ServiceProfileStore):
        raise TypeError("profile_tools.install requires config['store']=ServiceProfileStore")
    return store


def _do_query(
    store: ServiceProfileStore,
    service_names: str,
    anomalous_only: bool,
) -> str:
    if service_names:
        names = [name.strip() for name in service_names.split(",") if name.strip()]
        if len(names) == 1:
            return store.format_profile(names[0])
        return "\n\n".join(store.format_profile(name) for name in names)
    if anomalous_only:
        profiles = store.query(anomalous_only=True)
        if not profiles:
            return "No anomalous services recorded yet."
        return "\n".join(store.format_profile(profile.service_name) for profile in profiles)
    return store.format_for_llm() or "No service profiles recorded yet."


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"Expected list[str] value, got {type(value).__name__}")
    return [str(item) for item in value]


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _ok(text: str, *, details: Any = None) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], details=details)
