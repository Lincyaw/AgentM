"""Shared fpg ModelRCAOutput contract prompt atom."""

from __future__ import annotations

from typing import Final

from pydantic import ConfigDict

from agentm.core.abi import (
    BeforeAgentStartEvent,
    DiagnosticEvent,
    ExtensionAPI,
    SessionReadyEvent,
)
from agentm.extensions import ExtensionManifest

from rca.fpg_schema import (
    FpgOutputConfig,
    resolve_profile_path,
    vocab_for_contract,
)


class _FpgContractConfig(FpgOutputConfig):
    model_config = ConfigDict(extra="forbid")


MANIFEST = ExtensionManifest(
    name="fpg_contract",
    description="Inject the fpg ModelRCAOutput contract into RCA prompts.",
    registers=(
        f"event:{SessionReadyEvent.CHANNEL}",
        f"event:{BeforeAgentStartEvent.CHANNEL}",
        f"event:{DiagnosticEvent.CHANNEL}",
    ),
    config_schema=_FpgContractConfig,
    tier=2,
)


async def _load_agent_contract_block(
    api: ExtensionAPI, config: _FpgContractConfig
) -> str:
    try:
        profile_path = resolve_profile_path(
            config.profile_path, scenario_dir=api.scenario_dir
        )
        vocab = vocab_for_contract(profile_path)
    except Exception as exc:  # noqa: BLE001
        reason = str(exc) or exc.__class__.__name__
        await api.events.emit(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="warning",
                source="fpg_contract",
                message=f"fpg contract unavailable: {reason}",
            ),
        )
        return f'<contract status="unavailable" reason="{_xml_attr(reason)}" />'

    vocab_lines = "\n".join(f"- `{key}`: {brief}" for key, brief in vocab.items())
    return (
        "<agent_contract>\n"
        "Submit the final answer as fpg `ModelRCAOutput` via "
        "`submit_final_report`.\n\n"
        "Rules:\n"
        "- `nodes[]` are anomaly nodes. Use profile entity refs for `subject`: "
        "`svc:<service_name>` for service states, and `link:<source_service>->"
        "<target_service>` for a faulty service-to-service network path.\n"
        "- If the evidence indicates a fault between two services, represent "
        "that relationship as a `link:*` root node; do not collapse it into "
        "either endpoint service.\n"
        "- `predicate` must be one of the vocabulary values below.\n"
        "- Non-hypothesis nodes need at least one evidence item. For DuckDB, "
        "use `query.language = \"sql\"` and put the SQL in "
        "`query.statement`.\n"
        "- `edges[]` point from cause node id to effect node id.\n"
        "- `root_causes[]` must list node ids from `nodes[]`, ordered by "
        "descending confidence. Do not infer roots from missing edges.\n\n"
        "Predicate vocabulary:\n"
        f"{vocab_lines}\n\n"
        "</agent_contract>"
    )


def _xml_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


async def install(api: ExtensionAPI, config: _FpgContractConfig) -> None:
    cached_contract = ""

    async def _load(_event: SessionReadyEvent) -> None:
        nonlocal cached_contract
        cached_contract = await _load_agent_contract_block(api, config)

    def _inject_prompt(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_contract:
            return None
        existing = event.system or ""
        merged = f"{existing}\n\n{cached_contract}" if existing else cached_contract
        event.system = merged
        return {"system": merged}

    api.on(SessionReadyEvent.CHANNEL, _load)
    api.on(BeforeAgentStartEvent.CHANNEL, _inject_prompt)


__all__: Final = ["MANIFEST", "install"]
