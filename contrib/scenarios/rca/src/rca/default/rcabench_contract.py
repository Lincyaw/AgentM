"""Shared rcabench-platform agent contract prompt atom."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from agentm.core.abi import (
    BeforeAgentStartEvent,
    DiagnosticEvent,
    ExtensionAPI,
    SessionReadyEvent,
)
from agentm.extensions import ExtensionManifest


class _RcabenchContractConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


MANIFEST = ExtensionManifest(
    name="rcabench_contract",
    description="Inject the official rcabench-platform agent contract into RCA prompts.",
    registers=(
        f"event:{SessionReadyEvent.CHANNEL}",
        f"event:{BeforeAgentStartEvent.CHANNEL}",
        f"event:{DiagnosticEvent.CHANNEL}",
    ),
    config_schema=_RcabenchContractConfig,
    tier=2,
)

async def _load_agent_contract_block(api: ExtensionAPI) -> str:
    try:
        from rcabench_platform.v3.sdk.evaluation.v2 import (  # type: ignore[import-not-found]
            get_agent_contract_prompt,
        )
    except ImportError as exc:
        reason = str(exc) or exc.__class__.__name__
        await api.events.emit(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="warning",
                source="rcabench_contract",
                message=f"rcabench-platform contract unavailable: {reason}",
            ),
        )
        return f'<contract status="unavailable" reason="{_xml_attr(reason)}" />'

    body = str(get_agent_contract_prompt()).strip()
    if not body:
        return '<contract status="unavailable" reason="empty vendor prompt" />'
    return (
        "<agent_contract>\n"
        "The shape and vocabulary below are enforced by `submit_final_report`.\n"
        "Match `service` and `fault_kind` exactly; evidence SQL must be "
        "runnable on the case dir.\n\n"
        f"{body}\n"
        "</agent_contract>"
    )

def _xml_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

async def install(api: ExtensionAPI, _config: _RcabenchContractConfig) -> None:
    cached_contract = ""

    async def _load(_event: SessionReadyEvent) -> None:
        nonlocal cached_contract
        cached_contract = await _load_agent_contract_block(api)

    def _inject_prompt(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_contract:
            return None
        existing = event.system or ""
        merged = f"{cached_contract}\n\n{existing}" if existing else cached_contract
        event.system = merged
        return {"system": merged}

    api.on(SessionReadyEvent.CHANNEL, _load)
    api.on(BeforeAgentStartEvent.CHANNEL, _inject_prompt)
