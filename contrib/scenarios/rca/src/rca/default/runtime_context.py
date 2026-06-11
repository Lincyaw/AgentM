"""Scenario-local atom: append runtime context (case directory) to the system prompt.

The investigator prompt under ``prompts/investigator.md`` is a verbatim
ThinkDepthAI artifact and intentionally avoids embedding host-specific
paths. The case directory is conveyed at runtime via
``AGENTM_RCA_DATA_DIR`` and the data tools' empty-arg fallback
(``list_tables_in_directory("")`` resolves to the env value). That
contract is invisible to the model unless we surface it.

This atom appends a short ``## Runtime context`` block to the system
prompt at ``BeforeAgentStartEvent`` so the agent knows (a) where the
case files live and (b) that bare filenames / empty-string ``directory``
arguments resolve against that path. Mounted only by manifest variants
that explicitly opt into it (e.g. ``manifest.harness.yaml``); the
parity-locked ``rca:baseline`` does not load this atom.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

from agentm.extensions import ExtensionManifest
from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI

class RuntimeContextConfig(BaseModel):
    # Override the env-derived path. Useful in tests or when the
    # case dir is not driven by AGENTM_RCA_DATA_DIR.
    data_dir: str | None = None

MANIFEST = ExtensionManifest(
    name="runtime_context",
    description=(
        "Append a runtime-context block (case directory path + bare-name "
        "resolution rule) to the system prompt."
    ),
    registers=(f"event:{BeforeAgentStartEvent.CHANNEL}",),
    config_schema=RuntimeContextConfig,
    # tier defaults to 1: out-of-tree atom; see prompt_loader.py for the rationale.
)

_BLOCK_HEADER = "## Runtime context"

def _build_block(data_dir: str) -> str:
    return (
        f"{_BLOCK_HEADER}\n"
        f"- Case directory: `{data_dir}`\n"
        "- `list_tables_in_directory` and `query_parquet_files` resolve bare "
        "filenames (e.g. `abnormal_traces.parquet`) against this directory. "
        "Pass an empty string as `directory` to `list_tables_in_directory` "
        "to list this case directory; do not invent sibling paths like "
        "`data/` or `/data`."
    )

def install(api: ExtensionAPI, config: RuntimeContextConfig) -> None:
    data_dir_override = config.data_dir.strip() if config.data_dir else None

    def _inject(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        data_dir = data_dir_override or os.environ.get("AGENTM_RCA_DATA_DIR", "").strip()
        if not data_dir:
            # No case dir bound — stay silent rather than print a placeholder.
            return None
        block = _build_block(data_dir)
        existing = event.system or ""
        # Idempotent: don't double-append if some upstream handler already
        # produced a system prompt that contains our header.
        if _BLOCK_HEADER in existing:
            return None
        merged = f"{existing}\n\n{block}" if existing else block
        event.system = merged
        return {"system": merged}

    api.on(BeforeAgentStartEvent.CHANNEL, _inject)
