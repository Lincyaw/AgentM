"""Acceptance tests for the self-modifiable MVP.

Covers the scenarios from `.claude/designs/self-modifiable-architecture.md` §9
and `.claude/designs/evolution-substrate.md` §9 that have a real runtime
contract to assert against on this branch. Phase 2 scenarios (compare /
find_best / propose_change) and structural checks duplicated by unit tests
are intentionally not represented here.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.catalog.freeze import freeze_current
from agentm.core.catalog.hashing import compute_active_set_fingerprint
from agentm.core.catalog.indexer import index_trace, rebuild_catalog
from agentm.core.catalog.manifest import reload_manifest
from agentm.core.kernel import AssistantMessage, AssistantStreamEvent, MessageEnd, Model, TextContent
from agentm.extensions.builtin import observability as observability_mod
from agentm.extensions.builtin import tool_catalog as tool_catalog_mod
from agentm.extensions.builtin import tool_read as tool_read_mod
from agentm.harness.extension import ProviderConfig
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class _StaticProvider:
    def __init__(self, text: str = "ok") -> None:
        self._text = text

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        return self._iter(
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=self._text)],
                timestamp=1.0,
                stop_reason="end_turn",
            )
        )

    async def _iter(self, msg: AssistantMessage) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(message=msg)


def _install_provider_module(name: str, provider: _StaticProvider) -> str:
    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-self-mod-mvp",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-self-mod-mvp",
                    provider="fake",
                    context_window=10_000,
                    max_output_tokens=1_000,
                ),
                name="fake-self-mod-mvp",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


async def _create_mvp_session(tmp_path: Path) -> AgentSession:
    provider_module = _install_provider_module(
        f"tests.integration._self_mod_provider_{tmp_path.name}",
        _StaticProvider("integration complete"),
    )
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.observability", {}),
                ("agentm.extensions.builtin.tool_read", {}),
                ("agentm.extensions.builtin.tool_catalog", {}),
            ],
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )


def _module_source(module: types.ModuleType) -> str:
    source_path = inspect.getsourcefile(module)
    if source_path is None:
        raise AssertionError(f"no source path for {module!r}")
    return Path(source_path).read_text(encoding="utf-8")


def _seed_atom_version(tmp_path: Path, *, module: types.ModuleType, manifest: Any) -> str:
    return freeze_current(
        manifest.name,
        _module_source(module),
        manifest,
        root=tmp_path,
    )


def _capture_metrics(root: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for metrics_path in sorted(root.rglob("metrics.jsonl")):
        entries: list[dict[str, Any]] = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            payload.pop("indexed_at", None)
            entries.append(payload)
        rows[str(metrics_path.relative_to(root))] = entries
    return rows


def _append_fingerprint_record(trace_path: Path, fingerprint: dict[str, Any]) -> None:
    record = {
        "schema": "otel/span/v0",
        "kind": "session.fingerprint",
        "trace_id": trace_path.stem,
        "span_id": "fixturefingerprint",
        "name": "session.fingerprint",
        "start_time_unix_nano": 0,
        "attributes": fingerprint,
        "status": {"code": "OK"},
    }
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


@pytest.mark.asyncio
async def test_S10_manifest_change_moves_constitution_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§9 / S10: editing core-manifest.yaml redraws what counts as constitution."""
    from agentm.core.catalog import manifest as manifest_mod

    custom_manifest = tmp_path / "core-manifest.yaml"
    custom_manifest.write_text(
        "version: 1\n"
        "constitution:\n"
        "  paths:\n"
        "    - src/agentm/core/operations.py\n"
        "    - core-manifest.yaml\n"
        "extension_api:\n"
        "  current: 1\n"
        "  semver_rules: {major: x, minor: x, patch: x}\n"
        "  deprecation:\n"
        "    grace: 1\n"
        "reload:\n"
        "  tier_2_atoms: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manifest_mod, "_MANIFEST_PATH", custom_manifest)
    reload_manifest()

    assert manifest_mod.is_constitution_path("src/agentm/core/kernel/loop.py") is False
    assert manifest_mod.is_constitution_path("core-manifest.yaml") is True
    assert manifest_mod.is_constitution_path("src/agentm/core/operations.py") is True


@pytest.mark.asyncio
async def test_E5_rebuild_is_idempotent(tmp_path: Path) -> None:
    """§9 / E5: rebuilding the catalog from observability traces is byte-stable."""
    session = await _create_mvp_session(tmp_path)
    try:
        await session.prompt("hello")
    finally:
        await session.shutdown()

    catalog_root = tmp_path / ".agentm" / "catalog"
    observability_root = tmp_path / ".agentm" / "observability"
    trace_path = next(observability_root.glob("*.jsonl"))

    tool_read_hash = _seed_atom_version(
        tmp_path, module=tool_read_mod, manifest=tool_read_mod.MANIFEST
    )
    _seed_atom_version(
        tmp_path, module=tool_catalog_mod, manifest=tool_catalog_mod.MANIFEST
    )
    _seed_atom_version(
        tmp_path, module=observability_mod, manifest=observability_mod.MANIFEST
    )
    fingerprint = compute_active_set_fingerprint(
        {"tool_read": tool_read_hash},
        scenario=None,
        core_hash=None,
    )
    _append_fingerprint_record(trace_path, fingerprint)

    first = index_trace(trace_path, root=catalog_root)
    assert first.n_atoms_attributed == 1

    before = _capture_metrics(catalog_root)
    assert before

    rebuild_catalog(root=catalog_root, observability=observability_root)
    after = _capture_metrics(catalog_root)
    assert before == after


@pytest.mark.asyncio
async def test_M1_freeze_idempotent(tmp_path: Path) -> None:
    """Freezing the same source twice produces the same version dir, no duplication."""
    first = _seed_atom_version(tmp_path, module=tool_read_mod, manifest=tool_read_mod.MANIFEST)
    second = _seed_atom_version(tmp_path, module=tool_read_mod, manifest=tool_read_mod.MANIFEST)

    version_dir = tmp_path / ".agentm" / "catalog" / "atoms" / "tool_read" / first
    assert first == second
    assert version_dir.is_dir()
    children = {child.name for child in version_dir.iterdir()}
    assert {"source.py", "manifest.yaml", "runs"} <= children


@pytest.mark.asyncio
async def test_M3_list_versions_after_first_session(tmp_path: Path) -> None:
    """catalog_list_versions surfaces a frozen atom version through the live tool API."""
    version_hash = _seed_atom_version(
        tmp_path, module=tool_read_mod, manifest=tool_read_mod.MANIFEST
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_catalog", {})],
            provider=(
                _install_provider_module(
                    f"tests.integration._tool_catalog_provider_{tmp_path.name}",
                    _StaticProvider(),
                ),
                {},
            ),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    try:
        tool = next(tool for tool in session.tools if tool.name == "catalog_list_versions")
        result = await tool.execute({"atom": "tool_read"})
    finally:
        await session.shutdown()

    assert result.is_error is False
    assert result.details == [version_hash]
