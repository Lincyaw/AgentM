"""ResourceWriter constitution classification must survive path indirection."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm.core._internal.catalog.manifest import override_manifest_path
from agentm.core.runtime.resource_writer import LocalResourceWriter


@pytest.mark.asyncio
async def test_symlink_cannot_bypass_constitution_mutation_guards(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "core-manifest.yaml"
    manifest.write_text(
        """\
version: 1
constitution:
  paths:
    - protected/**
managed:
  globs: []
extension_api:
  current: 1
  deprecation:
    grace: 1
reload:
  tier_2_atoms: []
""",
        encoding="utf-8",
    )
    protected = tmp_path / "protected"
    protected.mkdir()
    targets = {
        "write-link": protected / "write.txt",
        "replace-link": protected / "replace.txt",
        "delete-link": protected / "delete.txt",
    }
    for target in targets.values():
        target.write_text("original", encoding="utf-8")
    for link, target in targets.items():
        (tmp_path / link).symlink_to(target)

    writer = LocalResourceWriter(cwd=str(tmp_path))
    with override_manifest_path(manifest):
        write_result = await writer.write(
            "write-link",
            b"mutated",
            rationale="boundary regression",
        )
        replace_result = await writer.replace(
            "replace-link",
            b"original",
            b"mutated",
            rationale="boundary regression",
        )
        delete_result = await writer.delete(
            "delete-link",
            rationale="boundary regression",
        )

    for result in (write_result, replace_result, delete_result):
        assert result.path_class == "constitution"
        assert result.error is not None
    for target in targets.values():
        assert target.read_text(encoding="utf-8") == "original"


@pytest.mark.asyncio
async def test_batch_discards_queued_mutations_when_body_fails(
    tmp_path: Path,
) -> None:
    writer = LocalResourceWriter(cwd=str(tmp_path))

    with pytest.raises(RuntimeError, match="abort"):
        async with writer.batch(rationale="transactional batch") as batch:
            await batch.write("should-not-exist.txt", b"mutated")
            raise RuntimeError("abort")

    assert not (tmp_path / "should-not-exist.txt").exists()
