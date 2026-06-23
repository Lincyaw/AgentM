"""Constitution boundary: is_constitution_path correctness."""

from __future__ import annotations

from pathlib import Path

from agentm.core._internal.catalog.manifest import is_constitution_path


def test_kernel_abi_is_protected() -> None:
    assert is_constitution_path("src/agentm/core/abi/tool.py") is True


def test_kernel_runtime_deeply_nested_is_protected() -> None:
    assert is_constitution_path("src/agentm/core/runtime/deep/nested/file.py") is True


def test_manifest_self_lock() -> None:
    assert is_constitution_path("core-manifest.yaml") is True


def test_catalog_data_is_protected() -> None:
    assert is_constitution_path(".agentm/catalog/atoms/ops/abc123/decisions.jsonl") is True


def test_provider_boundary_is_protected() -> None:
    assert is_constitution_path("src/agentm/ai/provider.py") is True


def test_builtin_atom_source_is_not_protected() -> None:
    assert is_constitution_path("src/agentm/extensions/builtin/operations.py") is False


def test_contrib_extension_is_not_protected() -> None:
    assert is_constitution_path("contrib/extensions/foo.py") is False


def test_toplevel_file_is_not_protected() -> None:
    assert is_constitution_path("README.md") is False


def test_absolute_path_normalization() -> None:
    repo_root = Path(__file__).parents[3]
    absolute = str(repo_root / "src/agentm/core/abi/tool.py")
    assert is_constitution_path(absolute) is True


def test_validator_module_is_protected() -> None:
    assert is_constitution_path("src/agentm/extensions/validate.py") is True
