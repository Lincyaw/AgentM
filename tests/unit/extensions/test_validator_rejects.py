"""§11 validator: synthetic bad atoms must be rejected."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from agentm.extensions.validate import (
    _check_ast_rules,
    _check_imports,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_source(tmp_path: Path, source: str) -> Path:
    p = tmp_path / "bad_atom.py"
    p.write_text(dedent(source), encoding="utf-8")
    return p


_INSTALL_PREAMBLE = """\
from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(name="test", description="test")

def install(api: ExtensionAPI, config: dict) -> None:
"""


def _ast_source(body: str) -> str:
    """Wrap *body* inside a minimal atom skeleton for AST-rule checks."""
    return _INSTALL_PREAMBLE + "".join(f"    {line}\n" for line in body.splitlines())


# ---------------------------------------------------------------------------
# Import violations (rule 11.4.5-import)
# ---------------------------------------------------------------------------


def test_rejects_runtime_import(tmp_path: Path) -> None:
    src = _write_source(tmp_path, "from agentm.core.runtime import session\n")
    issues = _check_imports("test_module", src)

    assert len(issues) > 0
    assert any("11.4.5" in i.rule for i in issues)


def test_rejects_internal_import(tmp_path: Path) -> None:
    src = _write_source(tmp_path, "from agentm.core._internal import catalog\n")
    issues = _check_imports("test_module", src)

    assert len(issues) > 0
    assert any("11.4.5" in i.rule for i in issues)


def test_rejects_atom_to_atom_import(tmp_path: Path) -> None:
    src = _write_source(
        tmp_path, "from agentm.extensions.builtin.operations import MANIFEST\n"
    )
    issues = _check_imports("test_module", src)

    assert len(issues) > 0
    assert any("11.4.5" in i.rule for i in issues)


def test_rejects_langchain_import(tmp_path: Path) -> None:
    src = _write_source(tmp_path, "import langchain\n")
    issues = _check_imports("test_module", src)

    assert len(issues) > 0
    assert any("11.4.5" in i.rule for i in issues)


def test_accepts_allowed_import(tmp_path: Path) -> None:
    src = _write_source(tmp_path, "from agentm.core.abi import ExtensionAPI\n")
    issues = _check_imports("test_module", src)

    assert len(issues) == 0


# ---------------------------------------------------------------------------
# AST hygiene violations
# ---------------------------------------------------------------------------


def test_rejects_private_api_reflection(tmp_path: Path) -> None:
    src = _write_source(tmp_path, _ast_source("getattr(api, '_runtime')"))
    issues = _check_ast_rules("test_module", src)

    assert len(issues) > 0
    assert any("D1" in i.rule for i in issues)


def test_rejects_dynamic_fstring_import(tmp_path: Path) -> None:
    src = _write_source(
        tmp_path, _ast_source('import_module(f"agentm.{name}")')
    )
    issues = _check_ast_rules("test_module", src)

    assert len(issues) > 0
    assert any("D5" in i.rule for i in issues)


def test_rejects_service_isinstance_downcast(tmp_path: Path) -> None:
    src = _write_source(
        tmp_path, _ast_source("isinstance(svc, BashOperations)")
    )
    issues = _check_ast_rules("test_module", src)

    assert len(issues) > 0
    assert any("D6" in i.rule for i in issues)


def test_rejects_mutable_global(tmp_path: Path) -> None:
    source = dedent("""\
        from agentm.core.abi import ExtensionAPI
        from agentm.extensions import ExtensionManifest

        MANIFEST = ExtensionManifest(name="test", description="test")

        CACHE = {}

        def install(api: ExtensionAPI, config: dict) -> None:
            pass
    """)
    src = _write_source(tmp_path, source)
    issues = _check_ast_rules("test_module", src)

    assert len(issues) > 0
    assert any("D3" in i.rule for i in issues)
