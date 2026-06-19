"""Tests must import modules through normal package paths."""

from __future__ import annotations

from pathlib import Path


_FORBIDDEN_SNIPPETS = (
    "importlib" + ".util",
    "spec_from" + "_file_location",
    "module_from" + "_spec",
    ".exec" + "_module(",
)


def test_tests_do_not_use_path_based_import_helpers() -> None:
    tests_root = Path(__file__).resolve().parents[1]
    this_file = Path(__file__).resolve()
    offenders: list[str] = []

    for path in sorted(tests_root.rglob("*.py")):
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8")
        hits = [snippet for snippet in _FORBIDDEN_SNIPPETS if snippet in text]
        if hits:
            rel = path.relative_to(tests_root.parent)
            offenders.append(f"{rel}: {', '.join(hits)}")

    assert offenders == [], "\n".join(offenders)
