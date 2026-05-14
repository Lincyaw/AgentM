"""Stub-mode tests for the ``judge_investigation_genuine`` atom.

Mirrors the pattern of ``test_judges_stub.py`` for the other 4 judges:

* Install with ``mode="stub"`` and three scripted verdicts.
* Call ``.judge(ctx)`` three times with three distinct contexts; expect
  v1, v2, v3 in order.
* Call once more with a fresh context; expect ``IndexError`` (the impl
  exhausts its scripted list).
* Same context twice → the second call returns the cached verdict
  without consuming a scripted entry (the third entry is still
  available).
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm_rca_hfsm.atoms import judge_investigation_genuine
from agentm_rca_hfsm.judges import JudgeContext


class _ServiceOnlyAPI:
    """Minimal ExtensionAPI shim — just the ``set_service`` surface."""

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def set_service(self, name: str, obj: Any) -> None:
        self._services[name] = obj

    def get_service(self, name: str) -> Any:
        return self._services.get(name)


def _ctx(i: int) -> JudgeContext:
    return JudgeContext(graph_slice={"i": i}, operands={})


def _scripted() -> list[dict[str, str]]:
    return [
        {"verdict": "genuine_investigation", "reason": "r1", "confidence": "high"},
        {"verdict": "speculation", "reason": "r2", "confidence": "high"},
        {"verdict": "unclear", "reason": "r3", "confidence": "low"},
    ]


def test_stub_returns_scripted_in_order() -> None:
    api = _ServiceOnlyAPI()
    judge_investigation_genuine.install(api, {"mode": "stub", "scripted": _scripted()})
    impl = api.get_service("rca.judge.investigation_genuine")

    out = [impl.judge(_ctx(i)).verdict for i in range(3)]

    assert out == ["genuine_investigation", "speculation", "unclear"]


def test_stub_raises_on_exhaustion() -> None:
    api = _ServiceOnlyAPI()
    judge_investigation_genuine.install(api, {"mode": "stub", "scripted": _scripted()})
    impl = api.get_service("rca.judge.investigation_genuine")

    for i in range(3):
        impl.judge(_ctx(i))

    with pytest.raises(IndexError):
        impl.judge(_ctx(99))


def test_stub_caches_identical_context() -> None:
    api = _ServiceOnlyAPI()
    judge_investigation_genuine.install(api, {"mode": "stub", "scripted": _scripted()})
    impl = api.get_service("rca.judge.investigation_genuine")

    ctx = _ctx(0)
    first = impl.judge(ctx)
    second = impl.judge(ctx)
    # Cache hit — no scripted entry consumed; the third call with a NEW
    # context should still receive the SECOND scripted entry.
    third = impl.judge(_ctx(1))

    assert first.verdict == "genuine_investigation"
    assert second.verdict == "genuine_investigation"
    assert third.verdict == "speculation"
