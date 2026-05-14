"""Stub-mode tests for the 4 judge atoms.

For each judge (satisfied / coverage / independence / falsified_genuinely):

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

from agentm_rca_hfsm.atoms import (
    judge_coverage,
    judge_falsified_genuinely,
    judge_independence,
    judge_satisfied,
)
from agentm_rca_hfsm.judges import JudgeContext


class _ServiceOnlyAPI:
    """Minimal ExtensionAPI shim — just the ``set_service`` surface."""

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def set_service(self, name: str, obj: Any) -> None:
        self._services[name] = obj

    def get_service(self, name: str) -> Any:
        return self._services.get(name)


_JUDGES = [
    (judge_satisfied, "rca.judge.satisfied"),
    (judge_coverage, "rca.judge.coverage"),
    (judge_independence, "rca.judge.independence"),
    (judge_falsified_genuinely, "rca.judge.falsified_genuinely"),
]


def _ctx(i: int) -> JudgeContext:
    return JudgeContext(graph_slice={"i": i}, operands={})


def _scripted() -> list[dict[str, str]]:
    return [
        {"verdict": "v1", "reason": "r1", "confidence": "high"},
        {"verdict": "v2", "reason": "r2", "confidence": "medium"},
        {"verdict": "v3", "reason": "r3", "confidence": "low"},
    ]


@pytest.mark.parametrize(("module", "service_name"), _JUDGES)
def test_stub_returns_scripted_in_order(module: Any, service_name: str) -> None:
    api = _ServiceOnlyAPI()
    module.install(api, {"mode": "stub", "scripted": _scripted()})
    impl = api.get_service(service_name)

    out = [impl.judge(_ctx(i)).verdict for i in range(3)]

    assert out == ["v1", "v2", "v3"]


@pytest.mark.parametrize(("module", "service_name"), _JUDGES)
def test_stub_raises_on_exhaustion(module: Any, service_name: str) -> None:
    api = _ServiceOnlyAPI()
    module.install(api, {"mode": "stub", "scripted": _scripted()})
    impl = api.get_service(service_name)

    for i in range(3):
        impl.judge(_ctx(i))

    with pytest.raises(IndexError):
        impl.judge(_ctx(99))


@pytest.mark.parametrize(("module", "service_name"), _JUDGES)
def test_stub_caches_identical_context(module: Any, service_name: str) -> None:
    api = _ServiceOnlyAPI()
    module.install(api, {"mode": "stub", "scripted": _scripted()})
    impl = api.get_service(service_name)

    ctx = _ctx(0)
    first = impl.judge(ctx)
    second = impl.judge(ctx)
    # Cache hit — no scripted entry consumed; the third call with a NEW
    # context should still receive v2 (the second scripted entry).
    third = impl.judge(_ctx(1))

    assert first.verdict == "v1"
    assert second.verdict == "v1"
    assert third.verdict == "v2"
