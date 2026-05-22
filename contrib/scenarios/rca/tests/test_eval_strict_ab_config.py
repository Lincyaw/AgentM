"""Fail-stop tests for the strict-A/B fork wiring on ``AgentMAgent``.

Why these are load-bearing:

* Major 1 (review): ``_scenario_mounts_harness`` used to swallow every
  exception from ``load_scenario``. A malformed manifest then silently
  produced a non-harness verdict, which dropped the distill binding,
  which produced an unjoinable replay sidecar for the entire rollout.
  The new code propagates ``ScenarioLoadError`` for anything but the
  legitimate "manifest not found" case.

* Major 3 (review): unknown harness variants used to map to themselves
  in strict-A/B fork mode, making ``control_scenario == branch_scenario``
  and silently breaking the "immutable control prefix" semantics. The
  new code raises ``ValueError`` at ``AgentMAgent`` construction time
  for unmapped variants, unless an explicit ``control_scenario`` is
  provided.

We intentionally stop at the ``AgentMAgent.__init__`` and the
``_scenario_mounts_harness`` helper — exercising the whole rollout would
require the (pre-existing-broken) ``rcabench_platform.v3.sdk.evaluation.v2``
import in ``_execute_session``.
"""

from __future__ import annotations

import pytest

from agentm_rca.eval.agent import AgentMAgent, _scenario_mounts_harness


# ---------------------------------------------------------------------------
# Major 3 — unknown harness variant + strict-A/B must raise


def test_strict_ab_unknown_variant_without_explicit_control_raises() -> None:
    """``rca:harness`` (the async production variant) is intentionally
    not in ``_HARNESS_VARIANT_TO_CONTROL`` per ``_VARIANTS.md``. Asking
    for ``fork_audit=True`` without ``control_scenario=...`` must fail
    loudly rather than silently letting ``control == branch``."""

    with pytest.raises(ValueError, match=r"strict-A/B"):
        AgentMAgent(scenario="rca:harness", fork_audit=True)


def test_strict_ab_unknown_variant_with_explicit_control_is_ok() -> None:
    """The escape hatch must keep working: passing ``control_scenario``
    explicitly (as ``--ak control_scenario=...`` or
    ``AGENTM_FORK_CONTROL_SCENARIO``) lets callers fork from any
    scenario, including ones not in the registered map."""

    agent = AgentMAgent(
        scenario="rca:harness",
        fork_audit=True,
        control_scenario="rca:baseline",
    )
    assert agent._control_scenario == "rca:baseline"


def test_strict_ab_mapped_variant_picks_registered_control() -> None:
    """Sanity: registered variants still resolve to their registered
    control without any caller intervention."""

    agent = AgentMAgent(scenario="rca:harness.sync", fork_audit=True)
    assert agent._control_scenario == "rca:harness.sync.extractor5"


def test_non_fork_rollout_does_not_validate_control_scenario() -> None:
    """When ``fork_audit`` is not set, the control scenario is unused;
    we must not raise on an unknown variant just because someone might
    later flip the fork flag. (Otherwise every non-fork harness run
    that uses a not-yet-registered variant would be blocked.)"""

    agent = AgentMAgent(scenario="rca:harness")  # no fork_audit
    # ``_control_scenario`` falls through to the scenario itself; the
    # field exists but is never read on non-fork rollouts.
    assert agent._control_scenario == "rca:harness"


# ---------------------------------------------------------------------------
# Major 1 — _scenario_mounts_harness error policy


def test_scenario_mounts_harness_returns_false_when_manifest_missing(
    tmp_path,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scenario name that resolves to no manifest on disk must NOT
    raise — it returns ``False`` so eval drivers handle the missing
    case as 'this isn't a harness scenario'. The wire-up failure is
    logged at WARNING (verified by capsys / caplog elsewhere)."""

    # Point the loader at an empty directory so it can't find anything.
    monkeypatch.setenv("AGENTM_PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert _scenario_mounts_harness("definitely:not:a:real:scenario") is False


def test_scenario_mounts_harness_propagates_malformed_manifest(
    tmp_path,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scenario whose manifest exists but is malformed (bad YAML,
    missing required keys, etc.) MUST raise — silently treating it
    as 'not a harness scenario' was the failure mode that motivated
    the review note."""

    scenario_dir = tmp_path / "contrib" / "scenarios" / "broken"
    scenario_dir.mkdir(parents=True)
    # A scenario file that the loader can find but cannot parse:
    (scenario_dir / "manifest.yaml").write_text(
        "this is not: valid: yaml: at all: [\n", encoding="utf-8"
    )
    monkeypatch.setenv("AGENTM_PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    from agentm.extensions.loader import ScenarioLoadError

    with pytest.raises(ScenarioLoadError):
        _scenario_mounts_harness("broken")
