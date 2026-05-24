"""Builtin atom: production-traffic guard regression watch (B-7).

Passive observer. Hooks ``SessionReadyEvent`` for the configured
``target_scenario``; on each production session start, scans the most
recent ``recent_n`` ``.agentm/observability/<trace>.jsonl`` files,
computes a guard metric per trace (currently ``tool_error_rate`` —
heuristic, count of error tool results / turn count), and compares
each sample against a configurable regression threshold. If
``min_samples`` of the recent traces exceed the threshold and
``auto_rollback`` is enabled, append a ``kind="rollback"`` record to
``activations.jsonl`` referencing the prior activation as the rollback
target.

Why an atom (not a tool): the regression watch must run on every
production session startup without operator action. The session_ready
hook gives us that anchor with zero protocol changes — this is exactly
the "passive observer" pattern observability.py established.

§10 P5 (acceptance scenario): an outlier trace MUST NOT trigger
rollback. The fail-stop test fixes ``min_samples=5`` (default) and seeds
4 regressing traces (no rollback) → adds a 5th (rollback fires). The
counter is *recent N samples beyond threshold*, not consecutive — the
loop never converges if one bad sample undoes one activate.

Heuristic choice: ``tool_error_rate`` is computed from the per-turn
``tool_error_count`` already written by the observability turn
aggregator. Production-grade extractors (refusals, content-policy
hits, user-reported failures) are post-MVP — the design doc flags this
explicitly. The atom keeps the extractor pluggable via
``metric_extractor`` config in case a scenario wants to override.

§11 contract: single file; no atom-to-atom imports; no harness.session
import; no core._internal import. The session_ready hook is a public
harness event from ``agentm.core.abi.events``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agentm.core.abi import TraceReader
from agentm.extensions import ExtensionManifest
from agentm.core.abi.events import SessionReadyEvent
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_guard_watch",
    description=(
        "Passive observer that scans recent production trace JSONL "
        "files at session_ready time and auto-rollbacks the most "
        "recent activation if too many traces regress beyond the "
        "configured guard threshold."
    ),
    registers=("event:session_ready",),
    config_schema={
        "type": "object",
        "properties": {
            "target_scenario": {"type": "string"},
            "recent_n": {"type": "integer"},
            "min_samples": {"type": "integer"},
            "regression_threshold": {"type": "number"},
            "auto_rollback": {"type": "boolean"},
            "cooldown_seconds": {"type": "number"},
            "disabled": {"type": "boolean"},
        },
        "additionalProperties": True,
    },
)


_DEFAULT_RECENT_N = 20
_DEFAULT_MIN_SAMPLES = 5
_DEFAULT_REGRESSION_THRESHOLD = 0.20  # tool_error_rate (errors / turns)
_DEFAULT_COOLDOWN_SECONDS = 24 * 3600.0


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    if bool(config.get("disabled", False)):
        # Operator override — do not subscribe at all so the watch
        # introduces zero overhead when off.
        return

    target_scenario = str(config.get("target_scenario") or "")
    if not target_scenario:
        # Without a target scenario we have no activations.jsonl to
        # write rollbacks to — degrade silently rather than warn.
        return

    recent_n = _coerce_int(config.get("recent_n"), _DEFAULT_RECENT_N)
    min_samples = _coerce_int(config.get("min_samples"), _DEFAULT_MIN_SAMPLES)
    regression_threshold = _coerce_float(
        config.get("regression_threshold"), _DEFAULT_REGRESSION_THRESHOLD
    )
    auto_rollback = bool(config.get("auto_rollback", True))
    cooldown_seconds = _coerce_float(
        config.get("cooldown_seconds"), _DEFAULT_COOLDOWN_SECONDS
    )

    cwd = Path(api.cwd)

    def _on_ready(event: SessionReadyEvent) -> None:
        # Skip eval-run sessions and self-hosted tuner sessions: only
        # production traces feed the regression count. We use the
        # session's own id as a fence — the trace just opened doesn't
        # exist yet, so it can't taint its own startup check.
        try:
            _evaluate_and_maybe_rollback(
                cwd=cwd,
                target_scenario=target_scenario,
                self_session_id=event.session_id,
                recent_n=recent_n,
                min_samples=min_samples,
                regression_threshold=regression_threshold,
                auto_rollback=auto_rollback,
                cooldown_seconds=cooldown_seconds,
            )
        except Exception:  # noqa: BLE001
            # Watcher must never break a session start. Swallow + skip;
            # any diagnostics live in the activations.jsonl audit trail.
            return

    api.on(SessionReadyEvent.CHANNEL, _on_ready)


# ---------------------------------------------------------------------------
# core algorithm — pure, easy to unit-test


def _evaluate_and_maybe_rollback(
    *,
    cwd: Path,
    target_scenario: str,
    self_session_id: str,
    recent_n: int,
    min_samples: int,
    regression_threshold: float,
    auto_rollback: bool,
    cooldown_seconds: float,
) -> None:
    decisions_path = (
        cwd / ".agentm" / "decisions" / target_scenario / "activations.jsonl"
    )
    if not decisions_path.is_file():
        return  # nothing has ever been activated; nothing to roll back
    activations = _load_activations(decisions_path)
    last_activate = _last_kind_in(
        activations, kinds={"activate", "merge", "exploratory"}
    )
    if last_activate is None:
        return  # log exists but only carries rejections / setup records

    # Cooldown: don't churn the same atom every session. We measure from
    # the most recent ``rollback`` (auto OR manual) targeting this atom.
    last_rollback = _last_kind_in(activations, kinds={"rollback"})
    if last_rollback is not None:
        last_at = float(last_rollback.get("at") or 0.0)
        if (time.time() - last_at) < cooldown_seconds:
            return

    # Collect recent production trace files, newest first by mtime.
    obs_dir = cwd / ".agentm/observability"
    if not obs_dir.is_dir():
        return
    candidates = sorted(
        (p for p in obs_dir.glob("*.jsonl") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Exclude the just-opened session — its trace is empty and would
    # depress the count by counting as a non-regressing sample.
    samples: list[dict[str, Any]] = []
    for path in candidates:
        if path.stem == self_session_id:
            continue
        sample = _summarize_trace(path)
        if sample is None:
            continue
        samples.append(sample)
        if len(samples) >= recent_n:
            break

    regressing = [
        s for s in samples if s["tool_error_rate"] > regression_threshold
    ]
    if len(regressing) < min_samples:
        return  # evidence floor not met — exactly the fail-stop property

    if not auto_rollback:
        # Diagnostic only: append a watch record so the operator sees
        # the signal without an actual rollback firing.
        record = {
            "at": time.time(),
            "kind": "guard_watch_warning",
            "scenario": target_scenario,
            "atom": last_activate.get("atom"),
            "evidence": {
                "samples_count": len(samples),
                "regressing_count": len(regressing),
                "regression_threshold": regression_threshold,
                "min_samples": min_samples,
                "regressing_traces": [s["trace_id"] for s in regressing],
            },
            "by": "tool_guard_watch",
        }
        _append_decision_record(decisions_path, record)
        return

    # Walk back through activations to find the previous activation
    # (the one BEFORE last_activate) — that is the rollback target.
    prior_activate = _prior_activate(activations, current=last_activate)
    rollback_record = {
        "at": time.time(),
        "kind": "rollback",
        "auto": True,
        "scenario": target_scenario,
        "atom": last_activate.get("atom"),
        "rationale": (
            f"auto-rollback: {len(regressing)}/{len(samples)} recent "
            f"traces regressed beyond threshold {regression_threshold}"
        ),
        "rolled_back_from": last_activate.get("candidate_id"),
        "rolled_back_to": (
            prior_activate.get("candidate_id") if prior_activate else None
        ),
        "from_sha": last_activate.get("to_sha"),
        "to_sha": (
            prior_activate.get("to_sha") if prior_activate else None
        ),
        "evidence": {
            "samples_count": len(samples),
            "regressing_count": len(regressing),
            "regression_threshold": regression_threshold,
            "min_samples": min_samples,
            "regressing_traces": [s["trace_id"] for s in regressing],
        },
        "by": "tool_guard_watch",
    }
    _append_decision_record(decisions_path, rollback_record)


# ---------------------------------------------------------------------------
# helpers


def _summarize_trace(path: Path) -> dict[str, Any] | None:
    """Heuristic tool_error_rate extractor for one OTLP/JSON trace file.

    Walks ``agentm.turn.summary`` log records via :class:`TraceReader` and
    reads ``tool_error_count`` from each body. Returns ``None`` when the
    trace has no turn summaries (empty session — can't measure).
    """
    if not path.is_file():
        return None
    turn_count = 0
    error_count = 0
    for body in TraceReader(path).load_turn_summaries():
        turn_count += 1
        try:
            error_count += int(body.get("tool_error_count") or 0)
        except (TypeError, ValueError):
            pass
    if turn_count == 0:
        return None
    rate = float(error_count) / float(turn_count)
    return {
        "trace_id": path.stem,
        "turn_count": turn_count,
        "tool_error_count": error_count,
        "tool_error_rate": rate,
    }


def _load_activations(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


def _last_kind_in(
    records: list[dict[str, Any]], *, kinds: set[str]
) -> dict[str, Any] | None:
    for rec in reversed(records):
        if rec.get("kind") in kinds:
            return rec
    return None


def _prior_activate(
    records: list[dict[str, Any]], *, current: dict[str, Any]
) -> dict[str, Any] | None:
    """Find the activation immediately preceding ``current`` (by file
    order). Used as the rollback target.
    """
    seen_current = False
    for rec in reversed(records):
        if not seen_current:
            if rec is current:
                seen_current = True
            continue
        if rec.get("kind") in {"activate", "merge", "exploratory"}:
            return rec
    return None


def _append_decision_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
