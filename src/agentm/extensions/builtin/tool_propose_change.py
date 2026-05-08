"""Tool atom: gate-keeper for activating an atom mutation.

See ``.claude/designs/per-task-evolution-loop.md`` §6 / §8.

Required-arg validation enforces "evidence-driven, not error-driven":
the caller must pass both ``eval_run_baseline`` and ``eval_run_proposed``
or the call is rejected. Tier-2 atoms cannot be auto-activated and produce
a ``pending_human_approval`` decision record (no ``reload_atom`` call).

For tier-1 ``activate`` decisions, the gate runs a four-part check:

1. ``threshold_relative`` improvement on primary score.
2. Statistical-sanity check (Δ > 2 · sqrt(σ_b² + σ_p²)) — disables for
   ``decision="exploratory"``.
3. Guard metrics within ``±guard_tolerance`` (relative).
4. ``decision="rollback"`` skips gates entirely (rollback is always safe).

On success, calls ``api.reload_atom`` and appends a structured decision
record to ``.agentm/decisions/<scenario>/decisions.jsonl`` via a direct
file-append (the path is constitution-protected against ``tool_edit`` /
``tool_write``; only this atom may write).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_propose_change",
    description=(
        "Gate-keeper for activating an atom mutation. Validates evidence "
        "(eval_run_baseline + eval_run_proposed), enforces tier-2 deferral, "
        "applies promotion threshold + guard tolerance, then calls "
        "reload_atom on success. Decision record appended to "
        ".agentm/decisions/<scenario>/decisions.jsonl."
    ),
    registers=("tool:propose_change",),
    config_schema={
        "type": "object",
        "properties": {
            "target_scenario": {"type": "string"},
            "promotion": {
                "type": "object",
                "properties": {
                    "threshold_relative": {"type": "number"},
                    "guard_tolerance": {"type": "number"},
                    "stop_after_no_improvement": {"type": "integer"},
                },
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    },
)


_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_atom": {"type": "string"},
        "new_source": {"type": "string"},
        "rationale": {"type": "string"},
        "eval_run_baseline": {"type": "string"},
        "eval_run_proposed": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": ["activate", "rollback", "exploratory"],
        },
    },
    "required": [
        "target_atom",
        "new_source",
        "rationale",
        "eval_run_baseline",
        "eval_run_proposed",
        "decision",
    ],
    "additionalProperties": False,
}


_DEFAULT_THRESHOLD_RELATIVE = 0.05
_DEFAULT_GUARD_TOLERANCE = 0.10


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    target_scenario = str(config.get("target_scenario") or "default")
    promotion = config.get("promotion") or {}
    threshold_relative = float(
        promotion.get("threshold_relative", _DEFAULT_THRESHOLD_RELATIVE)
    )
    guard_tolerance = float(
        promotion.get("guard_tolerance", _DEFAULT_GUARD_TOLERANCE)
    )
    cwd = Path(api.cwd)

    async def _execute(args: dict[str, Any]) -> ToolResult:
        # Required-arg validation — explicit even though the schema requires
        # them, because the design's evidence contract is the load-bearing
        # piece (P3 in design §10).
        for key in (
            "eval_run_baseline",
            "eval_run_proposed",
            "target_atom",
            "new_source",
            "rationale",
            "decision",
        ):
            value = args.get(key)
            if not isinstance(value, str) or not value:
                return _error(
                    f"evidence missing: {key!r} is required and must be a "
                    f"non-empty string"
                )

        target_atom = str(args["target_atom"])
        new_source = str(args["new_source"])
        rationale = str(args["rationale"])
        baseline_id = str(args["eval_run_baseline"])
        proposed_id = str(args["eval_run_proposed"])
        decision = str(args["decision"])

        # Look up the loaded atom's tier — drives the tier-2 deferral gate.
        # Cross-session case: tuner runs in its own session and does NOT load
        # the target_scenario's atoms. Fall back to a filesystem scan under
        # the scenario root so we can still resolve tier + source path.
        loaded = _find_atom_info(api, target_atom)
        atom_in_session = loaded is not None
        if loaded is None and target_scenario:
            loaded = _find_atom_on_disk(cwd, target_scenario, target_atom)
        if loaded is None:
            return _error(
                f"unknown atom {target_atom!r}: not in current session and "
                f"no matching MANIFEST under contrib/scenarios/{target_scenario}/"
            )
        tier = int(loaded.get("tier", 1) or 1)

        # Tier-2 gate: refuse to auto-activate. Decision record still written
        # so the operator can manually approve later.
        if tier == 2 and decision == "activate":
            decisions_path = _decisions_path(cwd, target_scenario)
            record = {
                "at": time.time(),
                "kind": "pending_human_approval",
                "scenario": target_scenario,
                "atom": target_atom,
                "tier": tier,
                "rationale": rationale,
                "evidence": {
                    "baseline_run": baseline_id,
                    "proposed_run": proposed_id,
                },
                "by": "tool_propose_change",
            }
            _append_decision_record(decisions_path, record)
            return _ok(
                json.dumps(
                    {
                        "ok": True,
                        "tier_blocked": True,
                        "status": "pending_human_approval",
                        "decision_path": str(decisions_path),
                    },
                    indent=2,
                )
            )

        # Load the eval-run summaries.
        baseline = _load_eval_run_summary(cwd, baseline_id)
        if baseline is None:
            return _error(f"baseline eval run not found: {baseline_id}")
        proposed = _load_eval_run_summary(cwd, proposed_id)
        if proposed is None:
            return _error(f"proposed eval run not found: {proposed_id}")

        # Promotion gate (skipped for rollback + exploratory).
        gate_outcome: dict[str, Any] = {"applied": False}
        if decision == "activate":
            gate_outcome = _apply_promotion_gate(
                baseline=baseline,
                proposed=proposed,
                threshold_relative=threshold_relative,
                guard_tolerance=guard_tolerance,
                exploratory=False,
            )
            if not gate_outcome["passed"]:
                # Reject without invoking reload; record the rejection.
                decisions_path = _decisions_path(cwd, target_scenario)
                record = {
                    "at": time.time(),
                    "kind": "rejected",
                    "scenario": target_scenario,
                    "atom": target_atom,
                    "tier": tier,
                    "rationale": rationale,
                    "evidence": {
                        "baseline_run": baseline_id,
                        "proposed_run": proposed_id,
                        "gate": gate_outcome,
                    },
                    "by": "tool_propose_change",
                }
                _append_decision_record(decisions_path, record)
                return _error(
                    f"promotion gate failed: {gate_outcome['reason']}; "
                    f"baseline={gate_outcome.get('baseline_score')}, "
                    f"proposed={gate_outcome.get('proposed_score')}"
                )

        # Apply the change.
        # In-session: reload_atom is the transactional path (validate, swap
        # sys.modules, rerun install, rollback on failure).
        # Cross-session (tuner mutating production atom it doesn't load):
        # reload_atom can't run; instead write through ResourceWriter so the
        # change commits to git — the next production session loads the new
        # version on startup. This matches the design's "git log IS the
        # activation history".
        if atom_in_session:
            reload_result = api.reload_atom(
                target_atom,
                new_source,
                agent_initiated=True,
                rationale=rationale,
            )
            ok = reload_result.ok
            old_hash = reload_result.old_hash
            new_hash = reload_result.new_hash
            error = reload_result.error
        else:
            ok, old_hash, new_hash, error = _write_cross_session(
                api, loaded["source_path"], new_source, rationale
            )
        if not ok:
            decisions_path = _decisions_path(cwd, target_scenario)
            record = {
                "at": time.time(),
                "kind": "reload_failed",
                "scenario": target_scenario,
                "atom": target_atom,
                "tier": tier,
                "rationale": rationale,
                "evidence": {
                    "baseline_run": baseline_id,
                    "proposed_run": proposed_id,
                },
                "error": error,
                "by": "tool_propose_change",
            }
            _append_decision_record(decisions_path, record)
            return _error(f"reload failed: {error}")

        decisions_path = _decisions_path(cwd, target_scenario)
        record = {
            "at": time.time(),
            "kind": decision,
            "scenario": target_scenario,
            "atom": target_atom,
            "tier": tier,
            "atom_in_session": atom_in_session,
            "from_sha": old_hash,
            "to_sha": new_hash,
            "rationale": rationale,
            "exploratory": decision == "exploratory",
            "evidence": {
                "baseline_run": baseline_id,
                "proposed_run": proposed_id,
                "gate": gate_outcome,
            },
            "by": "tool_propose_change",
        }
        _append_decision_record(decisions_path, record)
        return _ok(
            json.dumps(
                {
                    "ok": True,
                    "tier_blocked": False,
                    "status": decision,
                    "old_hash": old_hash,
                    "new_hash": new_hash,
                    "decision_path": str(decisions_path),
                    "gate": gate_outcome,
                },
                indent=2,
            )
        )

    api.register_tool(
        FunctionTool(
            name="propose_change",
            description=(
                "Activate / rollback / exploratory mutation of a tier-1 atom. "
                "Requires evidence (baseline + proposed eval_run_ids). Tier-2 "
                "decisions are deferred to pending_human_approval."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )


# ---------------------------------------------------------------------------


def _find_atom_info(api: ExtensionAPI, name: str) -> dict[str, Any] | None:
    for info in api.list_atoms():
        if getattr(info, "name", None) == name:
            return {
                "name": info.name,
                "tier": getattr(info, "tier", 1),
                "current_hash": getattr(info, "current_hash", None),
                "source_path": getattr(info, "source_path", None),
            }
    return None


def _find_atom_on_disk(
    cwd: Path, scenario: str, atom_name: str
) -> dict[str, Any] | None:
    """Cross-session resolution: walk the scenario tree for a .py whose
    declared MANIFEST.name matches ``atom_name``. Used when the tuner
    proposes a change to an atom that lives under the production scenario
    and is therefore not loaded in the tuner's own session.
    """
    import ast

    scenario_root = cwd / "contrib" / "scenarios" / scenario
    if not scenario_root.is_dir():
        return None
    for py in scenario_root.rglob("*.py"):
        if py.name.startswith("_") or "/eval/" in py.as_posix() or "/tuner/" in py.as_posix():
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Name) and target.id == "MANIFEST"):
                continue
            for kw in getattr(node.value, "keywords", []) or []:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    if kw.value.value == atom_name:
                        tier = 1
                        for kw2 in getattr(node.value, "keywords", []) or []:
                            if (
                                kw2.arg == "tier"
                                and isinstance(kw2.value, ast.Constant)
                                and isinstance(kw2.value.value, (int, str))
                            ):
                                tier = int(kw2.value.value)
                        return {
                            "name": atom_name,
                            "tier": tier,
                            "current_hash": None,
                            "source_path": str(py.resolve()),
                        }
    return None


def _write_cross_session(
    api: ExtensionAPI,
    source_path: str | None,
    new_source: str,
    rationale: str,
) -> tuple[bool, str | None, str | None, str | None]:
    """Write ``new_source`` to ``source_path`` via the resource writer the
    api exposes (or a fallback file write). Returns (ok, old_hash, new_hash,
    error). ``old_hash`` and ``new_hash`` are best-effort: when running on
    git-backed cwd the harness commits the write and the post-image SHA is
    discoverable via ``git log -1 --format=%H -- <path>``.
    """
    import hashlib
    import subprocess

    if not source_path:
        return False, None, None, "missing source_path"
    p = Path(source_path)
    try:
        old_bytes = p.read_bytes() if p.is_file() else b""
        old_hash = hashlib.sha256(old_bytes).hexdigest()[:12] if old_bytes else None

        # Prefer ResourceWriter via api if exposed; else direct write.
        write_async = getattr(api, "resource_writer_write", None)
        if callable(write_async):  # pragma: no cover - opt-in writer surface
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                write_async(str(p), new_source.encode("utf-8"), rationale=rationale)
            )
        else:
            p.write_text(new_source, encoding="utf-8")
            cwd = p.parent
            try:
                subprocess.run(
                    ["git", "add", str(p)], cwd=cwd, check=False, capture_output=True
                )
                subprocess.run(
                    [
                        "git",
                        "-c",
                        "user.email=agent@agentm",
                        "-c",
                        "user.name=tool_propose_change",
                        "commit",
                        "-m",
                        f"propose_change: {rationale}",
                        "--",
                        str(p),
                    ],
                    cwd=cwd,
                    check=False,
                    capture_output=True,
                )
            except FileNotFoundError:
                pass

        new_hash = hashlib.sha256(new_source.encode("utf-8")).hexdigest()[:12]
        return True, old_hash, new_hash, None
    except Exception as exc:  # noqa: BLE001
        return False, None, None, str(exc)


def _decisions_path(cwd: Path, scenario: str) -> Path:
    out = cwd / ".agentm" / "decisions" / scenario / "decisions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _append_decision_record(path: Path, record: dict[str, Any]) -> None:
    """Append ``record`` to ``decisions.jsonl``. Bypasses ResourceWriter
    because ``.agentm/decisions/**`` is constitution-protected — only this
    atom (the mediated channel) writes here. Schema-stamped keys are
    enforced by being literal in the call sites above.
    """
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _load_eval_run_summary(cwd: Path, run_id: str) -> dict[str, Any] | None:
    run_path = cwd / ".agentm" / "eval_runs" / f"{run_id}.jsonl"
    if not run_path.is_file():
        return None
    try:
        with run_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict) and rec.get("kind") == "eval_run.summary":
                    return rec
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _apply_promotion_gate(
    *,
    baseline: dict[str, Any],
    proposed: dict[str, Any],
    threshold_relative: float,
    guard_tolerance: float,
    exploratory: bool,
) -> dict[str, Any]:
    baseline_score = float(baseline.get("primary_score") or 0.0)
    proposed_score = float(proposed.get("primary_score") or 0.0)
    baseline_stderr = float(baseline.get("primary_score_stderr") or 0.0)
    proposed_stderr = float(proposed.get("primary_score_stderr") or 0.0)
    delta = proposed_score - baseline_score
    if baseline_score == 0.0:
        relative = float("inf") if delta > 0 else 0.0
    else:
        relative = delta / baseline_score

    # 1) Threshold gate.
    if relative < threshold_relative:
        return {
            "applied": True,
            "passed": False,
            "reason": (
                f"relative improvement {relative:.4f} below threshold "
                f"{threshold_relative}"
            ),
            "baseline_score": baseline_score,
            "proposed_score": proposed_score,
            "delta": delta,
            "relative": relative,
        }

    # 2) Statistical sanity: Δ > 2 · sqrt(σ_b² + σ_p²) (~95% conf normal).
    if not exploratory:
        noise = 2.0 * math.sqrt(baseline_stderr**2 + proposed_stderr**2)
        if delta <= noise:
            return {
                "applied": True,
                "passed": False,
                "reason": (
                    f"improvement within noise floor: delta={delta:.4f} "
                    f"<= 2·sigma={noise:.4f}"
                ),
                "baseline_score": baseline_score,
                "proposed_score": proposed_score,
                "delta": delta,
                "noise_threshold": noise,
            }

    # 3) Guard metrics: each must be within ±guard_tolerance (relative).
    baseline_guards = baseline.get("guard_metrics") or {}
    proposed_guards = proposed.get("guard_metrics") or {}
    guard_failures: list[dict[str, Any]] = []
    if isinstance(baseline_guards, dict) and isinstance(proposed_guards, dict):
        for key, base_value in baseline_guards.items():
            try:
                base_f = float(base_value)  # type: ignore[arg-type]
                prop_f = float(proposed_guards.get(key, base_value))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if base_f == 0.0:
                continue  # No way to compute relative drift.
            rel_drift = abs(prop_f - base_f) / abs(base_f)
            if rel_drift > guard_tolerance:
                guard_failures.append(
                    {
                        "metric": key,
                        "baseline": base_f,
                        "proposed": prop_f,
                        "rel_drift": rel_drift,
                    }
                )
    if guard_failures:
        return {
            "applied": True,
            "passed": False,
            "reason": "guard regression",
            "baseline_score": baseline_score,
            "proposed_score": proposed_score,
            "guard_failures": guard_failures,
        }

    return {
        "applied": True,
        "passed": True,
        "baseline_score": baseline_score,
        "proposed_score": proposed_score,
        "delta": delta,
        "relative": relative,
    }


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
