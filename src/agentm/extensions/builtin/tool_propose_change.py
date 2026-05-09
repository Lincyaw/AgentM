"""Tool atom: gate-keeper for activating an atom mutation.

See ``.claude/designs/per-task-evolution-loop.md`` §6 / §8 / §11.

Required-arg validation enforces "evidence-driven, not error-driven":
the caller must pass both ``eval_run_baseline`` and ``eval_run_proposed``
or the call is rejected. Tier-2 atoms cannot be auto-activated and produce
a ``pending_human_approval`` decision record (no ``reload_atom`` call).

Phase 2 (B-1) splits the gate role into two:

- **Inclusion gate** — Pareto: a candidate enters the pool iff it wins
  on >=1 task vs the current frontier. Strictly-dominated candidates are
  marked pruned (``candidates/<id>.json.pruned`` sidecar; the .json
  itself stays for audit). This runs whenever an eval lookup succeeds,
  even when the deployment gate ultimately rejects.
- **Deployment gate** — the legacy four-floor check (threshold,
  statistical sanity, guard tolerance, rollback bypass). Decides whether
  to swap the live atom; *not* whether to keep the candidate.

For tier-1 ``activate`` decisions the deployment gate runs:

1. ``threshold_relative`` improvement on primary score.
2. Statistical-sanity check (delta > 2 * sqrt(sigma_b^2 + sigma_p^2)) —
   disabled for ``decision="exploratory"``.
3. Guard metrics within ``+/- guard_tolerance`` (relative).
4. ``decision="rollback"`` skips gates entirely (rollback is always safe).

ChangeSpec kinds (B-3): MVP supported ``atom_source``; Phase 2 also
supports ``system_prompt``, ``manifest_field``, ``manifest_extensions``
through validators under ``contrib/extensions/changespec_validators/``.
``scenario_compose`` remains ``not_yet_implemented``.

Budget (B-6): both ``rollouts_used`` and ``usd_used`` are read from
``.agentm/decisions/<scenario>/budget.json`` before evidence checks and
incremented on each call. A configured ``rollouts_budget`` /
``usd_budget`` causes the next call to abort with ``budget_exhausted``.

Activation records append to ``.agentm/decisions/<scenario>/activations.jsonl``;
a ``tree.jsonl`` sibling records parent->child edges between candidates.
Both paths are constitution-protected — only this atom writes here.
"""

from __future__ import annotations

import json
import math
import time
import uuid
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
        "reload_atom on success. Activation record appended to "
        ".agentm/decisions/<scenario>/activations.jsonl."
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
            "rollouts_budget": {
                "type": ["integer", "null"],
                "description": (
                    "B-6: per-tuning-session rollout cap. None = unbounded."
                ),
            },
            "usd_budget": {
                "type": ["number", "null"],
                "description": (
                    "B-6: per-tuning-session USD cap; mirrors "
                    "tool_eval_run.max_cost_usd. None = unbounded."
                ),
            },
        },
        "additionalProperties": True,
    },
)


_CHANGE_KINDS = (
    "atom_source",
    "system_prompt",
    "manifest_extensions",
    "manifest_field",
    "scenario_compose",
)

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target": {
            "type": "object",
            "description": (
                "ChangeSpec — what to change. MVP accepts only "
                "kind='atom_source'; other kinds reserved for Phase 2 and "
                "rejected with not_yet_implemented."
            ),
            "properties": {
                "kind": {"type": "string", "enum": list(_CHANGE_KINDS)},
                "path": {
                    "type": "string",
                    "description": (
                        "Target file relative to "
                        "contrib/scenarios/<target_scenario>/."
                    ),
                },
                "new_content": {
                    "type": "string",
                    "description": (
                        "Full replacement content (or structured patch — "
                        "Phase 2)."
                    ),
                },
                "target_atom": {
                    "type": ["string", "null"],
                    "description": (
                        "Atom name, only for kind='atom_source'; null "
                        "otherwise."
                    ),
                },
            },
            "required": ["kind", "path", "new_content"],
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
        "eval_run_baseline": {"type": "string"},
        "eval_run_proposed": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": ["activate", "rollback", "exploratory"],
        },
    },
    "required": [
        "target",
        "rationale",
        "eval_run_baseline",
        "eval_run_proposed",
        "decision",
    ],
    "additionalProperties": False,
}


_DEFAULT_THRESHOLD_RELATIVE = 0.05
_DEFAULT_GUARD_TOLERANCE = 0.10
_DEFAULT_STOP_AFTER_NO_IMPROVEMENT = 3


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    target_scenario = str(config.get("target_scenario") or "default")
    promotion = config.get("promotion") or {}
    threshold_relative = float(
        promotion.get("threshold_relative", _DEFAULT_THRESHOLD_RELATIVE)
    )
    guard_tolerance = float(
        promotion.get("guard_tolerance", _DEFAULT_GUARD_TOLERANCE)
    )
    # B-9: structural anti-thrash counter. Default 3 consecutive rejections;
    # explicit None disables. Counter reads from activations.jsonl on every
    # call so it persists across tuner restarts (the load-bearing property).
    stop_after_raw = promotion.get(
        "stop_after_no_improvement", _DEFAULT_STOP_AFTER_NO_IMPROVEMENT
    )
    stop_after_no_improvement: int | None
    if stop_after_raw is None:
        stop_after_no_improvement = None
    else:
        try:
            stop_after_no_improvement = int(stop_after_raw)
        except (TypeError, ValueError):
            stop_after_no_improvement = _DEFAULT_STOP_AFTER_NO_IMPROVEMENT
        if (
            stop_after_no_improvement is not None
            and stop_after_no_improvement <= 0
        ):
            stop_after_no_improvement = None
    rollouts_budget_raw = config.get("rollouts_budget")
    rollouts_budget: int | None
    try:
        rollouts_budget = (
            int(rollouts_budget_raw) if rollouts_budget_raw is not None else None
        )
    except (TypeError, ValueError):
        rollouts_budget = None
    usd_budget_raw = config.get("usd_budget")
    usd_budget: float | None
    try:
        usd_budget = (
            float(usd_budget_raw) if usd_budget_raw is not None else None
        )
    except (TypeError, ValueError):
        usd_budget = None
    cwd = Path(api.cwd)

    async def _execute(args: dict[str, Any]) -> ToolResult:
        # B-6: budget gate runs first. If a prior call already exhausted
        # the cap, refuse before doing any evidence work.
        budget_check = _check_budget(
            cwd, target_scenario, rollouts_budget, usd_budget
        )
        if budget_check is not None:
            return _error(budget_check)

        # Required-arg validation — explicit even though the schema requires
        # them, because the design's evidence contract is the load-bearing
        # piece (P3 in design §10).
        for key in (
            "eval_run_baseline",
            "eval_run_proposed",
            "rationale",
            "decision",
        ):
            value = args.get(key)
            if not isinstance(value, str) or not value:
                return _error(
                    f"evidence missing: {key!r} is required and must be a "
                    f"non-empty string"
                )

        # Validate ChangeSpec (design §6).
        target_spec = args.get("target")
        if not isinstance(target_spec, dict):
            return _error(
                "evidence missing: 'target' is required and must be a "
                "ChangeSpec object {kind, path, new_content, target_atom?}"
            )
        kind = target_spec.get("kind")
        if not isinstance(kind, str) or kind not in _CHANGE_KINDS:
            return _error(
                f"target.kind must be one of {list(_CHANGE_KINDS)!r}; "
                f"got {kind!r}"
            )
        # B-3: dispatch to per-kind validator. ``scenario_compose`` is
        # explicitly out of scope for Phase 2 — keep the not_yet_implemented
        # rejection in place so callers get a clear signal.
        if kind == "scenario_compose":
            return _error(
                f"not_yet_implemented: target.kind={kind!r} requires "
                f"harness compose-graph reload; out of scope for Phase 2 "
                f"(design §11)"
            )
        validator = _load_validator(kind)
        if validator is None:
            return _error(
                f"not_yet_implemented: no validator registered for "
                f"target.kind={kind!r}"
            )
        v_result = validator(target_spec, cwd, target_scenario)
        if not v_result.get("ok"):
            err_text = v_result.get("error") or "validator rejected"
            return _error(str(err_text))

        path_value = str(target_spec["path"])
        new_content = str(target_spec.get("new_content") or "")
        atom_name_raw = target_spec.get("target_atom")

        # Per-kind branching — only ``atom_source`` interacts with
        # in-session reload + tier metadata. Manifest-bearing kinds use
        # the cross-session writer path.
        if kind == "atom_source":
            target_atom = str(atom_name_raw) if isinstance(atom_name_raw, str) else ""
        else:
            target_atom = ""
        new_source = new_content

        change_spec: dict[str, Any] = {
            "kind": kind,
            "path": path_value,
            "new_content": new_content,
            "target_atom": atom_name_raw if isinstance(atom_name_raw, str) else None,
        }
        rationale = str(args["rationale"])
        baseline_id = str(args["eval_run_baseline"])
        proposed_id = str(args["eval_run_proposed"])
        decision = str(args["decision"])

        # Look up the loaded atom's tier — drives the tier-2 deferral gate.
        # Cross-session case: tuner runs in its own session and does NOT load
        # the target_scenario's atoms. Fall back to a filesystem scan under
        # the scenario root so we can still resolve tier + source path.
        # Only atom_source kinds carry an atom name; manifest-bearing kinds
        # use the resolved_path from the validator and are treated as tier-1.
        loaded: dict[str, Any] | None
        atom_in_session: bool
        if kind == "atom_source":
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
        else:
            # Manifest-bearing kinds: there is no MANIFEST.tier to read;
            # the target is the validated on-disk path returned by the
            # validator. Treat as tier-1 (auto-promotable subject to the
            # deployment gate) — manifest field tweaks are scenario-shape
            # policy, not capability boundaries.
            loaded = {
                "name": kind,
                "tier": 1,
                "current_hash": None,
                "source_path": v_result.get("resolved_path"),
            }
            atom_in_session = False
            tier = 1

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
                "change_spec": change_spec,
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

        # B-9: structural anti-thrash gate. After N consecutive ``rejected``
        # entries in activations.jsonl, refuse further deployment-gated
        # decisions (``activate`` / ``merge``). The counter persists across
        # tuner restarts because we re-read the log on every call. A
        # ``stop_blocked`` record is appended to the log but does NOT itself
        # increment the counter (else the constraint self-perpetuates).
        if (
            stop_after_no_improvement is not None
            and decision == "activate"
        ):
            decisions_path = _decisions_path(cwd, target_scenario)
            consec = _count_consecutive_rejections(decisions_path)
            if consec >= stop_after_no_improvement:
                stop_record = {
                    "at": time.time(),
                    "kind": "stop_blocked",
                    "scenario": target_scenario,
                    "atom": target_atom,
                    "tier": tier,
                    "rationale": rationale,
                    "change_spec": change_spec,
                    "evidence": {
                        "baseline_run": baseline_id,
                        "proposed_run": proposed_id,
                        "consecutive_rejections": consec,
                        "threshold": stop_after_no_improvement,
                    },
                    "decision": decision,
                    "by": "tool_propose_change",
                }
                _append_decision_record(decisions_path, stop_record)
                return _error(
                    f"stop_after_no_improvement: {consec} consecutive "
                    f"rejections; change strategy or escalate"
                )

        # Load the eval-run summaries.
        baseline = _load_eval_run_summary(cwd, baseline_id)
        if baseline is None:
            return _error(f"baseline eval run not found: {baseline_id}")
        proposed = _load_eval_run_summary(cwd, proposed_id)
        if proposed is None:
            return _error(f"proposed eval run not found: {proposed_id}")

        # B-1 inclusion phase: build the candidate record from the
        # proposed eval-run, write it under candidates/ unconditionally,
        # then run Pareto pruning across the pool. This phase runs *before*
        # the deployment gate so dominated-but-niche-winning candidates
        # are retained for future search even when the four-floor gate
        # rejects this activation.
        decisions_path = _decisions_path(cwd, target_scenario)
        decisions_dir = decisions_path.parent
        parent_id = _last_candidate_id(decisions_path)
        candidate_id = f"c_{uuid.uuid4().hex[:12]}"
        proposed_per_task = _load_eval_run_per_task(cwd, proposed_id)
        candidate_record = {
            "candidate_id": candidate_id,
            "parent_id": parent_id,
            "change_spec": change_spec,
            "per_task_scores": _per_task_score_map(
                proposed_per_task, holdout=False
            ),
            "holdout_scores": _per_task_score_map(
                proposed_per_task, holdout=True
            ),
            "eval_run_id": proposed_id,
            "created_at": time.time(),
        }
        _write_candidate_record(decisions_dir, candidate_record)
        _append_tree_edge(decisions_dir, child=candidate_id, parent=parent_id)
        # Pareto pruning: a candidate is on the frontier iff it wins on
        # >=1 task across the pool. Strictly-dominated peers get a
        # ``.pruned`` sidecar (the .json itself stays for audit).
        _prune_dominated_candidates(decisions_dir)

        # B-6: increment rollouts_used (one per call). usd_used is
        # incremented by tool_eval_run; we don't double-count cost here.
        _bump_budget_rollouts(cwd, target_scenario)

        # B-1 deployment gate (formerly "promotion gate"). Skipped for
        # rollback + exploratory. Even on rejection the candidate stays in
        # the pool — rejection only prevents the swap.
        gate_outcome: dict[str, Any] = {"applied": False}
        if decision == "activate":
            gate_outcome = _apply_deployment_gate(
                baseline=baseline,
                proposed=proposed,
                threshold_relative=threshold_relative,
                guard_tolerance=guard_tolerance,
                exploratory=False,
            )
            if not gate_outcome["passed"]:
                # Reject without invoking reload; record the rejection.
                # The candidate record is already on disk (inclusion phase).
                record = {
                    "at": time.time(),
                    "kind": "rejected",
                    "scenario": target_scenario,
                    "atom": target_atom,
                    "tier": tier,
                    "rationale": rationale,
                    "change_spec": change_spec,
                    "candidate_id": candidate_id,
                    "evidence": {
                        "baseline_run": baseline_id,
                        "proposed_run": proposed_id,
                        "gate": gate_outcome,
                    },
                    "by": "tool_propose_change",
                }
                _append_decision_record(decisions_path, record)
                return _error(
                    f"deployment gate failed: {gate_outcome['reason']}; "
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
        if kind == "atom_source" and atom_in_session:
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
            record = {
                "at": time.time(),
                "kind": "reload_failed",
                "scenario": target_scenario,
                "atom": target_atom,
                "tier": tier,
                "rationale": rationale,
                "change_spec": change_spec,
                "candidate_id": candidate_id,
                "evidence": {
                    "baseline_run": baseline_id,
                    "proposed_run": proposed_id,
                },
                "error": error,
                "by": "tool_propose_change",
            }
            _append_decision_record(decisions_path, record)
            return _error(f"reload failed: {error}")

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
            "change_spec": change_spec,
            "candidate_id": candidate_id,
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
                    "candidate_id": candidate_id,
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
    out = cwd / ".agentm" / "decisions" / scenario / "activations.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _append_decision_record(path: Path, record: dict[str, Any]) -> None:
    """Append ``record`` to ``activations.jsonl``. Bypasses ResourceWriter
    because ``.agentm/decisions/**`` is constitution-protected — only this
    atom (the mediated channel) writes here. Schema-stamped keys are
    enforced by being literal in the call sites above.
    """
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _write_candidate_record(decisions_dir: Path, record: dict[str, Any]) -> None:
    """Write ``record`` to ``candidates/<candidate_id>.json`` under
    ``decisions_dir``. The ``.agentm/decisions/**`` glob covers this path,
    so it inherits the constitution-protect rejection from ResourceWriter
    — only this atom (the mediated channel) writes here.
    """
    candidates_dir = decisions_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = str(record["candidate_id"])
    path = candidates_dir / f"{candidate_id}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)


def _last_candidate_id(activations_path: Path) -> str | None:
    """Scan ``activations.jsonl`` for the most recent record carrying a
    ``candidate_id`` and return it, or None if the log is empty / first
    activation."""
    if not activations_path.is_file():
        return None
    last: str | None = None
    try:
        with activations_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = rec.get("candidate_id") if isinstance(rec, dict) else None
                if isinstance(cid, str) and cid:
                    last = cid
    except OSError:
        return None
    return last


def _load_eval_run_per_task(cwd: Path, run_id: str) -> list[dict[str, Any]]:
    """Read ``.agentm/eval_runs/<run_id>.jsonl`` and return the
    ``eval_run.task`` records (one per task). Returns an empty list when
    the run is unknown or contains no task records (degenerate case is
    fine — candidate scores just default to empty maps).
    """
    run_path = cwd / ".agentm" / "eval_runs" / f"{run_id}.jsonl"
    if not run_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        with run_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("kind") == "eval_run.task":
                    out.append(rec)
    except OSError:
        return []
    return out


def _per_task_score_map(
    records: list[dict[str, Any]], *, holdout: bool
) -> dict[str, float]:
    """Project ``eval_run.task`` records into a {task_id: grade_mean}
    map, filtered by holdout flag."""
    out: dict[str, float] = {}
    for rec in records:
        if bool(rec.get("holdout", False)) != holdout:
            continue
        task_id = rec.get("task_id")
        score = rec.get("grade_mean")
        if isinstance(task_id, str) and isinstance(score, (int, float)):
            out[task_id] = float(score)
    return out


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


def _apply_deployment_gate(
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


def _load_validator(kind: str) -> Any:
    """Load the per-kind validator from
    ``contrib/extensions/changespec_validators/<kind>.py`` and return its
    ``validate`` callable. Returns ``None`` when no module is registered.

    The validators live under ``contrib/`` (not ``src/agentm/``) because
    they encode scenario-shape policy, not core mechanism. Importing them
    from this builtin atom is acceptable: validators are not atoms (no
    MANIFEST), so the §11 "no atom-to-atom imports" rule does not apply.
    """
    import importlib

    safe_kind = kind.strip().lower()
    if not safe_kind.replace("_", "").isalnum():
        return None
    module_name = (
        f"contrib.extensions.changespec_validators.{safe_kind}"
    )
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    fn = getattr(module, "validate", None)
    return fn if callable(fn) else None


def _budget_path(cwd: Path, scenario: str) -> Path:
    return cwd / ".agentm" / "decisions" / scenario / "budget.json"


def _read_budget(cwd: Path, scenario: str) -> dict[str, Any]:
    path = _budget_path(cwd, scenario)
    default = {
        "scenario": scenario,
        "rollouts_used": 0,
        "usd_used": 0.0,
        "updated_at": 0.0,
    }
    if not path.is_file():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default
    return {
        "scenario": str(data.get("scenario") or scenario),
        "rollouts_used": int(data.get("rollouts_used") or 0),
        "usd_used": float(data.get("usd_used") or 0.0),
        "updated_at": float(data.get("updated_at") or 0.0),
    }


def _write_budget(cwd: Path, scenario: str, budget: dict[str, Any]) -> None:
    import os

    path = _budget_path(cwd, scenario)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(budget, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _check_budget(
    cwd: Path,
    scenario: str,
    rollouts_budget: int | None,
    usd_budget: float | None,
) -> str | None:
    """B-6 budget gate. Returns a ``budget_exhausted: ...`` error message
    when the configured cap is hit, else None. Caps are read from
    ``budget.json``; tool_eval_run increments ``usd_used`` and this atom
    increments ``rollouts_used`` per call."""
    if rollouts_budget is None and usd_budget is None:
        return None
    budget = _read_budget(cwd, scenario)
    if (
        rollouts_budget is not None
        and budget["rollouts_used"] >= rollouts_budget
    ):
        return (
            f"budget_exhausted: rollouts_used={budget['rollouts_used']} "
            f">= rollouts_budget={rollouts_budget}"
        )
    if usd_budget is not None and budget["usd_used"] >= usd_budget:
        return (
            f"budget_exhausted: usd_used={budget['usd_used']:.4f} "
            f">= usd_budget={usd_budget:.4f}"
        )
    return None


def _bump_budget_rollouts(cwd: Path, scenario: str) -> None:
    """Increment ``rollouts_used`` by one. Called once per propose_change
    call (B-6). The eval_run atom continues to be the sole writer of
    ``usd_used``; we deliberately don't double-count cost here."""
    budget = _read_budget(cwd, scenario)
    budget["rollouts_used"] = int(budget["rollouts_used"]) + 1
    budget["updated_at"] = time.time()
    _write_budget(cwd, scenario, budget)


def _append_tree_edge(
    decisions_dir: Path, *, child: str, parent: str | None
) -> None:
    """Append a parent->child edge to ``tree.jsonl`` (B-1). The file lives
    under the constitution-protected decisions dir; only this atom writes
    here. ``parent`` is null on the first candidate."""
    path = decisions_dir / "tree.jsonl"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    record = {"child": child, "parent": parent, "at": time.time()}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _prune_dominated_candidates(decisions_dir: Path) -> None:
    """Pareto pruning (B-1). A candidate is on the frontier iff it is the
    sole top-scorer on >=1 task across the pool's ``per_task_scores`` (a
    candidate that ties for the best on a task does *not* claim that win
    — strict-dominance only). Strictly-dominated peers get a sidecar
    ``<id>.json.pruned`` flag; the .json itself stays for audit. Already-
    pruned candidates that happen to win on a task on a later pass have
    their flag removed (re-include).

    We use strict argmax (a candidate must be strictly greater than all
    others on at least one task) to avoid infinite tie-breaking when two
    identical candidates are added — neither would otherwise claim
    inclusion.
    """
    candidates_dir = decisions_dir / "candidates"
    if not candidates_dir.is_dir():
        return
    records: dict[str, dict[str, Any]] = {}
    for p in candidates_dir.glob("c_*.json"):
        try:
            with p.open("r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        cid = rec.get("candidate_id")
        scores = rec.get("per_task_scores")
        if isinstance(cid, str) and isinstance(scores, dict):
            records[cid] = rec
    if not records:
        return

    # Collect the union of task ids across all candidates.
    task_ids: set[str] = set()
    for rec in records.values():
        for tid in rec["per_task_scores"].keys():
            task_ids.add(tid)

    winners: set[str] = set()
    for tid in task_ids:
        best_score = float("-inf")
        best_holders: list[str] = []
        for cid, rec in records.items():
            score = rec["per_task_scores"].get(tid)
            if not isinstance(score, (int, float)):
                continue
            score_f = float(score)
            if score_f > best_score:
                best_score = score_f
                best_holders = [cid]
            elif score_f == best_score:
                best_holders.append(cid)
        # Strict winner only — ties claim no inclusion via this task.
        if len(best_holders) == 1:
            winners.add(best_holders[0])

    for cid in records:
        flag = candidates_dir / f"{cid}.json.pruned"
        if cid in winners:
            # Restore: drop a stale .pruned flag if a later candidate
            # changed who wins.
            if flag.is_file():
                try:
                    flag.unlink()
                except OSError:
                    pass
        else:
            if not flag.is_file():
                try:
                    flag.write_text("", encoding="utf-8")
                except OSError:
                    pass


def _count_consecutive_rejections(activations_path: Path) -> int:
    """B-9: walk ``activations.jsonl`` from the most recent entry backward
    and count contiguous ``kind == "rejected"`` records. The walk stops on
    any record that represents *forward progress* — ``activate``,
    ``exploratory``, ``rollback``, ``merge`` — at which point the counter
    resets to whatever has been seen so far (which is the count between
    the last forward-progress record and the present).

    ``stop_blocked`` entries are skipped (they neither increment nor
    reset). Without this carve-out the constraint would self-perpetuate:
    one block becomes a permanent block. ``reload_failed`` is treated as
    a rejection — it represents an attempted swap that didn't land.

    The counter persists across tuner restarts because it is computed
    from on-disk state, never cached in memory. This is the load-bearing
    property of B-9's fail-stop test.
    """
    if not activations_path.is_file():
        return 0
    try:
        with activations_path.open("r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    except OSError:
        return 0
    consec = 0
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        kind = rec.get("kind")
        if kind == "stop_blocked":
            # Transparent: doesn't break the chain, doesn't extend it.
            continue
        if kind in ("rejected", "reload_failed"):
            consec += 1
            continue
        # Any other terminal kind (activate / exploratory / rollback /
        # merge / pending_human_approval) breaks the chain.
        break
    return consec


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
