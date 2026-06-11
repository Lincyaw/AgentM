"""``llmharness-distill`` CLI: label and export.

Two subcommands:

* ``label`` — walks a replay-log directory, joins each session to its
  meta sidecar + dataset row, and runs the two-stage oracle/rewriter
  per auditor record. Writes one ``<session_id>.labels.jsonl`` per
  session under the output directory.

* ``export`` — converts labels + replay logs into SFT JSONL. Produces
  ``extractor.jsonl``, ``auditor.jsonl``, and ``dropped.jsonl`` under
  the output directory.

Usage::

    llmharness-distill label \\
        --replay-dir /path/to/.agentm/audit_replay \\
        --dataset    /path/to/rca/data.jsonl \\
        --out        ./distill_labels \\
        --cwd        /tmp/agentm-distill \\
        --oracle-provider   agentm.ai.providers.anthropic_provider \\
        --rewriter-provider agentm.ai.providers.openai_provider

    llmharness-distill export \\
        --labels ./distill_labels \\
        --replay-dir /path/to/.agentm/audit_replay \\
        --out  ./sft \\
        --phase both
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agentm.core.abi import TraceReader

from llmharness.replay.record import iter_records

from .binding import read_sample_meta
from .export import (
    auditor_records_from_labels,
    dropped_records_from_labels,
    extractor_records_from_replay,
    write_jsonl,
)
from .gt import GroundTruth, load_dataset
from .oracle import label_auditor_record
from .rl_prompts import Phase as RlPhase
from .rl_prompts import rl_rows_from_replay, serialize_row

_logger = logging.getLogger(__name__)


def _parse_provider(spec: str | None) -> tuple[str, dict[str, Any]] | None:
    """Accept ``provider_id`` or ``provider_id#{json}``. None → AgentM default.

    A short id (no dot, e.g. ``openai``) is resolved through
    :data:`agentm.ai.types.DEFAULT_PROVIDER_REGISTRY` so env-derived knobs
    like ``WARPGATE_TICKET`` and ``OPENAI_VERIFY_SSL`` are folded in —
    matching what the ``agentm`` CLI does. A dotted module path is used
    verbatim; the caller owns the full config dict in that case.
    """
    if not spec:
        return None
    if "#" in spec:
        module, raw = spec.split("#", 1)
        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError:
            cfg = {}
    else:
        module, cfg = spec, {}
    if not isinstance(cfg, dict):
        cfg = {}
    if "." not in module:
        try:
            from agentm.ai.types import DEFAULT_PROVIDER_REGISTRY

            return DEFAULT_PROVIDER_REGISTRY.build(module, cfg)
        except (ImportError, KeyError):
            pass
    return module, cfg


def _replay_record_dicts(path: Path) -> list[dict[str, Any]]:
    """All replay records of a sidecar file, as plain dicts."""
    return [
        {
            "phase": r.phase,
            "turn_index": r.turn_index,
            "session_id": r.session_id,
            "trace_id": r.trace_id,
            "ts_ns": r.ts_ns,
            "compose_kwargs": r.compose_kwargs,
            "payload": r.payload,
            "provider": r.provider,
            "output": r.output,
            "status": r.status,
            "error": r.error,
            "latency_ms": r.latency_ms,
            "extras": r.extras,
            "raw_assistant_messages": r.raw_assistant_messages,
        }
        for r in iter_records(path)
    ]


# ----- label ----------------------------------------------------------------


async def _label_session(
    *,
    replay_file: Path,
    out_dir: Path,
    cwd: str,
    gt_index: dict[str, GroundTruth],
    oracle_provider: tuple[str, dict[str, Any]] | None,
    rewriter_provider: tuple[str, dict[str, Any]] | None,
) -> tuple[int, int]:
    """Label one session. Returns (kept, dropped)."""
    session_id = replay_file.stem
    # ``replay_file`` is <cwd>/.agentm/audit_replay/<sid>.jsonl;
    # meta sidecar lives next to it as <sid>.meta.json.
    meta = read_sample_meta(replay_file.with_suffix(".meta.json"))
    if meta is None:
        _logger.warning("no meta sidecar for %s; skipping", replay_file.name)
        return 0, 0

    gt = gt_index.get(meta.sample_id)
    if gt is None:
        _logger.warning(
            "sample_id %r not in dataset; skipping %s", meta.sample_id, replay_file.name
        )
        return 0, 0

    records = _replay_record_dicts(replay_file)
    auditor_records = [r for r in records if r.get("phase") == "auditor"]

    out_path = out_dir / f"{session_id}.labels.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in auditor_records:
            sample = await label_auditor_record(
                cwd=cwd,
                record=rec,
                gt=gt,
                sample_id=meta.sample_id,
                oracle_provider=oracle_provider,
                rewriter_provider=rewriter_provider,
            )
            fh.write(sample.to_jsonl())
            fh.write("\n")
            if sample.drop:
                dropped += 1
            else:
                kept += 1
    return kept, dropped


def _cmd_label(args: argparse.Namespace) -> int:
    replay_dir = Path(args.replay_dir)
    out_dir = Path(args.out)
    dataset_path = Path(args.dataset)
    cwd = args.cwd or str(Path.cwd())
    oracle_provider = _parse_provider(args.oracle_provider)
    rewriter_provider = _parse_provider(args.rewriter_provider or args.oracle_provider)

    if not replay_dir.is_dir():
        print(f"replay-dir not found: {replay_dir}", file=sys.stderr)
        return 2
    if not dataset_path.is_file():
        print(f"dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    gt_index = load_dataset(dataset_path)
    if not gt_index:
        print(f"dataset {dataset_path} loaded zero rows", file=sys.stderr)
        return 2

    sessions = sorted(replay_dir.glob("*.jsonl"))
    if not sessions:
        print(f"no replay logs in {replay_dir}", file=sys.stderr)
        return 2

    total_kept = 0
    total_dropped = 0
    for replay_file in sessions:
        kept, dropped = asyncio.run(
            _label_session(
                replay_file=replay_file,
                out_dir=out_dir,
                cwd=cwd,
                gt_index=gt_index,
                oracle_provider=oracle_provider,
                rewriter_provider=rewriter_provider,
            )
        )
        print(f"{replay_file.name}: kept={kept} dropped={dropped}")
        total_kept += kept
        total_dropped += dropped

    print(f"total: kept={total_kept} dropped={total_dropped} out={out_dir}")
    return 0


# ----- export ---------------------------------------------------------------


def _iter_label_rows(label_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(label_dir.glob("*.labels.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _cmd_export(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    label_dir = Path(args.labels)
    replay_dir = Path(args.replay_dir) if args.replay_dir else None
    phase = args.phase

    want_extractor = phase in ("extractor", "both")
    want_auditor = phase in ("auditor", "both")

    n_ext = n_aud = n_drop = 0

    if want_extractor:
        if replay_dir is None or not replay_dir.is_dir():
            print(
                "--replay-dir is required when phase includes extractor",
                file=sys.stderr,
            )
            return 2

        def _iter() -> Iterable[Any]:
            for replay_file in sorted(replay_dir.glob("*.jsonl")):
                meta = read_sample_meta(replay_file.with_suffix(".meta.json"))
                if meta is None:
                    continue
                yield from extractor_records_from_replay(
                    _replay_record_dicts(replay_file), sample_id=meta.sample_id
                )

        n_ext = write_jsonl(out_dir / "extractor.jsonl", _iter())

    if want_auditor:
        label_rows = list(_iter_label_rows(label_dir))
        n_aud = write_jsonl(out_dir / "auditor.jsonl", auditor_records_from_labels(label_rows))
        n_drop = write_jsonl(out_dir / "dropped.jsonl", dropped_records_from_labels(label_rows))

    print(f"extractor={n_ext} auditor={n_aud} dropped={n_drop} out={out_dir}")
    return 0


# ----- rl-prompts -----------------------------------------------------------


def _write_rl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(serialize_row(row))
            fh.write("\n")
            n += 1
    return n


def _cmd_rl_prompts(args: argparse.Namespace) -> int:
    replay_dir = Path(args.replay_dir)
    out_path = Path(args.out)
    phase = args.phase
    phases: tuple[RlPhase, ...]
    if phase == "extractor":
        phases = ("extractor",)
    elif phase == "auditor":
        phases = ("auditor",)
    else:
        phases = ("extractor", "auditor")

    if not replay_dir.is_dir():
        print(f"replay-dir not found: {replay_dir}", file=sys.stderr)
        return 2

    sessions = sorted(replay_dir.glob("*.jsonl"))
    if not sessions:
        print(f"no replay logs in {replay_dir}", file=sys.stderr)
        return 2

    def _iter() -> Iterable[dict[str, Any]]:
        for replay_file in sessions:
            yield from rl_rows_from_replay(
                _replay_record_dicts(replay_file),
                phases=phases,
            )

    n = _write_rl_rows(out_path, _iter())
    print(f"rl-prompts={n} out={out_path}")
    return 0


# ----- annotate-case-outcome ------------------------------------------------


def _last_submit_final_report(main_jsonl: Path) -> dict[str, Any] | None:
    """Scan a bundle's main.jsonl for the last submit_final_report args.

    Recognizes three on-disk shapes:

    * **OTLP/JSON observability** (current AgentM writer, post
      single-event-log cutover): spans named ``execute_tool
      submit_final_report`` whose ``gen_ai.tool.call.arguments`` attribute
      carries the JSON-encoded args.
    * **Legacy observability rows**: ``kind == "event.dispatch"`` with
      ``name == "emit:tool_call"`` and ``attributes.event.tool_name ==
      "submit_final_report"`` — args under ``attributes.event.args``.
      Kept for backward compatibility with traces written before the
      OTLP cutover.
    * **Bare session-log rows**: ``type == "tool_call"`` with
      ``payload.tool_name == "submit_final_report"`` — args under
      ``payload.args``.

    Returns the last matching args dict, or ``None`` if none seen.
    """
    last_args: dict[str, Any] | None = None
    if not main_jsonl.is_file():
        return None
    # OTLP/JSON shape: walk spans for execute_tool submit_final_report and
    # decode the JSON-encoded ``gen_ai.tool.call.arguments`` attribute via
    # :class:`TraceReader` (which already unwraps the tagged union).
    for span in TraceReader(main_jsonl).iter_spans(name="execute_tool submit_final_report"):
        raw = span.attributes.get("gen_ai.tool.call.arguments")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            args = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(args, dict):
            last_args = args
    if last_args is not None:
        return last_args
    # Fallback: scan legacy + session-log shapes that may still live in
    # older bundles untouched by the OTLP cutover. These shapes are
    # already plain JSON lines so we read them directly.
    try:
        with main_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                # Legacy observability shape.
                if rec.get("kind") == "event.dispatch" and rec.get("name") == "emit:tool_call":
                    attrs = rec.get("attributes") or {}
                    if isinstance(attrs, dict):
                        event = attrs.get("event") or {}
                        if (
                            isinstance(event, dict)
                            and event.get("tool_name") == "submit_final_report"
                        ):
                            args = event.get("args")
                            if isinstance(args, dict):
                                last_args = args
                                continue
                # Session-log shape.
                if rec.get("type") == "tool_call":
                    payload = rec.get("payload") or {}
                    if (
                        isinstance(payload, dict)
                        and payload.get("tool_name") == "submit_final_report"
                    ):
                        args = payload.get("args")
                        if isinstance(args, dict):
                            last_args = args
    except OSError:
        return None
    return last_args


def _grade_submission(
    args: dict[str, Any],
    *,
    ground_truth: list[str],
    fault_type: str,
) -> tuple[float, float, float]:
    """Compute (service_hit, fault_kind_hit, composite).

    Mirrors the substring rule used by
    ``contrib/scenarios/rca/eval/baseline/grader.py`` — keep them in
    sync. We replicate (don't import) because the canonical grader is
    coupled to the on-disk observability trace path, whereas here we
    have the submission args already in hand.
    """
    raw = json.dumps(args, ensure_ascii=False).lower()
    service_hit = 1.0 if any(isinstance(s, str) and s.lower() in raw for s in ground_truth) else 0.0
    ft = (fault_type or "").strip().lower()
    fault_kind_hit = 1.0 if ft and ft in raw else 0.0
    composite = 0.7 * service_hit + 0.3 * fault_kind_hit
    return service_hit, fault_kind_hit, composite


def _annotate_one_bundle(
    bundle_dir: Path,
    *,
    case_metadata_override: Path | None = None,
) -> dict[str, Any] | None:
    meta_path = case_metadata_override or (bundle_dir / "case_metadata.json")
    if not meta_path.is_file():
        print(f"missing case_metadata.json for {bundle_dir}", file=sys.stderr)
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"failed to read {meta_path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(meta, dict):
        print(f"case_metadata.json must be an object: {meta_path}", file=sys.stderr)
        return None

    case_id = str(meta.get("case_id") or "")
    datapack_name = str(meta.get("datapack_name") or "")
    gt_raw = meta.get("ground_truth") or []
    ground_truth = [s for s in gt_raw if isinstance(s, str)]
    fault_type = str(meta.get("fault_type") or "")

    main_jsonl = bundle_dir / "main.jsonl"
    submission = _last_submit_final_report(main_jsonl) if main_jsonl.is_file() else None
    submission_seen = submission is not None

    if submission_seen:
        assert submission is not None  # narrowing for mypy
        service_hit, fault_kind_hit, composite = _grade_submission(
            submission, ground_truth=ground_truth, fault_type=fault_type
        )
    else:
        service_hit = fault_kind_hit = composite = 0.0

    outcome: dict[str, Any] = {
        "case_id": case_id,
        "datapack_name": datapack_name,
        "service_hit": service_hit,
        "fault_kind_hit": fault_kind_hit,
        "composite_score": composite,
        "ground_truth": ground_truth,
        "fault_type": fault_type,
        "submission_seen": submission_seen,
    }
    out_path = bundle_dir / "case_outcome.json"
    out_path.write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outcome


def _cmd_annotate_case_outcome(args: argparse.Namespace) -> int:
    bundles = [Path(b) for b in args.bundle]
    override = Path(args.case_metadata) if args.case_metadata else None
    if override is not None and len(bundles) != 1:
        print(
            "--case-metadata may only be combined with a single --bundle",
            file=sys.stderr,
        )
        return 2
    n_ok = 0
    n_skip = 0
    for bundle in bundles:
        if not bundle.is_dir():
            print(f"bundle not a directory: {bundle}", file=sys.stderr)
            n_skip += 1
            continue
        outcome = _annotate_one_bundle(bundle, case_metadata_override=override)
        if outcome is None:
            n_skip += 1
            continue
        n_ok += 1
        print(
            f"{bundle}: case_id={outcome['case_id']} "
            f"composite={outcome['composite_score']:.3f} "
            f"submission_seen={outcome['submission_seen']}"
        )
    print(f"annotated={n_ok} skipped={n_skip}")
    return 0 if n_skip == 0 else 1


# ----- top-level ------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmharness-distill")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_label = sub.add_parser("label", help="Two-stage offline labeling.")
    p_label.add_argument("--replay-dir", required=True)
    p_label.add_argument("--dataset", required=True, help="rca-shaped JSONL")
    p_label.add_argument("--out", required=True)
    p_label.add_argument("--cwd", default=None, help="cwd for spawned children")
    p_label.add_argument(
        "--oracle-provider",
        default=None,
        help="dotted module[#json_config]; uses AgentM default when omitted",
    )
    p_label.add_argument(
        "--rewriter-provider",
        default=None,
        help="defaults to --oracle-provider when omitted",
    )
    p_label.set_defaults(func=_cmd_label)

    p_export = sub.add_parser("export", help="Convert labels + replay → SFT JSONL.")
    p_export.add_argument("--labels", required=True)
    p_export.add_argument("--replay-dir", default=None)
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--phase", choices=("extractor", "auditor", "both"), default="both")
    p_export.set_defaults(func=_cmd_export)

    p_rl = sub.add_parser(
        "rl-prompts",
        help="Emit prompt-only rows for online-RL sampling.",
    )
    p_rl.add_argument("--replay-dir", required=True)
    p_rl.add_argument("--out", required=True)
    p_rl.add_argument("--phase", choices=("extractor", "auditor", "both"), default="both")
    p_rl.set_defaults(func=_cmd_rl_prompts)

    p_oc = sub.add_parser(
        "annotate-case-outcome",
        help="Write case_outcome.json into one or more bundle dirs.",
    )
    p_oc.add_argument("--bundle", action="append", required=True)
    p_oc.add_argument(
        "--case-metadata",
        default=None,
        help="override path; only valid with a single --bundle",
    )
    p_oc.set_defaults(func=_cmd_annotate_case_outcome)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
