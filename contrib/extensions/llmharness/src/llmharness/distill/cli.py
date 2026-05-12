"""``llmharness-distill`` CLI: label and export.

Two subcommands:

* ``label`` — walks a replay-log directory, joins each session to its
  meta sidecar + dataset row, and runs the two-stage oracle/rewriter
  per auditor record. Writes one ``<root_session_id>.labels.jsonl`` per
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

from ..replay.record import iter_records
from .binding import read_sample_meta
from .export import (
    auditor_records_from_labels,
    dropped_records_from_labels,
    extractor_records_from_replay,
    write_jsonl,
)
from .gt import GroundTruth, load_dataset
from .oracle import label_auditor_record

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
            "root_session_id": r.root_session_id,
            "ts_ns": r.ts_ns,
            "compose_kwargs": r.compose_kwargs,
            "payload": r.payload,
            "provider": r.provider,
            "output": r.output,
            "status": r.status,
            "error": r.error,
            "latency_ms": r.latency_ms,
            "extras": r.extras,
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
    root_session_id = replay_file.stem
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

    out_path = out_dir / f"{root_session_id}.labels.jsonl"
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
        n_aud = write_jsonl(
            out_dir / "auditor.jsonl", auditor_records_from_labels(label_rows)
        )
        n_drop = write_jsonl(
            out_dir / "dropped.jsonl", dropped_records_from_labels(label_rows)
        )

    print(f"extractor={n_ext} auditor={n_aud} dropped={n_drop} out={out_dir}")
    return 0


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
    p_export.add_argument(
        "--phase", choices=("extractor", "auditor", "both"), default="both"
    )
    p_export.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
