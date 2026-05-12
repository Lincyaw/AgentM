"""``llmharness-aggregate`` CLI: replay sidecar → per-case directory.

Walks ``<cwd>/.agentm/audit_replay/`` for every session, collects each
session into a :class:`CaseData`, and writes it under ``--out`` using
the canonical case layout.

Usage::

    llmharness-aggregate \\
        --cwd /path/to/run-dir \\
        --out ./cases

    # Aggregate a single session by id
    llmharness-aggregate \\
        --cwd /path/to/run-dir \\
        --root-session-id abc123 \\
        --out ./cases
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .collector import collect_case
from .writer import write_case


def _replay_dir(cwd: Path) -> Path:
    return cwd / ".agentm" / "audit_replay"


def _resolve_sessions(replay_dir: Path, root_session_id: str | None) -> list[Path]:
    if root_session_id:
        candidate = replay_dir / f"{root_session_id}.jsonl"
        return [candidate] if candidate.is_file() else []
    return sorted(replay_dir.glob("*.jsonl"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmharness-aggregate")
    parser.add_argument(
        "--cwd",
        required=True,
        help="Run directory containing .agentm/audit_replay/",
    )
    parser.add_argument("--out", required=True, help="Output base directory for cases/")
    parser.add_argument(
        "--root-session-id",
        default=None,
        help="Aggregate only this session; default = all sessions in --cwd",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help=(
            "Override case sample_id. Useful when the run did not mount "
            "llmharness.distill.binding (e.g. rca llm-eval runs). When "
            "aggregating multiple sessions in one invocation, applies to "
            "all of them — pair with --root-session-id for per-sample runs."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Override case dataset_name (paired with --sample-id).",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override case dataset_path (paired with --sample-id).",
    )

    args = parser.parse_args(argv)
    cwd = Path(args.cwd)
    out_dir = Path(args.out)

    replay_dir = _replay_dir(cwd)
    if not replay_dir.is_dir():
        print(
            f"no replay dir at {replay_dir}; nothing to aggregate",
            file=sys.stderr,
        )
        return 2

    sessions = _resolve_sessions(replay_dir, args.root_session_id)
    if not sessions:
        print(f"no replay sidecars matched in {replay_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    for replay_path in sessions:
        meta_path = replay_path.with_suffix(".meta.json")
        case = collect_case(
            replay_path=replay_path,
            meta_path=meta_path if meta_path.is_file() else None,
            sample_id_override=args.sample_id,
            dataset_name_override=args.dataset_name,
            dataset_path_override=args.dataset_path,
        )
        case_dir = write_case(case, out_dir)
        print(
            f"{replay_path.name} → {case_dir}/ "
            f"(ext={case.meta.extractor_firings} aud={case.meta.auditor_firings} "
            f"surfaced={case.meta.surfaced_reminders})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
