"""Seed ``DatasetSample`` rows from a local ``data.jsonl`` file.

``rcabench-platform``'s llm-eval pipeline pulls samples from a SQL
``data`` table keyed by ``(dataset, source)``. Our dataset ships as a
JSONL file under ``/home/ddq/AoyangSpace/dataset/rca``, so we one-shot it
into the configured DB before the first eval run. Idempotent — re-runs
update existing rows in place rather than duplicating.

Invocation::

    uv run python -m rca_eval.seed_dataset \\
        --jsonl /home/ddq/AoyangSpace/dataset/rca/data.jsonl \\
        --dataset rca-openrca2-lite

The DB URL is read from ``LLM_EVAL_DB_URL`` (or ``UTU_DB_URL``) — set
either via ``.env`` or the shell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _walk_dotenv() -> None:
    """Mirror the ``rca llm-eval run`` lookup so seeding sees the same DB."""
    cwd = Path.cwd()
    for d in (cwd, *cwd.parents):
        candidate = d / ".env"
        if candidate.is_file():
            load_dotenv(str(candidate))
            return
    load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        required=True,
        type=Path,
        help="Path to data.jsonl",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset label written to DatasetSample.dataset",
    )
    args = parser.parse_args()

    _walk_dotenv()

    from rcabench_platform.v3.sdk.llm_eval.db import DatasetSample
    from rcabench_platform.v3.sdk.llm_eval.utils import SQLModelUtils
    from sqlmodel import SQLModel, select

    SQLModel.metadata.create_all(SQLModelUtils.get_engine())

    rows: list[dict[str, Any]] = []
    with args.jsonl.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(
                    f"line {line_no}: skipping malformed json ({exc})",
                    file=sys.stderr,
                )

    inserted = 0
    updated = 0
    with SQLModelUtils.create_session() as session:
        for idx, row in enumerate(rows, start=1):
            source = str(row.get("source", "")).strip()
            if not source:
                continue
            existing = session.exec(
                select(DatasetSample).where(
                    DatasetSample.dataset == args.dataset,
                    DatasetSample.source == source,
                )
            ).first()
            payload = {
                "dataset": args.dataset,
                "index": idx,
                "source": source,
                "source_index": row.get("id"),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "tags": row.get("tags") or [],
                "meta": {
                    "ground_truth": row.get("ground_truth"),
                    "fault_type": row.get("fault_type"),
                    "fault_category": row.get("fault_category"),
                    "difficulty_spl": row.get("difficulty_spl"),
                    "difficulty_n_svc": row.get("difficulty_n_svc"),
                    "difficulty_n_edge": row.get("difficulty_n_edge"),
                    "datapack_name": row.get("datapack_name"),
                },
            }
            if existing is None:
                session.add(DatasetSample(**payload))
                inserted += 1
            else:
                for key, value in payload.items():
                    setattr(existing, key, value)
                updated += 1
        session.commit()

    print(
        f"dataset={args.dataset!r}  inserted={inserted}  updated={updated}  "
        f"total_rows={len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
