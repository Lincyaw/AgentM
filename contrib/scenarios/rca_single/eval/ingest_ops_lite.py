"""Ingest the anon-ops/ops-lite Hugging Face dataset into the rca llm-eval DB.

Walks ``datasets/ops-lite/cases/*`` and inserts a ``DatasetSample`` row per
case under ``dataset='ops-lite'`` with ``tags=['ops-lite']``. The agent's
``query_sql`` resolves parquets via the case dir; ``causal_graph.json`` is
required by the RCABench preprocer. Idempotent — re-running only inserts
missing rows.

Run:
    uv run python contrib/scenarios/rca_single/eval/ingest_ops_lite.py
"""

from __future__ import annotations

from pathlib import Path

DATASET_NAME = "ops-lite"
CASES_DIR = Path("datasets/ops-lite/cases").resolve()


def main() -> None:
    # Optional deps are imported lazily so the surrounding rca_single
    # scenario stays importable for tooling that walks the contrib tree on
    # machines where rcabench-platform / sqlmodel are absent. Surface a
    # single actionable SystemExit instead of a bare ``ImportError`` at
    # module load.
    try:
        from rcabench_platform.v3.sdk.llm_eval.db import DatasetSample
        from rcabench_platform.v3.sdk.llm_eval.utils.sqlmodel_utils import (
            SQLModelUtils,
        )
        from sqlmodel import select
    except ImportError as exc:
        raise SystemExit(
            "ingest_ops_lite.py needs the rcabench-platform + sqlmodel "
            "packages installed. Run this script from a venv where "
            "`uv run rca llm-eval` works (typically the rca-autorl "
            f"workspace). Original error: {exc}"
        ) from exc

    if not CASES_DIR.is_dir():
        raise SystemExit(f"missing dataset dir: {CASES_DIR}")

    case_dirs = sorted(p for p in CASES_DIR.iterdir() if p.is_dir())
    print(f"found {len(case_dirs)} case dirs under {CASES_DIR}")

    skipped_no_cg = 0
    skipped_existing = 0
    inserted = 0

    with SQLModelUtils.create_session() as session:
        existing = {
            row.source
            for row in session.exec(
                select(DatasetSample).where(DatasetSample.dataset == DATASET_NAME)
            ).all()
        }
        print(f"already in DB: {len(existing)}")

        new_rows: list[DatasetSample] = []
        for idx, case_dir in enumerate(case_dirs, start=1):
            name = case_dir.name
            if name in existing:
                skipped_existing += 1
                continue
            if not (case_dir / "causal_graph.json").exists():
                skipped_no_cg += 1
                continue

            row = DatasetSample(
                dataset=DATASET_NAME,
                index=idx,
                source=name,
                question="",  # rendered from augmentation template at preprocess time
                answer="",  # GT comes from injection.json at judge time, not this column
                tags=[DATASET_NAME],
                meta={"path": str(case_dir.absolute())},
            )
            new_rows.append(row)

        if new_rows:
            session.add_all(new_rows)
            session.commit()
            inserted = len(new_rows)

    print(
        f"inserted={inserted}  skipped_existing={skipped_existing}  "
        f"skipped_no_cg={skipped_no_cg}"
    )


if __name__ == "__main__":
    main()
