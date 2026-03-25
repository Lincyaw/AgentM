#!/usr/bin/env python3
"""Export judged evaluation trajectories from PostgreSQL to local JSON files.

Usage:
    python scripts/export_eval_trajectories.py --exp-id agentm-v11 --output-dir ./eval-trajectories
    python scripts/export_eval_trajectories.py --correct false --limit 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_URL = os.environ.get("LLM_EVAL_DB_URL")
DEFAULT_OUTPUT_DIR = "./eval-trajectories"


def build_query(args: argparse.Namespace) -> tuple[str, list]:
    conditions = ["stage = 'judged'", "trajectories IS NOT NULL"]
    params: list = []
    idx = 1

    if args.exp_id:
        conditions.append(f"exp_id = ${idx}")
        params.append(args.exp_id)
        idx += 1

    if args.correct in ("true", "false"):
        conditions.append(f"correct = ${idx}")
        params.append(args.correct == "true")
        idx += 1

    if args.agent_type:
        conditions.append(f"agent_type = ${idx}")
        params.append(args.agent_type)
        idx += 1

    where_clause = " AND ".join(conditions)

    # psycopg2 uses %s placeholders, not $N
    query = f"""
        SELECT id, exp_id, dataset_index, correct, correct_answer,
               extracted_final_answer, agent_type, model_name, reasoning,
               trajectories
        FROM evaluation_data
        WHERE {where_clause}
        ORDER BY id
    """

    # Replace $N placeholders with %s for psycopg2
    for i in range(len(params), 0, -1):
        query = query.replace(f"${i}", "%s")

    if args.limit:
        query += " LIMIT %s"
        params.append(args.limit)

    return query, params


def export_row(row: dict, output_dir: Path) -> str:
    """Export a single row to a JSON file. Returns the filename."""
    row_id = row["id"]
    exp_id = row["exp_id"]
    correct = row["correct"]
    label = "correct" if correct else "incorrect"
    filename = f"{exp_id}_{row_id}_{label}.json"

    # Build eval metadata
    eval_meta = {
        "id": row_id,
        "exp_id": exp_id,
        "dataset_index": row["dataset_index"],
        "correct": correct,
        "correct_answer": row["correct_answer"],
        "extracted_final_answer": row["extracted_final_answer"],
        "agent_type": row["agent_type"],
        "model_name": row["model_name"],
        "reasoning": row["reasoning"],
    }

    # Parse trajectories column (JSON string like {"trajectories": [...]})
    raw_trajectories = row["trajectories"]
    if isinstance(raw_trajectories, str):
        trajectory_data = json.loads(raw_trajectories)
    else:
        # Already parsed (e.g. jsonb column)
        trajectory_data = raw_trajectories

    # Merge: _eval_meta + trajectory contents at the same level
    output = {"_eval_meta": eval_meta}
    if isinstance(trajectory_data, dict):
        output.update(trajectory_data)
    else:
        # Unexpected format, store as-is
        output["trajectories"] = trajectory_data

    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export judged evaluation trajectories to local JSON files."
    )
    parser.add_argument(
        "--exp-id", type=str, default=None, help="Filter by experiment ID"
    )
    parser.add_argument(
        "--correct",
        type=str,
        choices=["true", "false", "all"],
        default="all",
        help="Filter by correctness (default: all)",
    )
    parser.add_argument(
        "--agent-type", type=str, default=None, help="Filter by agent type"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of rows to export"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    db_url = os.environ.get("LLM_EVAL_DB_URL", DEFAULT_DB_URL)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import psycopg2 here so the script fails with a clear message
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print(
            "ERROR: psycopg2 is required. Install with: uv add psycopg2-binary",
            file=sys.stderr,
        )
        sys.exit(1)

    query, params = build_query(args)

    print(f"Connecting to database...")
    print(f"Output directory: {output_dir.resolve()}")

    try:
        conn = psycopg2.connect(db_url)
    except psycopg2.Error as e:
        print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    except psycopg2.Error as e:
        print(f"ERROR: Query failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()

    if not rows:
        print("No matching rows found.")
        return

    print(f"Found {len(rows)} rows. Exporting...")

    exported = 0
    errors = 0
    for row in rows:
        try:
            filename = export_row(dict(row), output_dir)
            exported += 1
        except Exception as e:
            errors += 1
            print(f"  ERROR exporting row {row.get('id', '?')}: {e}", file=sys.stderr)

    # Summary
    print()
    print("=" * 50)
    print(f"Export complete.")
    print(f"  Total rows found : {len(rows)}")
    print(f"  Files exported   : {exported}")
    if errors:
        print(f"  Errors           : {errors}")
    print(f"  Output directory : {output_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
