"""Summarize replay-fork JSONL results.

This is intentionally file-only: it does not query ClickHouse and does not call
models. Use it after a replay-fork batch to inspect correction rate and the
root-cause deltas written into each result row.

Example:

    uv run python -m rca_eval.replay_fork.summarize \
      --cases runs/rca-fork-reminder-allwrong-31/cases.tsv \
      runs/rca-fork-reminder-auto-31/single-surface.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseInfo:
    dataset_case_id: str
    status: str
    success_session_id: str
    notes: str


def _read_cases(path: Path | None) -> dict[str, CaseInfo]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        cases: dict[str, CaseInfo] = {}
        for row in reader:
            baseline = (row.get("baseline_session_id") or "").strip()
            if not baseline:
                continue
            cases[baseline] = CaseInfo(
                dataset_case_id=(row.get("case_id") or "").strip(),
                status=(row.get("status") or "").strip(),
                success_session_id=(row.get("success_session_id") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
        return cases


def _read_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no}: expected object row")
        rows.append(payload)
    return rows


def _is_true(row: dict[str, Any], key: str) -> bool:
    return row.get(key) is True


def _root_labels(summary: Any) -> str:
    if not isinstance(summary, dict):
        return "-"
    parse_error = summary.get("parse_error")
    if parse_error:
        return f"parse_error:{parse_error}"
    roots = summary.get("root_causes")
    if not isinstance(roots, list):
        return "-"
    if not roots:
        return "<empty>"
    labels: list[str] = []
    for root in roots:
        if not isinstance(root, dict):
            continue
        subject = root.get("subject")
        predicate = root.get("predicate")
        root_id = root.get("id")
        if isinstance(subject, str) and isinstance(predicate, str):
            labels.append(f"{subject}|{predicate}")
        elif isinstance(subject, str):
            labels.append(subject)
        elif isinstance(root_id, str):
            labels.append(root_id)
    return ", ".join(labels) if labels else "-"


def _truncate(text: str, width: int) -> str:
    if width <= 3 or len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    helped = sum(
        1
        for row in rows
        if row.get("control_correct") is False and row.get("intervene_correct") is True
    )
    harmed = sum(
        1
        for row in rows
        if row.get("control_correct") is True and row.get("intervene_correct") is False
    )
    return {
        "completed": len(rows),
        "fired": sum(1 for row in rows if _is_true(row, "fired")),
        "errored": sum(1 for row in rows if row.get("error")),
        "control_correct": sum(1 for row in rows if _is_true(row, "control_correct")),
        "intervene_correct": sum(1 for row in rows if _is_true(row, "intervene_correct")),
        "helped": helped,
        "harmed": harmed,
    }


def summarize(results_path: Path, *, cases_path: Path | None = None) -> str:
    rows = _read_results(results_path)
    cases = _read_cases(cases_path)
    counts = _counts(rows)
    planned = len(cases) if cases else counts["completed"]
    remaining = max(0, planned - counts["completed"])

    lines = [
        "=== replay-fork results ===",
        f"results: {results_path}",
        f"planned={planned} completed={counts['completed']} remaining={remaining}",
        (
            f"fired={counts['fired']} errored={counts['errored']} "
            f"control_correct={counts['control_correct']} "
            f"intervene_correct={counts['intervene_correct']} "
            f"helped={counts['helped']} harmed={counts['harmed']}"
        ),
        "",
        "case\tbaseline\tctrl\tiv\tfire\tsurface\tcontrol_roots\tintervene_roots",
    ]
    for row in rows:
        baseline = str(row.get("case_id") or "")
        info = cases.get(baseline)
        label = info.dataset_case_id if info else baseline
        control = "Y" if row.get("control_correct") is True else "N" if row.get("control_correct") is False else "-"
        intervene = "Y" if row.get("intervene_correct") is True else "N" if row.get("intervene_correct") is False else "-"
        fired = "Y" if row.get("fired") is True else "N"
        surface = "-" if row.get("surface_turn") is None else str(row.get("surface_turn"))
        control_roots = _truncate(
            _root_labels(row.get("control_submission_summary")),
            96,
        )
        intervene_roots = _truncate(
            _root_labels(row.get("intervene_submission_summary")),
            96,
        )
        lines.append(
            "\t".join(
                [
                    label,
                    baseline,
                    control,
                    intervene,
                    fired,
                    surface,
                    control_roots,
                    intervene_roots,
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="replay-fork JSONL result file")
    parser.add_argument("--cases", type=Path, help="optional prior case TSV")
    args = parser.parse_args()
    print(summarize(args.results, cases_path=args.cases))


if __name__ == "__main__":
    main()
