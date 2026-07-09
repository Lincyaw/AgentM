"""Experiment lifecycle: ID generation, output directory, result recording."""

from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from loguru import logger


def _default_output_root() -> Path:
    home = os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm"))
    return Path(home) / "eval_runs"


def generate_exp_id(benchmark: str, model: str | None = None) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", benchmark.lower()).strip("-")
    short = uuid.uuid4().hex[:4]
    parts = [slug]
    if model:
        m = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:16]
        parts.append(m)
    parts.extend([ts, short])
    return "-".join(parts)


@dataclass
class Experiment:
    exp_id: str
    benchmark: str
    output_dir: Path
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    _results_file: Any = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        benchmark: str,
        *,
        model: str | None = None,
        exp_id: str | None = None,
        output_root: Path | None = None,
        **params: Any,
    ) -> Experiment:
        eid = exp_id or generate_exp_id(benchmark, model)
        root = output_root or _default_output_root()
        out = root / eid
        out.mkdir(parents=True, exist_ok=True)
        (out / "artifacts").mkdir(exist_ok=True)

        exp = cls(
            exp_id=eid,
            benchmark=benchmark,
            output_dir=out,
            model=model,
            params=params,
        )
        exp._write_meta("running")
        logger.info("Experiment {} → {}", eid, out)
        return exp

    @classmethod
    def load(cls, exp_id: str, output_root: Path | None = None) -> Experiment:
        root = output_root or _default_output_root()
        out = root / exp_id
        meta_file = out / "meta.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"No experiment found: {exp_id}")
        meta = json.loads(meta_file.read_text())
        return cls(
            exp_id=exp_id,
            benchmark=meta["benchmark"],
            output_dir=out,
            model=meta.get("model"),
            params=meta.get("params", {}),
            start_time=datetime.fromisoformat(meta["start_time"]),
        )

    @property
    def artifacts_dir(self) -> Path:
        return self.output_dir / "artifacts"

    @property
    def results_path(self) -> Path:
        return self.output_dir / "results.jsonl"

    def record_result(self, result: dict[str, Any]) -> None:
        with open(self.results_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            f.flush()

    def load_results(self) -> list[dict[str, Any]]:
        if not self.results_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.results_path.read_text().splitlines()
            if line.strip()
        ]

    def session_config_overrides(self, task_id: str | None = None) -> dict[str, Any]:
        overrides: dict[str, Any] = {
            "eval_run_id": self.exp_id,
            "task_class": self.benchmark,
        }
        if task_id:
            overrides["eval_task_id"] = task_id
        return overrides

    def finish(
        self,
        status: str = "completed",
        summary: dict[str, Any] | None = None,
    ) -> None:
        self._write_meta(status, summary)
        report_file = self.output_dir / "report.txt"
        if summary:
            report_file.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str)
            )
        logger.info("Experiment {} finished: {}", self.exp_id, status)

    def _write_meta(
        self, status: str, summary: dict[str, Any] | None = None
    ) -> None:
        meta: dict[str, Any] = {
            "exp_id": self.exp_id,
            "benchmark": self.benchmark,
            "model": self.model,
            "params": self.params,
            "start_time": self.start_time.isoformat(),
            "status": status,
        }
        if status != "running":
            meta["end_time"] = datetime.now(UTC).isoformat()
        if summary:
            meta["summary"] = summary
        (self.output_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str)
        )


@contextmanager
def experiment_context(
    benchmark: str,
    *,
    model: str | None = None,
    exp_id: str | None = None,
    output_root: Path | None = None,
    **params: Any,
) -> Iterator[Experiment]:
    exp = Experiment.create(
        benchmark, model=model, exp_id=exp_id, output_root=output_root, **params,
    )
    try:
        yield exp
    except Exception as e:
        exp.finish(status="failed", summary={"error": str(e)})
        raise
    else:
        exp.finish(status="completed")
