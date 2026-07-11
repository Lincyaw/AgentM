"""Experiment lifecycle: ID generation, output directory, result recording.

Unified experiment tracking: every eval run produces an experiment ID,
registers sessions, provides per-session artifact directories, and
supports trajectory export from ClickHouse.
"""

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

    # --- Session registration ---

    @property
    def sessions_path(self) -> Path:
        return self.output_dir / "sessions.jsonl"

    def session_dir(self, session_id: str) -> Path:
        return self.output_dir / "sessions" / session_id

    def register_session(
        self,
        session_id: str,
        *,
        case_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Register a session and return its artifact directory."""
        record: dict[str, Any] = {
            "session_id": session_id,
            "registered_at": datetime.now(UTC).isoformat(),
        }
        if case_id is not None:
            record["case_id"] = case_id
        if trace_id is not None:
            record["trace_id"] = trace_id
        if metadata:
            record.update(metadata)

        with open(self.sessions_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()

        sdir = self.session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        return sdir

    def load_sessions(self) -> list[dict[str, Any]]:
        if not self.sessions_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.sessions_path.read_text().splitlines()
            if line.strip()
        ]

    def session_ids(self) -> list[str]:
        return [s["session_id"] for s in self.load_sessions()]

    # --- Per-session artifact writing ---

    def write_session_artifact(
        self,
        session_id: str,
        category: str,
        filename: str,
        data: dict[str, Any] | str,
    ) -> Path:
        """Write a file into ``sessions/<session_id>/<category>/<filename>``."""
        if category:
            target_dir = self.session_dir(session_id) / category
        else:
            target_dir = self.session_dir(session_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / filename
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        return path

    # --- Trace session tracking ---

    @property
    def traces_path(self) -> Path:
        return self.output_dir / "traces.jsonl"

    def record_trace(
        self,
        trace_session_id: str,
        *,
        case_id: str,
        role: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a real ClickHouse session ID produced during evaluation.

        Every benchmark calls this for each session it spawns (index
        extraction, auditor, TEL agent, etc.).  The ``role`` field
        distinguishes session types (e.g. ``"index"``, ``"auditor"``,
        ``"tel_agent"``).  ``case_id`` is the trajectory / instance ID
        this session belongs to.
        """
        record: dict[str, Any] = {
            "trace_session_id": trace_session_id,
            "case_id": case_id,
            "role": role,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        if metadata:
            record.update(metadata)
        with open(self.traces_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()

    def load_traces(self) -> list[dict[str, Any]]:
        if not self.traces_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.traces_path.read_text().splitlines()
            if line.strip()
        ]

    def trace_session_ids(
        self, *, role: str | None = None, case_id: str | None = None,
    ) -> list[str]:
        """Return recorded trace session IDs, optionally filtered."""
        traces = self.load_traces()
        if role:
            traces = [t for t in traces if t.get("role") == role]
        if case_id:
            traces = [t for t in traces if t.get("case_id") == case_id]
        seen: set[str] = set()
        out: list[str] = []
        for t in traces:
            sid = t["trace_session_id"]
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
        return out

    # --- Trajectory export ---

    def export_trajectory(
        self,
        trace_session_id: str,
        *,
        fmt: str = "ndjson",
        out_dir: Path | None = None,
    ) -> Path | None:
        """Export a single session's trajectory from ClickHouse."""
        target = out_dir or self.output_dir / "trajectories"
        target.mkdir(parents=True, exist_ok=True)
        out_path = target / f"trajectory_{trace_session_id}.{fmt}"
        if out_path.is_file():
            return out_path
        try:
            from agentm.core.observability import clickhouse

            url = clickhouse.get_url()
            if not url:
                logger.debug("trajectory export: ClickHouse unavailable")
                return None
            entries = clickhouse.session_entries(url, trace_session_id)
            if not entries:
                logger.debug("trajectory export empty for {}", trace_session_id)
                return None
            lines = [json.dumps(e, ensure_ascii=False, default=str) for e in entries]
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return out_path
        except Exception as exc:
            logger.debug("trajectory export error for {}: {}", trace_session_id, exc)
            return None

    def export_all_trajectories(
        self, *, fmt: str = "ndjson", role: str | None = None,
    ) -> int:
        """Export trajectories for all recorded trace sessions.

        Uses ``traces.jsonl`` written by ``record_trace()`` during eval.
        Optionally filter by ``role`` (e.g. ``"auditor"``).
        """
        sids = self.trace_session_ids(role=role)
        if not sids:
            logger.info("export: no trace sessions found for {}", self.exp_id)
            return 0
        out_dir = self.output_dir / "trajectories"
        exported = 0
        for sid in sids:
            if self.export_trajectory(sid, fmt=fmt, out_dir=out_dir) is not None:
                exported += 1
        logger.info(
            "exported {}/{} trajectories for {}",
            exported, len(sids), self.exp_id,
        )
        return exported

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
