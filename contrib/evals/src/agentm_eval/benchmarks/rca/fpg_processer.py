"""FPG-based processer for RCA evaluation.

Replaces rcabench-platform's ``RCABenchProcesser.judge_one`` with
fpg ``compare_model_to_ground_truth`` scoring. The agent emits fpg
``ModelRCAOutput`` (root_causes = node-id strings), which is
incompatible with rcabench-platform's ``AgentRCAOutput`` (root_causes =
RootCauseClaim objects). This processer bridges the gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from rcabench_platform.v3.sdk.llm_eval.eval.data import EvaluationSample
from rcabench_platform.v3.sdk.llm_eval.eval.processer.rcabench import (
    RCABenchProcesser,
)


class FpgProcesser(RCABenchProcesser):
    name: str = "fpg"

    async def judge_one(self, sample: EvaluationSample) -> EvaluationSample:
        meta = dict(sample.meta) if isinstance(sample.meta, dict) else {}
        case_dir = self._resolve_case_dir(meta, sample)

        if not case_dir:
            sample.update(
                correct=False,
                confidence=0.0,
                eval_metrics={"error": "missing case dir"},
                meta=meta,
            )
            return sample

        response_text = sample.response or ""
        try:
            payload = json.loads(response_text)
        except (TypeError, ValueError):
            sample.update(
                correct=False,
                confidence=0.0,
                eval_metrics={"parse_error": "response is not valid JSON"},
                meta=meta,
            )
            return sample

        if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
            sample.update(
                correct=False,
                confidence=0.0,
                eval_metrics={"parse_error": "response missing nodes[]"},
                meta=meta,
            )
            return sample

        graph_path = case_dir / "causal_graph_verified.json"
        if not graph_path.is_file():
            graph_path = case_dir / "causal_graph.json"
        if not graph_path.is_file():
            sample.update(
                correct=False,
                confidence=0.0,
                eval_metrics={"error": f"no GT graph in {case_dir}"},
                meta=meta,
            )
            return sample

        try:
            from fpg import ModelRCAOutput, Scenario, compare_model_to_ground_truth
        except ImportError:
            sample.update(
                correct=False,
                confidence=0.0,
                eval_metrics={"error": "fpg package not installed"},
                meta=meta,
            )
            return sample

        try:
            model_output = ModelRCAOutput.model_validate(payload)
            scenario = Scenario.model_validate_json(
                graph_path.read_text(encoding="utf-8")
            )
            comparison = compare_model_to_ground_truth(model_output, scenario)
        except Exception as exc:
            logger.warning("fpg comparison failed for {}: {}", sample.source, exc)
            sample.update(
                correct=False,
                confidence=0.0,
                eval_metrics={"error": f"fpg comparison failed: {exc}"},
                meta=meta,
            )
            return sample

        result = comparison.model_dump(mode="json")
        score = float(result.get("score", 0.0))

        sql_eval = _evaluate_sql_evidence(payload, case_dir)
        if sql_eval:
            result["sql_evidence"] = sql_eval

        sample.update(
            correct=score >= 0.999,
            confidence=score,
            eval_metrics=result,
            meta=meta,
        )
        return sample


def _evaluate_sql_evidence(
    payload: dict[str, Any], case_dir: Path
) -> dict[str, Any] | None:
    """Replay fpg evidence SQL against case parquets."""
    statements: list[dict[str, Any]] = []
    for node in payload.get("nodes", []):
        if not isinstance(node, dict):
            continue
        for i, ev in enumerate(node.get("evidence") or []):
            if not isinstance(ev, dict):
                continue
            query = ev.get("query")
            if not isinstance(query, dict):
                continue
            if query.get("language", "").lower() != "sql":
                continue
            stmt = query.get("statement", "").strip()
            if stmt:
                statements.append({"index": i, "node_id": node.get("id", ""), "sql": stmt})

    if not statements:
        return {"total": 0, "executable": 0, "ratio": 1.0}

    try:
        import duckdb
        from agentm.core.lib import cap_duckdb_threads
    except ImportError:
        return {"total": len(statements), "executable": 0, "ratio": 0.0}

    try:
        conn = duckdb.connect(":memory:")
        cap_duckdb_threads(conn)
        for parquet in sorted(case_dir.iterdir()):
            if parquet.suffix != ".parquet" or parquet.name == "conclusion.parquet":
                continue
            path = parquet.as_posix().replace("'", "''")
            conn.execute(
                f"CREATE OR REPLACE VIEW {parquet.stem} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
    except Exception as exc:
        logger.debug("sql eval setup failed: {}", exc)
        return {"total": len(statements), "executable": 0, "ratio": 0.0}

    executable = 0
    try:
        for item in statements:
            sql = item["sql"].rstrip(";").strip()
            if ";" in sql:
                continue
            try:
                conn.execute(f"SELECT * FROM ({sql}) LIMIT 1").fetchone()
                executable += 1
            except duckdb.Error:
                pass
    finally:
        conn.close()

    return {
        "total": len(statements),
        "executable": executable,
        "ratio": executable / len(statements) if statements else 0.0,
    }
