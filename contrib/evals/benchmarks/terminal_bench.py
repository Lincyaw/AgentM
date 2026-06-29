"""Terminal Bench 1.0 adapter: Dockerfile + task.yaml + INSTRUCTION.md."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .base import TaskSpec, image_name, upload_file_to_sandbox

try:
    import yaml as _yaml
except ImportError:  # noqa: S110
    _yaml = None  # type: ignore[assignment]


class TerminalBenchAdapter:
    """Original Terminal Bench format: Dockerfile + task.yaml + INSTRUCTION.md."""

    def discover_tasks(self, source: str) -> list[TaskSpec]:
        repo = Path(source).expanduser().resolve()
        tasks: list[TaskSpec] = []
        for task_dir in sorted(repo.iterdir()):
            if not task_dir.is_dir():
                continue
            dockerfile = task_dir / "Dockerfile"
            if not dockerfile.is_file():
                continue
            has_from = any(
                line.strip().startswith("FROM")
                for line in dockerfile.read_text().splitlines()
            )
            if not has_from:
                continue

            prompt = ""
            difficulty = ""
            category = ""
            task_yaml = task_dir / "task.yaml"
            if task_yaml.is_file() and _yaml is not None:
                raw = _yaml.safe_load(task_yaml.read_text())
                if isinstance(raw, dict):
                    prompt = raw.get("instruction", "")
                    difficulty = raw.get("difficulty", "")
                    category = raw.get("category", "")

            instruction_file = task_dir / "INSTRUCTION.md"
            if not prompt and instruction_file.is_file():
                prompt = instruction_file.read_text().strip()

            tasks.append(TaskSpec(
                name=task_dir.name,
                prompt=prompt,
                path=str(task_dir),
                difficulty=difficulty,
                category=category,
            ))
        return tasks

    def get_image(self, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
        return image_name(task.name, registry, prefix, tag)

    def get_source_image(self, task: TaskSpec) -> str | None:
        return None

    def supports_build(self) -> bool:
        return True

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        task_dir = Path(task.path)
        folder = _detect_project_folder(task_dir)

        _upload_tests(session, task_dir)

        session.execute([{  # type: ignore[attr-defined]
            "name": "cp-tests",
            "command": ["bash", "-lc", f"cp -a /tests/. /app/{folder}/ 2>/dev/null || true"],
            "work_dir": "/app",
        }])

        r = session.execute([{  # type: ignore[attr-defined]
            "name": "eval",
            "command": ["bash", "-lc",
                f"export TEST_DIR=/tests && cd /app && "
                f"killall qemu-system-riscv64 2>/dev/null; "
                f"timeout {timeout} bash /tests/run-tests.sh 2>&1"],
            "work_dir": "/app",
        }])
        eval_out = r.results[0].output.stdout

        session.execute([{  # type: ignore[attr-defined]
            "name": "collect-test-output",
            "command": ["bash", "-lc",
                "mkdir -p /workspace/test_output && "
                "cp -a /app/test_output/. /workspace/test_output/ 2>/dev/null || true"],
            "work_dir": "/app",
        }])

        scores = _parse_scores(session, eval_out)
        scores["eval_output"] = eval_out or ""
        return scores

    def is_pass(self, result: dict) -> bool:
        f2p = result.get("f2p")
        p2p = result.get("p2p")
        f2p_ok = isinstance(f2p, dict) and isinstance(f2p.get("step_score"), (int, float)) and f2p["step_score"] >= 1.0
        p2p_ok = isinstance(p2p, dict) and p2p.get("total", 0) > 0 and p2p.get("passed") == p2p.get("total")
        return f2p_ok and p2p_ok

    def format_score_line(self, r: dict) -> str:
        name = r.get("task", "?")
        tools = r.get("tools", "?")
        status = r.get("status", "?").upper()
        f2p = r.get("f2p") or {}
        step = f2p.get("step_score", "-") if isinstance(f2p, dict) else "-"
        if isinstance(step, float):
            step = f"{step:.0%}"
        line = f"  [{status}] {name} tools={tools} f2p={step}"
        if status != "DONE" and r.get("error"):
            error = " ".join(str(r["error"]).split())
            if len(error) > 180:
                error = error[:177] + "..."
            line += f" error={error}"
        return line

    def summary_header(self) -> str:
        return (
            f"  {'Task':<25} {'F2P pass':<10} {'F2P step':<12} {'P2P':<12}\n"
            f"  {'-' * 70}"
        )

    def summary_row(self, name: str, r: dict) -> str:
        f2p = r.get("f2p") or {}
        p2p = r.get("p2p") or {}
        f2p_pass = f2p.get("is_pass", "-") if isinstance(f2p, dict) else "-"
        f2p_step = f2p.get("step_score", "-") if isinstance(f2p, dict) else "-"
        if isinstance(f2p_step, float):
            f2p_step = f"{f2p_step:.1%}"
        p2p_str = (
            f"{p2p['passed']}/{p2p['total']}"
            if isinstance(p2p, dict) and p2p.get("total")
            else "-"
        )
        return f"  {name:<25} {str(f2p_pass):<10} {str(f2p_step):<12} {p2p_str:<12}"

    def summary_footer(self, results: dict[str, dict]) -> str:
        return ""

    def pass_at_k_header(self) -> str:
        return (
            f"  {'Task':<25} {'Best F2P':<10} {'F2P full':<10} "
            f"{'Avg F2P step':<14} {'Avg P2P step'}\n"
            f"  {'-' * 70}"
        )

    def pass_at_k_row(self, name: str, runs: list[dict]) -> tuple[str, dict]:
        f2p_steps: list[float] = []
        p2p_steps: list[float] = []
        any_f2p_full = False

        for r in runs:
            f2p = r.get("f2p")
            p2p = r.get("p2p")
            f2p_s = f2p.get("step_score") if isinstance(f2p, dict) else None
            if isinstance(f2p_s, (int, float)):
                f2p_steps.append(f2p_s)
                if f2p_s >= 1.0:
                    any_f2p_full = True
            p2p_total = p2p.get("total", 0) if isinstance(p2p, dict) else 0
            p2p_passed = p2p.get("passed", 0) if isinstance(p2p, dict) else 0
            if p2p_total > 0:
                p2p_steps.append(p2p_passed / p2p_total)

        best_f2p = max(f2p_steps) if f2p_steps else None
        avg_f2p = sum(f2p_steps) / len(f2p_steps) if f2p_steps else None
        avg_p2p = sum(p2p_steps) / len(p2p_steps) if p2p_steps else None

        best_str = f"{best_f2p:.1%}" if best_f2p is not None else "-"
        pass_str = "YES" if any_f2p_full else "no"
        avg_f2p_str = f"{avg_f2p:.1%}" if avg_f2p is not None else "-"
        avg_p2p_str = f"{avg_p2p:.1%}" if avg_p2p is not None else "-"

        line = f"  {name:<25} {best_str:<10} {pass_str:<10} {avg_f2p_str:<14} {avg_p2p_str}"
        stats = {
            "pass_at_k": best_f2p,
            "any_pass": any_f2p_full,
            "best_f2p_step": best_f2p,
            "avg_f2p_step": avg_f2p,
            "avg_p2p_step": avg_p2p,
        }
        return line, stats

    def pass_at_k_footer(self, all_stats: list[dict], n_tasks: int) -> str:
        pass_values = [s["pass_at_k"] for s in all_stats if s.get("pass_at_k") is not None]
        full_count = sum(1 for s in all_stats if s.get("any_pass"))
        f2p_avgs = [s["avg_f2p_step"] for s in all_stats if s.get("avg_f2p_step") is not None]
        p2p_avgs = [s["avg_p2p_step"] for s in all_stats if s.get("avg_p2p_step") is not None]
        lines = []
        if pass_values:
            lines.append(
                f"\n  Overall pass@k F2P: {sum(pass_values) / n_tasks:.1%}"
            )
        lines.append(f"  Tasks with full F2P pass: {full_count}/{n_tasks}")
        if f2p_avgs:
            lines.append(f"  Avg F2P Step Score:  {sum(f2p_avgs) / len(f2p_avgs):.1%}")
        if p2p_avgs:
            lines.append(f"  Avg P2P Step Score:  {sum(p2p_avgs) / len(p2p_avgs):.1%}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_project_folder(task_dir: Path) -> str:
    task_yaml = task_dir / "task.yaml"
    if task_yaml.is_file() and _yaml is not None:
        raw = _yaml.safe_load(task_yaml.read_text())
        instruction = raw.get("instruction", "") if isinstance(raw, dict) else ""
        m = re.search(r"in folder (\S+)", instruction)
        if m:
            return m.group(1).rstrip(".")
    return task_dir.name.split("_", 1)[-1]


def _upload_tests(session: object, task_dir: Path) -> None:
    """Upload test files to /tests/ via staging dir."""
    tests_dir = task_dir / "tests"
    if not tests_dir.is_dir():
        return

    session.execute([{  # type: ignore[attr-defined]
        "name": "prep",
        "command": ["bash", "-lc", "mkdir -p /tests /app/test_output /app/_eval_staging"],
        "work_dir": "/app",
    }])

    for f in tests_dir.rglob("*"):
        if not f.is_file():
            continue
        rel = str(f.relative_to(tests_dir))
        staging_rel = f"_eval_staging/{rel}"
        upload_file_to_sandbox(session, staging_rel, f.read_bytes())

    session.execute([{  # type: ignore[attr-defined]
        "name": "mv-tests",
        "command": ["bash", "-lc",
            'cd /app/_eval_staging && find . -type f | while read f; do '
            'mkdir -p "/tests/$(dirname "$f")" && '
            'mv "$f" "/tests/$f" && chmod +x "/tests/$f"; done'],
        "work_dir": "/app",
    }])

    run_tests = task_dir / "run-tests.sh"
    if run_tests.is_file():
        upload_file_to_sandbox(session, "_eval_staging/run-tests.sh", run_tests.read_bytes())
        session.execute([{  # type: ignore[attr-defined]
            "name": "mv-rt",
            "command": ["bash", "-lc",
                "mv /app/_eval_staging/run-tests.sh /tests/run-tests.sh && "
                "chmod +x /tests/run-tests.sh"],
            "work_dir": "/app",
        }])

    session.execute([{  # type: ignore[attr-defined]
        "name": "cleanup",
        "command": ["bash", "-lc", "rm -rf /app/_eval_staging"],
        "work_dir": "/app",
    }])


def _parse_scores(session: object, eval_out: str) -> dict:
    """Parse f2p/p2p scores from TB1 evaluation output."""
    f2p = None
    try:
        data = session._client.download_file(  # type: ignore[attr-defined]
            session._session_id, "test_output/f2p_score.json"  # type: ignore[attr-defined]
        )
        f2p = json.loads(data)
    except Exception:  # noqa: S110
        pass
    if f2p is None:
        m = re.search(r"Score:\s*(\d+)\s*/\s*(\d+)", eval_out)
        if m:
            got, total = int(m.group(1)), int(m.group(2))
            f2p = {"is_pass": 1 if got == total else 0,
                   "step_score": got / total if total else 0}
    if f2p is None:
        try:
            f2p_out = session._client.download_file(  # type: ignore[attr-defined]
                session._session_id, "test_output/f2p_output.txt"  # type: ignore[attr-defined]
            ).decode("utf-8", errors="replace")
            m = re.search(r"Score:\s*(\d+)\s*/\s*(\d+)", f2p_out)
            if m:
                got, total = int(m.group(1)), int(m.group(2))
                f2p = {"is_pass": 1 if got == total else 0,
                       "step_score": got / total if total else 0}
            else:
                passed = len(re.findall(r"PASSED", f2p_out))
                failed = len(re.findall(r"FAILED", f2p_out))
                if passed + failed > 0:
                    f2p = {"is_pass": 1 if failed == 0 else 0,
                           "step_score": passed / (passed + failed)}
        except Exception:  # noqa: S110
            pass

    p2p = None
    try:
        data = session._client.download_file(  # type: ignore[attr-defined]
            session._session_id, "test_output/p2p_output.txt"  # type: ignore[attr-defined]
        )
        text = data.decode("utf-8", errors="replace")
        passed = len(re.findall(r"PASSED", text))
        failed = len(re.findall(r"FAILED", text))
        if passed + failed > 0:
            p2p = {"passed": passed, "total": passed + failed}
    except Exception:  # noqa: S110
        pass

    return {"f2p": f2p, "p2p": p2p}
