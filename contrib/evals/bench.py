#!/usr/bin/env python3
"""Multi-format benchmark runner: Terminal Bench 1.0/2.0 + SWE-bench.

Usage:
    uv run python contrib/evals/bench.py build --repo ~/longcli-bench/tasks_long_cli --push
    uv run python contrib/evals/bench.py list --repo ~/longcli-bench/tasks_long_cli
    uv run python contrib/evals/bench.py batch --repo ~/longcli-bench/tasks_long_cli --model litellm -j 20
    uv run python contrib/evals/bench.py list --bench swebench-verified --source princeton-nlp/SWE-bench_Verified
    uv run python contrib/evals/bench.py batch --bench tb2 --repo ~/harbor-bench/tasks -j 10
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Protocol

import typer

try:
    import yaml as _yaml
except ImportError:  # noqa: S110
    _yaml = None  # type: ignore[assignment]

try:
    import datasets as _datasets
except ImportError:  # noqa: S110
    _datasets = None  # type: ignore[assignment]


app = typer.Typer(
    name="bench",
    help="Build, push, run, and evaluate benchmark task images.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Task spec — uniform representation across all bench formats
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """Format-neutral description of a benchmark task."""

    name: str
    prompt: str
    image: str = ""
    path: str = ""
    difficulty: str = ""
    category: str = ""
    # SWE-bench extras
    instance_id: str = ""
    base_commit: str = ""
    repo: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bench adapter protocol
# ---------------------------------------------------------------------------

class BenchAdapter(Protocol):
    """Pluggable adapter for benchmark formats."""

    def discover_tasks(self, source: str) -> list[TaskSpec]:
        """Find tasks from a local repo path or HuggingFace dataset name."""
        ...

    def get_image(self, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
        """Return the Docker image name/tag for this task."""
        ...

    def supports_build(self) -> bool:
        """Whether this format supports local image building."""
        ...

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        """Run evaluation in the sandbox and return score dict."""
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_image_slug(name: str) -> str:
    return name.replace("_", "-")


def _tb1_image_name(task_name: str, registry: str, prefix: str, tag: str) -> str:
    safe = _safe_image_slug(task_name)
    return f"{registry}/{prefix}-{safe}:{tag}"


def _upload_file_to_sandbox(session: object, path: str, content: bytes) -> None:
    """Upload a file to the sandbox via ARL WriteFile API."""
    session._client.upload_file(  # type: ignore[attr-defined]
        session._session_id, path,  # type: ignore[attr-defined]
        base64.b64encode(content).decode(),
        encoding="base64",
    )


def _load_trace_tools(session_id: str) -> list[dict]:
    result = subprocess.run(
        ["agentm", "trace", "tools", "--session", session_id, "--format", "ndjson"],
        capture_output=True, text=True,
    )
    return [json.loads(line) for line in result.stdout.strip().split("\n") if line.strip()]


def _replay_tools_to_sandbox(
    session: object, tools: list[dict], *, up_to_turn: int | None = None
) -> int:
    """Replay side-effect tool calls in a sandbox. Returns count replayed."""
    replayed = 0
    assistant_index = -1
    for t in tools:
        tool, args = t.get("tool"), t.get("args", {})

        if tool in ("edit", "write", "bash", "read", "glob", "grep"):
            if assistant_index < 0:
                assistant_index = 0

        if up_to_turn is not None and assistant_index > up_to_turn:
            break

        try:
            if tool == "edit":
                path = args.get("path", "")
                old, new = args.get("old_string", ""), args.get("new_string", "")
                if path and old:
                    rel = path.lstrip("/").removeprefix("app/")
                    cur = session._client.download_file(  # type: ignore[attr-defined]
                        session._session_id, rel  # type: ignore[attr-defined]
                    ).decode("utf-8", errors="replace")
                    upd = cur.replace(old, new, 1)
                    if upd != cur:
                        _upload_file_to_sandbox(session, rel, upd.encode())
                        replayed += 1
            elif tool == "write":
                path = args.get("path", "")
                content = args.get("content", "")
                if path:
                    rel = path.lstrip("/").removeprefix("app/")
                    _upload_file_to_sandbox(session, rel, content.encode())
                    replayed += 1
            elif tool == "bash":
                cmd = args.get("cmd", "")
                skip = [
                    "make grade", "make qemu", "qemu-system", "timeout",
                    "python3 ok", "make test", "ctest",
                ]
                if cmd and not any(k in cmd for k in skip):
                    session.execute([{  # type: ignore[attr-defined]
                        "name": "r", "command": ["bash", "-lc", cmd],
                        "work_dir": "/app",
                    }])
                    replayed += 1
        except Exception:  # noqa: S110
            pass
    return replayed


def _replay_trajectory(session: object, session_id: str) -> int:
    tools = _load_trace_tools(session_id)
    return _replay_tools_to_sandbox(session, tools)


# ---------------------------------------------------------------------------
# Terminal Bench 1.0 adapter
# ---------------------------------------------------------------------------

class TerminalBench1Adapter:
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
            # Verify there's a FROM line
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
        return _tb1_image_name(task.name, registry, prefix, tag)

    def supports_build(self) -> bool:
        return True

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        task_dir = Path(task.path)
        folder = _detect_project_folder_tb1(task_dir)

        _upload_tests_tb1(session, task_dir)

        # Copy tests into project dir
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

        return _parse_tb1_scores(session, eval_out)


def _detect_project_folder_tb1(task_dir: Path) -> str:
    task_yaml = task_dir / "task.yaml"
    if task_yaml.is_file() and _yaml is not None:
        raw = _yaml.safe_load(task_yaml.read_text())
        instruction = raw.get("instruction", "") if isinstance(raw, dict) else ""
        m = re.search(r"in folder (\S+)", instruction)
        if m:
            return m.group(1).rstrip(".")
    return task_dir.name.split("_", 1)[-1]


def _upload_tests_tb1(session: object, task_dir: Path) -> None:
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
        _upload_file_to_sandbox(session, staging_rel, f.read_bytes())

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
        _upload_file_to_sandbox(session, "_eval_staging/run-tests.sh", run_tests.read_bytes())
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


def _parse_tb1_scores(session: object, eval_out: str) -> dict:
    """Parse f2p/p2p scores from TB1 evaluation output."""
    f2p = None
    # Try f2p_score.json
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
            ).decode()
            m = re.search(r"Score:\s*(\d+)\s*/\s*(\d+)", f2p_out)
            if m:
                got, total = int(m.group(1)), int(m.group(2))
                f2p = {"is_pass": 1 if got == total else 0,
                       "step_score": got / total if total else 0}
        except Exception:  # noqa: S110
            pass

    p2p = None
    try:
        data = session._client.download_file(  # type: ignore[attr-defined]
            session._session_id, "test_output/p2p_output.txt"  # type: ignore[attr-defined]
        )
        text = data.decode()
        passed = len(re.findall(r"PASSED", text))
        failed = len(re.findall(r"FAILED", text))
        if passed + failed > 0:
            p2p = {"passed": passed, "total": passed + failed}
    except Exception:  # noqa: S110
        pass

    return {"f2p": f2p, "p2p": p2p}


# ---------------------------------------------------------------------------
# Terminal Bench 2.0 adapter (Harbor format)
# ---------------------------------------------------------------------------

class TerminalBench2Adapter:
    """Terminal Bench 2.0: task.toml + instruction.md + environment/ dir."""

    def discover_tasks(self, source: str) -> list[TaskSpec]:
        repo = Path(source).expanduser().resolve()
        tasks: list[TaskSpec] = []
        for task_dir in sorted(repo.iterdir()):
            if not task_dir.is_dir():
                continue
            task_toml = task_dir / "task.toml"
            if not task_toml.is_file():
                continue

            meta = tomllib.loads(task_toml.read_text())

            prompt = ""
            instruction_file = task_dir / "instruction.md"
            if instruction_file.is_file():
                prompt = instruction_file.read_text().strip()

            # Image from environment config
            image = ""
            env_section = meta.get("environment", {})
            if isinstance(env_section, dict):
                image = env_section.get("docker_image", "")

            tasks.append(TaskSpec(
                name=task_dir.name,
                prompt=prompt,
                image=image,
                path=str(task_dir),
                difficulty=meta.get("difficulty", ""),
                category=meta.get("category", ""),
                extra=meta,
            ))
        return tasks

    def get_image(self, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
        # Prefer image from task.toml [environment].docker_image
        if task.image:
            return task.image
        # Fall back to convention-based name
        return _tb1_image_name(task.name, registry, prefix, tag)

    def supports_build(self) -> bool:
        return True

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        task_dir = Path(task.path)

        # Upload tests/test.sh to sandbox
        test_sh = task_dir / "tests" / "test.sh"
        if test_sh.is_file():
            session.execute([{  # type: ignore[attr-defined]
                "name": "prep",
                "command": ["bash", "-lc", "mkdir -p /tests /logs/verifier"],
                "work_dir": "/app",
            }])
            _upload_file_to_sandbox(session, "_eval_staging_test.sh", test_sh.read_bytes())
            session.execute([{  # type: ignore[attr-defined]
                "name": "mv-test",
                "command": ["bash", "-lc",
                    "mv /app/_eval_staging_test.sh /tests/test.sh && "
                    "chmod +x /tests/test.sh"],
                "work_dir": "/app",
            }])

        # Run test.sh
        r = session.execute([{  # type: ignore[attr-defined]
            "name": "eval",
            "command": ["bash", "-lc",
                f"mkdir -p /logs/verifier && "
                f"timeout {timeout} bash /tests/test.sh 2>&1"],
            "work_dir": "/app",
        }])
        eval_out = r.results[0].output.stdout

        # Read reward from /logs/verifier/reward.txt
        reward = None
        try:
            reward_data = session._client.download_file(  # type: ignore[attr-defined]
                session._session_id, "../logs/verifier/reward.txt"  # type: ignore[attr-defined]
            )
            reward = float(reward_data.decode().strip())
        except Exception:  # noqa: S110
            pass

        if reward is None:
            # Try reading via bash as fallback (absolute path outside workspace)
            try:
                r2 = session.execute([{  # type: ignore[attr-defined]
                    "name": "read-reward",
                    "command": ["bash", "-lc", "cat /logs/verifier/reward.txt 2>/dev/null"],
                    "work_dir": "/app",
                }])
                txt = r2.results[0].output.stdout.strip()
                if txt:
                    reward = float(txt)
            except Exception:  # noqa: S110
                pass

        return {
            "reward": reward,
            "eval_output": eval_out[:2000] if eval_out else "",
        }


# ---------------------------------------------------------------------------
# SWE-bench adapter (Verified + Pro)
# ---------------------------------------------------------------------------

class SWEBenchAdapter:
    """SWE-bench Verified / Pro: HuggingFace dataset + pre-built Docker images.

    Does NOT run gold tests inside the agent sandbox. Instead, extracts the
    agent's git diff as model_patch and writes predictions JSONL for
    upstream SWE-bench harness scoring.
    """

    def __init__(self, *, variant: str = "verified") -> None:
        # variant: "verified" or "pro"
        self.variant = variant

    def discover_tasks(self, source: str) -> list[TaskSpec]:
        if _datasets is None:
            typer.echo("Error: 'datasets' library required for SWE-bench. "
                       "Install with: pip install datasets", err=True)
            raise typer.Exit(1)

        ds = _datasets.load_dataset(source, split="test")
        tasks: list[TaskSpec] = []
        for row in ds:
            instance_id = row["instance_id"]
            tasks.append(TaskSpec(
                name=instance_id,
                prompt=row.get("problem_statement", ""),
                image=self._image_for_instance(instance_id, row),
                instance_id=instance_id,
                base_commit=row.get("base_commit", ""),
                repo=row.get("repo", ""),
                extra={k: v for k, v in row.items()
                       if k not in ("problem_statement", "instance_id",
                                    "base_commit", "repo")},
            ))
        return tasks

    def _image_for_instance(self, instance_id: str, row: dict) -> str:
        if self.variant == "pro":
            # SWE-bench Pro uses jefzda/sweap-images with a dockerhub_tag field
            docker_tag = row.get("dockerhub_tag", instance_id)
            return f"jefzda/sweap-images:{docker_tag}"
        # SWE-bench Verified: swebench/sweb.eval.x86_64.{instance_id}
        return f"swebench/sweb.eval.x86_64.{instance_id}:latest"

    def get_image(self, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
        # Pre-built images, ignore registry/prefix/tag
        return task.image

    def supports_build(self) -> bool:
        return False

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        """Extract git diff from sandbox as model_patch. No test execution."""
        # Run git diff in the sandbox to capture the agent's changes
        r = session.execute([{  # type: ignore[attr-defined]
            "name": "extract-patch",
            "command": ["bash", "-lc",
                "cd /app && git diff HEAD 2>/dev/null || "
                "git diff 2>/dev/null || echo ''"],
            "work_dir": "/app",
        }])
        model_patch = r.results[0].output.stdout

        return {
            "instance_id": task.instance_id,
            "model_patch": model_patch,
            "has_patch": bool(model_patch.strip()),
        }


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_BENCH_ADAPTERS: dict[str, type | object] = {
    "tb1": TerminalBench1Adapter,
    "tb2": TerminalBench2Adapter,
    "swebench-verified": lambda: SWEBenchAdapter(variant="verified"),
    "swebench-pro": lambda: SWEBenchAdapter(variant="pro"),
}

BENCH_CHOICES = list(_BENCH_ADAPTERS.keys())


def _get_adapter(bench: str) -> BenchAdapter:
    entry = _BENCH_ADAPTERS[bench]
    if callable(entry) and isinstance(entry, type):
        return entry()  # type: ignore[return-value]
    if callable(entry):
        return entry()  # type: ignore[return-value]
    return entry  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# TB1 legacy helpers (used by build command)
# ---------------------------------------------------------------------------

def _build_base_images(base_dir: Path) -> None:
    import glob as globmod
    for pattern in ["Dockerfile.*-base", "Dockerfile.*_base"]:
        for dockerfile_path in globmod.glob(str(base_dir / pattern)):
            dockerfile = Path(dockerfile_path)
            base_name = (
                dockerfile.name.replace("Dockerfile.", "")
                .replace("-base", "").replace("_base", "")
            )
            tag = f"tb/{base_name}:v0"
            if subprocess.run(
                ["docker", "image", "inspect", tag], capture_output=True
            ).returncode == 0:
                typer.echo(f"  Base {tag} exists, skip")
                continue
            typer.echo(f"  Building {tag} ...")
            subprocess.run(
                ["docker", "build", "-f", str(dockerfile), "-t", tag,
                 str(base_dir), "-q"],
                check=True, capture_output=True,
            )


def _generate_skaffold(tasks: list[TaskSpec], registry: str, prefix: str, tag: str) -> dict:
    artifacts = []
    for task in tasks:
        image = _tb1_image_name(task.name, registry, prefix, tag).rsplit(":", 1)[0]
        artifacts.append({"image": image, "context": task.path})
    return {
        "apiVersion": "skaffold/v4beta11",
        "kind": "Config",
        "metadata": {"name": f"{prefix}-images"},
        "build": {
            "tagPolicy": {"envTemplate": {"template": "{{.IMAGE_NAME}}:" + tag}},
            "local": {"concurrency": 4, "useBuildkit": True, "push": False},
            "artifacts": artifacts,
        },
    }


# ---------------------------------------------------------------------------
# Core run+eval logic
# ---------------------------------------------------------------------------

def _run_and_eval_one(
    task: TaskSpec, *,
    adapter: BenchAdapter,
    source: str,
    model: str, gateway: str,
    registry: str, prefix: str, tag: str,
    out: Path, eval_timeout: int,
) -> dict:
    """Run agent + evaluate one task. Returns result dict."""
    import arl

    name = task.name
    log = out / f"{name}.log"
    score_file = out / f"{name}.score.json"

    image = adapter.get_image(task, registry, prefix, tag)

    # --- Agent phase ---
    session_id = None
    if log.is_file():
        text = log.read_text()
        m = re.search(r"session_id=(\S+)", text)
        if m:
            session_id = m.group(1)

    if session_id is None:
        prompt = task.prompt
        if not prompt:
            return {"task": name, "status": "no_instruction"}

        env = {
            **os.environ,
            "AGENTM_AGENT_ENV_IMAGE": image,
            "AGENTM_AGENT_ENV_GATEWAY_URL": gateway,
            "AGENTM_AGENT_ENV_EXPERIMENT_ID": f"{prefix}-{model}-{name}",
        }
        cmd = [
            "uv", "run", "agentm",
            "--scenario", "terminal_bench_arl",
            "--model", model,
            "-p", prompt,
        ]
        with open(log, "w") as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

        text = log.read_text()
        m = re.search(r"session_id=(\S+)", text)
        if m:
            session_id = m.group(1)

    if session_id is None:
        return {"task": name, "status": "agent_failed"}

    tools_m = re.search(r"tool_calls=(\d+)", log.read_text())
    tools_count = tools_m.group(1) if tools_m else "?"

    # --- Eval phase ---
    if score_file.is_file():
        scores = json.loads(score_file.read_text())
        return {"task": name, "status": "done", "tools": tools_count, **scores}

    session = arl.ManagedSession(
        image=image,
        experiment_id=f"eval-{model}-{name}",
        gateway_url=gateway,
        workspace_dir="/app",
    )
    session.create_sandbox()

    try:
        _replay_trajectory(session, session_id)
        scores = adapter.evaluate(session, task, timeout=eval_timeout)
    finally:
        session.delete_sandbox()

    score_file.write_text(json.dumps({"task": name, **scores}, ensure_ascii=False))

    return {"task": name, "status": "done", "tools": tools_count, **scores}


def _format_score_line(r: dict, bench: str) -> str:
    """Format a one-line summary of a task result for console output."""
    name = r.get("task", "?")
    tools = r.get("tools", "?")
    status = r.get("status", "?").upper()

    if bench.startswith("swebench"):
        has_patch = r.get("has_patch", False)
        return f"  [{status}] {name} tools={tools} patch={'yes' if has_patch else 'no'}"
    if bench == "tb2":
        reward = r.get("reward")
        reward_str = f"{reward:.2f}" if isinstance(reward, (int, float)) else "-"
        return f"  [{status}] {name} tools={tools} reward={reward_str}"
    # tb1
    f2p = r.get("f2p") or {}
    step = f2p.get("step_score", "-") if isinstance(f2p, dict) else "-"
    if isinstance(step, float):
        step = f"{step:.0%}"
    return f"  [{status}] {name} tools={tools} f2p={step}"


def _print_summary_table(results: dict[str, dict], bench: str) -> None:
    """Print final summary table appropriate for the bench format."""
    typer.echo(f"\n{'=' * 75}")

    if bench.startswith("swebench"):
        typer.echo(f"  {'Task':<50} {'Patch'}")
        typer.echo(f"  {'-' * 70}")
        for name in sorted(results):
            r = results[name]
            has_patch = r.get("has_patch", False)
            typer.echo(f"  {name:<50} {'yes' if has_patch else 'no'}")
        patched = sum(1 for r in results.values() if r.get("has_patch"))
        typer.echo(f"\n{patched}/{len(results)} produced patches")
        return

    if bench == "tb2":
        typer.echo(f"  {'Task':<30} {'Reward':<12}")
        typer.echo(f"  {'-' * 45}")
        for name in sorted(results):
            r = results[name]
            reward = r.get("reward")
            reward_str = f"{reward:.2f}" if isinstance(reward, (int, float)) else "-"
            typer.echo(f"  {name:<30} {reward_str:<12}")
        scored: list[float] = [
            float(r["reward"]) for r in results.values()
            if isinstance(r.get("reward"), (int, float))
        ]
        if scored:
            avg = sum(scored) / len(scored)
            typer.echo(f"\nAvg reward: {avg:.3f} ({len(scored)}/{len(results)} scored)")
        return

    # tb1
    typer.echo(f"  {'Task':<25} {'F2P pass':<10} {'F2P step':<12} {'P2P':<12}")
    typer.echo(f"  {'-' * 70}")
    for name in sorted(results):
        r = results[name]
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
        typer.echo(f"  {name:<25} {str(f2p_pass):<10} {str(f2p_step):<12} {p2p_str:<12}")


def _write_swebench_predictions(
    results: dict[str, dict], out: Path, model: str
) -> None:
    """Write SWE-bench predictions JSONL for upstream harness scoring."""
    pred_file = out / "predictions.jsonl"
    with open(pred_file, "w") as f:
        for name in sorted(results):
            r = results[name]
            if r.get("status") != "done":
                continue
            f.write(json.dumps({
                "instance_id": r.get("instance_id", name),
                "model_patch": r.get("model_patch", ""),
                "model_name_or_path": model,
            }, ensure_ascii=False) + "\n")
    typer.echo(f"\nPredictions written to: {pred_file}")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def build(
    repo: Annotated[Path, typer.Option("--repo")],
    bench: Annotated[str, typer.Option("--bench")] = "tb1",
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    push: Annotated[bool, typer.Option("--push")] = False,
    base_dir: Annotated[Path | None, typer.Option("--base-dir")] = None,
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
) -> None:
    """Build (and optionally push) task environment images."""
    adapter = _get_adapter(bench)
    if not adapter.supports_build():
        typer.echo(f"Error: {bench} uses pre-built images, nothing to build.", err=True)
        raise typer.Exit(1)

    repo = repo.expanduser().resolve()
    if base_dir:
        _build_base_images(base_dir.expanduser().resolve())

    tasks = adapter.discover_tasks(str(repo))
    if task:
        tasks = [t for t in tasks if t.name in task]
    if not tasks:
        typer.echo("No tasks found.", err=True)
        raise typer.Exit(1)

    if _yaml is None:
        typer.echo("Error: PyYAML required for skaffold config generation.", err=True)
        raise typer.Exit(1)

    skaffold_cfg = _generate_skaffold(tasks, registry, prefix, tag)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        _yaml.dump(skaffold_cfg, f)
        skaffold_path = f.name
    try:
        cmd = ["skaffold", "build", "-f", skaffold_path]
        if push:
            cmd.append("--push")
        result = subprocess.run(cmd, env={**os.environ, "TAG": tag})
        if result.returncode != 0:
            raise typer.Exit(result.returncode)
    finally:
        os.unlink(skaffold_path)
    typer.echo(f"Built {len(tasks)} images ({registry}/{prefix}-*:{tag})")


@app.command("list")
def list_tasks(
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    bench: Annotated[str, typer.Option("--bench")] = "tb1",
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List discovered tasks and their image names."""
    resolved_source = _resolve_source(repo, source, bench)
    adapter = _get_adapter(bench)
    tasks = adapter.discover_tasks(resolved_source)

    if json_out:
        out = []
        for t in tasks:
            d = {
                "name": t.name,
                "image": adapter.get_image(t, registry, prefix, tag),
                "difficulty": t.difficulty,
                "category": t.category,
            }
            if t.instance_id:
                d["instance_id"] = t.instance_id
            if t.repo:
                d["repo"] = t.repo
            out.append(d)
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if bench.startswith("swebench"):
        typer.echo(f"{'Instance ID':<55} {'Repo':<30} {'Image'}")
        for t in tasks:
            typer.echo(f"{t.name:<55} {t.repo:<30} "
                       f"{adapter.get_image(t, registry, prefix, tag)}")
    else:
        typer.echo(f"{'Task':<30} {'Diff':<8} {'Image'}")
        for t in tasks:
            typer.echo(f"{t.name:<30} {t.difficulty:<8} "
                       f"{adapter.get_image(t, registry, prefix, tag)}")
    typer.echo(f"\n{len(tasks)} tasks")


@app.command()
def run(
    task: Annotated[str, typer.Option("--task", "-t")],
    instruction: Annotated[str | None, typer.Option("-p")] = None,
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    bench: Annotated[str, typer.Option("--bench")] = "tb1",
    model: Annotated[str, typer.Option()] = "glm47",
    gateway: Annotated[str, typer.Option()] = "http://localhost:28080",
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
) -> None:
    """Run a single task via ARL sandbox."""
    adapter = _get_adapter(bench)
    resolved_source = _resolve_source(repo, source, bench)

    # Find the task spec
    all_tasks = adapter.discover_tasks(resolved_source)
    matching = [t for t in all_tasks if t.name == task]
    if not matching:
        typer.echo(f"Error: task '{task}' not found in {bench} source.", err=True)
        raise typer.Exit(1)
    task_spec = matching[0]

    image = adapter.get_image(task_spec, registry, prefix, tag)
    prompt = instruction or task_spec.prompt
    if not prompt:
        typer.echo("Error: no prompt found. Provide -p or ensure task has instruction.", err=True)
        raise typer.Exit(1)

    env = {
        **os.environ,
        "AGENTM_AGENT_ENV_IMAGE": image,
        "AGENTM_AGENT_ENV_GATEWAY_URL": gateway,
        "AGENTM_AGENT_ENV_EXPERIMENT_ID": f"{prefix}-{task}",
    }
    result = subprocess.run([
        "uv", "run", "agentm", "--scenario", "terminal_bench_arl",
        "--model", model, "-p", prompt,
    ], env=env)
    raise typer.Exit(result.returncode)


@app.command()
def batch(
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    bench: Annotated[str, typer.Option("--bench")] = "tb1",
    model: Annotated[str, typer.Option()] = "glm47",
    gateway: Annotated[str, typer.Option()] = "http://localhost:28080",
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    concurrency: Annotated[int, typer.Option("-j")] = 5,
    results_dir: Annotated[Path, typer.Option("--results")] = Path("/tmp/bench-results"),
    eval_timeout: Annotated[int, typer.Option("--eval-timeout")] = 300,
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
) -> None:
    """Run all tasks in parallel, then evaluate. Results include scores."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    resolved_source = _resolve_source(repo, source, bench)
    adapter = _get_adapter(bench)
    tasks = adapter.discover_tasks(resolved_source)
    if task:
        tasks = [t for t in tasks if t.name in task]
    if not tasks:
        typer.echo("No tasks found.", err=True)
        raise typer.Exit(1)

    out = results_dir / f"{bench}-{model}"
    out.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Batch: {len(tasks)} tasks | bench={bench} | model={model} | concurrency={concurrency}")
    typer.echo(f"Results: {out}")
    typer.echo("")

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _run_and_eval_one, t,
                adapter=adapter, source=resolved_source,
                model=model, gateway=gateway,
                registry=registry, prefix=prefix, tag=tag,
                out=out, eval_timeout=eval_timeout,
            ): t.name
            for t in tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                r = future.result()
                results[name] = r
                typer.echo(_format_score_line(r, bench))
            except Exception as e:
                typer.echo(f"  [ERROR] {name}: {e}")

    _print_summary_table(results, bench)

    done = sum(1 for r in results.values() if r.get("status") == "done")
    typer.echo(f"\n{done}/{len(tasks)} completed")

    # Write predictions JSONL for SWE-bench
    if bench.startswith("swebench"):
        _write_swebench_predictions(results, out, model)


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def _resolve_source(
    repo: Path | None, source: str | None, bench: str
) -> str:
    """Resolve the task source from --repo or --source flags."""
    if bench.startswith("swebench"):
        if source:
            return source
        # Default dataset names
        defaults = {
            "swebench-verified": "princeton-nlp/SWE-bench_Verified",
            "swebench-pro": "ScaleAI/SWE-bench_Pro",
        }
        return defaults.get(bench, "princeton-nlp/SWE-bench_Verified")
    # Local repo path for tb1/tb2
    if repo is not None:
        return str(repo.expanduser().resolve())
    if source:
        return str(Path(source).expanduser().resolve())
    typer.echo("Error: --repo required for this bench format.", err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
