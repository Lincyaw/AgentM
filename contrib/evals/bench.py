#!/usr/bin/env python3
"""Terminal-bench image builder, task runner, and evaluator.

Usage:
    uv run python contrib/evals/bench.py build --repo ~/longcli-bench/tasks_long_cli --push
    uv run python contrib/evals/bench.py list --repo ~/longcli-bench/tasks_long_cli
    uv run python contrib/evals/bench.py batch --repo ~/longcli-bench/tasks_long_cli --model litellm -j 20
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="bench",
    help="Build, push, run, and evaluate terminal-bench task images.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_tasks(repo: Path) -> list[dict]:
    tasks = []
    for task_dir in sorted(repo.iterdir()):
        if not task_dir.is_dir():
            continue
        dockerfile = task_dir / "Dockerfile"
        if not dockerfile.is_file():
            continue
        first_from = None
        for line in dockerfile.read_text().splitlines():
            if line.strip().startswith("FROM"):
                first_from = line.strip()
                break
        if not first_from:
            continue
        instruction = task_dir / "INSTRUCTION.md"
        task_yaml = task_dir / "task.yaml"
        meta: dict = {}
        if task_yaml.is_file():
            import yaml
            raw = yaml.safe_load(task_yaml.read_text())
            if isinstance(raw, dict):
                meta = {
                    "instruction": raw.get("instruction", ""),
                    "difficulty": raw.get("difficulty", ""),
                    "category": raw.get("category", ""),
                }
        tasks.append({
            "name": task_dir.name,
            "path": str(task_dir),
            "base_image": first_from.removeprefix("FROM").strip(),
            "has_instruction": instruction.is_file(),
            **meta,
        })
    return tasks


def _image_name(task_name: str, registry: str, prefix: str, tag: str) -> str:
    safe = task_name.replace("_", "-")
    return f"{registry}/{prefix}-{safe}:{tag}"


def _detect_project_folder(task_dir: Path) -> str:
    task_yaml = task_dir / "task.yaml"
    if task_yaml.is_file():
        import yaml
        raw = yaml.safe_load(task_yaml.read_text())
        instruction = raw.get("instruction", "") if isinstance(raw, dict) else ""
        m = re.search(r"in folder (\S+)", instruction)
        if m:
            return m.group(1).rstrip(".")
    return task_dir.name.split("_", 1)[-1]


def _upload_file_to_sandbox(session, path: str, content: bytes) -> None:
    """Upload a file to the sandbox via ARL WriteFile API (no bash, no size limit)."""
    session._client.upload_file(
        session._session_id, path,
        base64.b64encode(content).decode(),
        encoding="base64",
    )


def _load_trace_tools(session_id: str) -> list[dict]:
    result = subprocess.run(
        ["agentm", "trace", "tools", "--session", session_id, "--format", "ndjson"],
        capture_output=True, text=True,
    )
    return [json.loads(line) for line in result.stdout.strip().split("\n") if line.strip()]


def _replay_tools_to_sandbox(session, tools: list[dict], *, up_to_turn: int | None = None) -> int:
    """Replay side-effect tool calls in a sandbox. Returns count replayed."""
    replayed = 0
    assistant_index = -1
    for t in tools:
        tool, args = t.get("tool"), t.get("args", {})

        # Track turn index (assistant messages carry tool calls)
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
                    cur = session._client.download_file(
                        session._session_id, rel
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
                    session.execute([{
                        "name": "r", "command": ["bash", "-lc", cmd],
                        "work_dir": "/app",
                    }])
                    replayed += 1
        except Exception:  # noqa: S110
            pass
    return replayed


def _replay_trajectory(session, session_id: str) -> int:
    tools = _load_trace_tools(session_id)
    return _replay_tools_to_sandbox(session, tools)


def _upload_tests(session, task_dir: Path) -> None:
    """Upload test files to /app/_eval_tests/ via WriteFile API, then mv to /tests/."""
    tests_dir = task_dir / "tests"
    if not tests_dir.is_dir():
        return

    session.execute([{
        "name": "prep",
        "command": ["bash", "-lc", "mkdir -p /tests /app/test_output /app/_eval_staging"],
        "work_dir": "/app",
    }])

    for f in tests_dir.rglob("*"):
        if not f.is_file():
            continue
        rel = str(f.relative_to(tests_dir))
        content = f.read_bytes()
        # Upload to workspace staging dir (WriteFile API requires workspace-relative path)
        staging_rel = f"_eval_staging/{rel}"
        _upload_file_to_sandbox(session, staging_rel, content)

    # Move from staging to /tests/ and make executable
    session.execute([{
        "name": "mv-tests",
        "command": ["bash", "-lc",
            'cd /app/_eval_staging && find . -type f | while read f; do '
            'mkdir -p "/tests/$(dirname "$f")" && '
            'mv "$f" "/tests/$f" && chmod +x "/tests/$f"; done'],
        "work_dir": "/app",
    }])

    # Upload run-tests.sh
    run_tests = task_dir / "run-tests.sh"
    if run_tests.is_file():
        _upload_file_to_sandbox(session, "_eval_staging/run-tests.sh", run_tests.read_bytes())
        session.execute([{
            "name": "mv-rt",
            "command": ["bash", "-lc", "mv /app/_eval_staging/run-tests.sh /tests/run-tests.sh && chmod +x /tests/run-tests.sh"],
            "work_dir": "/app",
        }])

    # Cleanup staging
    session.execute([{
        "name": "cleanup",
        "command": ["bash", "-lc", "rm -rf /app/_eval_staging"],
        "work_dir": "/app",
    }])


def _run_eval(session, task_dir: Path, timeout: int = 300) -> dict:
    """Run evaluation and return scores."""
    folder = _detect_project_folder(task_dir)

    # Copy tests into project dir (run-tests.sh expects this)
    session.execute([{
        "name": "cp-tests",
        "command": ["bash", "-lc", f'cp -a /tests/. /app/{folder}/ 2>/dev/null || true'],
        "work_dir": "/app",
    }])

    # Run official evaluation
    r = session.execute([{
        "name": "eval",
        "command": ["bash", "-lc",
            f"export TEST_DIR=/tests && cd /app && "
            f"killall qemu-system-riscv64 2>/dev/null; "
            f"timeout {timeout} bash /tests/run-tests.sh 2>&1"],
        "work_dir": "/app",
    }])
    eval_out = r.results[0].output.stdout

    # Read f2p score
    f2p = None
    try:
        data = session._client.download_file(session._session_id, "test_output/f2p_score.json")
        f2p = json.loads(data)
    except Exception:  # noqa: S110
        pass
    if f2p is None:
        # Fallback: parse Score: X/Y from output or f2p_output.txt
        for text in [eval_out]:
            m = re.search(r"Score:\s*(\d+)\s*/\s*(\d+)", text)
            if m:
                got, total = int(m.group(1)), int(m.group(2))
                f2p = {"is_pass": 1 if got == total else 0,
                       "step_score": got / total if total else 0}
                break
    if f2p is None:
        try:
            f2p_out = session._client.download_file(
                session._session_id, "test_output/f2p_output.txt"
            ).decode()
            m = re.search(r"Score:\s*(\d+)\s*/\s*(\d+)", f2p_out)
            if m:
                got, total = int(m.group(1)), int(m.group(2))
                f2p = {"is_pass": 1 if got == total else 0,
                       "step_score": got / total if total else 0}
        except Exception:  # noqa: S110
            pass

    # Read p2p score
    p2p = None
    try:
        data = session._client.download_file(session._session_id, "test_output/p2p_output.txt")
        text = data.decode()
        passed = len(re.findall(r"PASSED", text))
        failed = len(re.findall(r"FAILED", text))
        if passed + failed > 0:
            p2p = {"passed": passed, "total": passed + failed}
    except Exception:  # noqa: S110
        pass

    return {"f2p": f2p, "p2p": p2p}


def _run_and_eval_one(
    t: dict, *,
    repo: Path, model: str, gateway: str,
    registry: str, prefix: str, tag: str,
    out: Path, eval_timeout: int,
) -> dict:
    """Run agent + evaluate one task. Returns result dict."""
    import arl

    name = t["name"]
    log = out / f"{name}.log"
    score_file = out / f"{name}.score.json"

    # --- Agent phase ---
    session_id = None
    if log.is_file():
        text = log.read_text()
        m = re.search(r"session_id=(\S+)", text)
        if m:
            session_id = m.group(1)

    if session_id is None:
        image = _image_name(name, registry, prefix, tag)
        instruction = t.get("instruction", "")
        if not instruction:
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
            "-p", instruction,
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
    tools = tools_m.group(1) if tools_m else "?"

    # --- Eval phase ---
    if score_file.is_file():
        scores = json.loads(score_file.read_text())
        return {"task": name, "status": "done", "tools": tools, **scores}

    task_dir = repo / name
    image = _image_name(name, registry, prefix, tag)

    session = arl.ManagedSession(
        image=image,
        experiment_id=f"eval-{model}-{name}",
        gateway_url=gateway,
        workspace_dir="/app",
    )
    session.create_sandbox()

    try:
        _replay_trajectory(session, session_id)
        _upload_tests(session, task_dir)
        scores = _run_eval(session, task_dir, timeout=eval_timeout)
    finally:
        session.delete_sandbox()

    # Persist scores
    score_file.write_text(json.dumps({"task": name, **scores}, ensure_ascii=False))

    return {"task": name, "status": "done", "tools": tools, **scores}


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _build_base_images(base_dir: Path) -> None:
    import glob as globmod
    for pattern in ["Dockerfile.*-base", "Dockerfile.*_base"]:
        for dockerfile in globmod.glob(str(base_dir / pattern)):
            dockerfile = Path(dockerfile)
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


def _generate_skaffold(tasks: list[dict], registry: str, prefix: str, tag: str) -> dict:
    artifacts = []
    for task in tasks:
        image = _image_name(task["name"], registry, prefix, tag).rsplit(":", 1)[0]
        artifacts.append({"image": image, "context": task["path"]})
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


@app.command()
def build(
    repo: Annotated[Path, typer.Option("--repo")],
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    push: Annotated[bool, typer.Option("--push")] = False,
    base_dir: Annotated[Path | None, typer.Option("--base-dir")] = None,
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
) -> None:
    """Build (and optionally push) task environment images."""
    repo = repo.expanduser().resolve()
    if base_dir:
        _build_base_images(base_dir.expanduser().resolve())
    tasks = _discover_tasks(repo)
    if task:
        tasks = [t for t in tasks if t["name"] in task]
    if not tasks:
        raise typer.Exit(1)

    skaffold_cfg = _generate_skaffold(tasks, registry, prefix, tag)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        import yaml
        yaml.dump(skaffold_cfg, f)
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
    repo: Annotated[Path, typer.Option("--repo")],
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List discovered tasks and their image names."""
    repo = repo.expanduser().resolve()
    tasks = _discover_tasks(repo)
    if json_out:
        for t in tasks:
            t["image"] = _image_name(t["name"], registry, prefix, tag)
        typer.echo(json.dumps(tasks, indent=2, ensure_ascii=False))
        return
    typer.echo(f"{'Task':<30} {'Diff':<8} {'Image'}")
    for t in tasks:
        typer.echo(f"{t['name']:<30} {t.get('difficulty',''):<8} "
                   f"{_image_name(t['name'], registry, prefix, tag)}")
    typer.echo(f"\n{len(tasks)} tasks")


@app.command()
def run(
    task: Annotated[str, typer.Option("--task", "-t")],
    instruction: Annotated[str | None, typer.Option("-p")] = None,
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    model: Annotated[str, typer.Option()] = "glm47",
    gateway: Annotated[str, typer.Option()] = "http://localhost:28080",
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
) -> None:
    """Run a single task via ARL sandbox."""
    image = _image_name(task, registry, prefix, tag)
    if instruction is None:
        if repo is None:
            typer.echo("Error: -p or --repo required.", err=True)
            raise typer.Exit(1)
        import yaml
        raw = yaml.safe_load((repo.expanduser() / task / "task.yaml").read_text())
        instruction = raw.get("instruction", "")
    env = {
        **os.environ,
        "AGENTM_AGENT_ENV_IMAGE": image,
        "AGENTM_AGENT_ENV_GATEWAY_URL": gateway,
        "AGENTM_AGENT_ENV_EXPERIMENT_ID": f"{prefix}-{task}",
    }
    result = subprocess.run([
        "uv", "run", "agentm", "--scenario", "terminal_bench_arl",
        "--model", model, "-p", instruction,
    ], env=env)
    raise typer.Exit(result.returncode)


@app.command()
def batch(
    repo: Annotated[Path, typer.Option("--repo")],
    model: Annotated[str, typer.Option()] = "glm47",
    gateway: Annotated[str, typer.Option()] = "http://localhost:28080",
    registry: Annotated[str, typer.Option()] = "opspai",
    prefix: Annotated[str, typer.Option()] = "longcli",
    tag: Annotated[str, typer.Option()] = "v0",
    concurrency: Annotated[int, typer.Option("-j")] = 5,
    results_dir: Annotated[Path, typer.Option("--results")] = Path("/tmp/longcli-results"),
    eval_timeout: Annotated[int, typer.Option("--eval-timeout")] = 300,
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
) -> None:
    """Run all tasks in parallel, then evaluate. Results include scores."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    repo = repo.expanduser().resolve()
    tasks = _discover_tasks(repo)
    if task:
        tasks = [t for t in tasks if t["name"] in task]
    if not tasks:
        raise typer.Exit(1)

    out = results_dir / model
    out.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Batch: {len(tasks)} tasks | model={model} | concurrency={concurrency}")
    typer.echo(f"Results: {out}")
    typer.echo("")

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _run_and_eval_one, t,
                repo=repo, model=model, gateway=gateway,
                registry=registry, prefix=prefix, tag=tag,
                out=out, eval_timeout=eval_timeout,
            ): t["name"]
            for t in tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                r = future.result()
                results[name] = r
                f2p = r.get("f2p") or {}
                step = f2p.get("step_score", "-") if isinstance(f2p, dict) else "-"
                if isinstance(step, float):
                    step = f"{step:.0%}"
                tools = r.get("tools", "?")
                typer.echo(f"  [{r.get('status','?').upper()}] {name} tools={tools} f2p={step}")
            except Exception as e:
                typer.echo(f"  [ERROR] {name}: {e}")

    # Summary table
    typer.echo(f"\n{'='*75}")
    typer.echo(f"  {'Task':<25} {'F2P pass':<10} {'F2P step':<12} {'P2P':<12}")
    typer.echo(f"  {'-'*70}")
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

    done = sum(1 for r in results.values() if r.get("status") == "done")
    typer.echo(f"\n{done}/{len(tasks)} completed")


if __name__ == "__main__":
    app()
