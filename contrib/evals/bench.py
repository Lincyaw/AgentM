#!/usr/bin/env python3
"""Terminal-bench image builder and task runner.

Scans any terminal-bench-format benchmark repo for task directories
(Dockerfile + INSTRUCTION.md), builds images, and optionally pushes
to a container registry. Works with longcli-bench, terminal-bench core,
or any custom task set that follows the convention.

Usage:
    uv run python contrib/evals/bench.py build --repo ~/longcli-bench/tasks_long_cli --push
    uv run python contrib/evals/bench.py build --repo ~/terminal-bench/tasks --prefix tb-core --push
    uv run python contrib/evals/bench.py list --repo ~/longcli-bench/tasks_long_cli
    uv run python contrib/evals/bench.py run --task cs61_fa24_hog --model glm47
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="bench",
    help="Build, push, and run terminal-bench task images.",
    add_completion=False,
)


def _discover_tasks(repo: Path) -> list[dict]:
    """Scan a directory for terminal-bench task subdirectories."""
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
        meta = {}
        if task_yaml.is_file():
            import yaml

            raw = yaml.safe_load(task_yaml.read_text())
            if isinstance(raw, dict):
                meta = {
                    "instruction": raw.get("instruction", ""),
                    "difficulty": raw.get("difficulty", ""),
                    "category": raw.get("category", ""),
                }
        tasks.append(
            {
                "name": task_dir.name,
                "path": str(task_dir),
                "base_image": first_from.removeprefix("FROM").strip(),
                "has_instruction": instruction.is_file(),
                **meta,
            }
        )
    return tasks


def _image_name(task_name: str, registry: str, prefix: str, tag: str) -> str:
    safe = task_name.replace("_", "-")
    return f"{registry}/{prefix}-{safe}:{tag}"


def _build_base_images(base_dir: Path) -> None:
    """Build base images from a longcli_dockerImage-style directory."""
    for pattern in ["Dockerfile.*-base", "Dockerfile.*_base"]:
        import glob as globmod

        for dockerfile in globmod.glob(str(base_dir / pattern)):
            dockerfile = Path(dockerfile)
            base_name = dockerfile.name.replace("Dockerfile.", "").replace("-base", "").replace("_base", "")
            tag = f"tb/{base_name}:v0"
            if (
                subprocess.run(
                    ["docker", "image", "inspect", tag],
                    capture_output=True,
                ).returncode
                == 0
            ):
                typer.echo(f"  Base {tag} exists, skip")
                continue
            typer.echo(f"  Building {tag} ...")
            subprocess.run(
                ["docker", "build", "-f", str(dockerfile), "-t", tag, str(base_dir), "-q"],
                check=True,
                capture_output=True,
            )


def _generate_skaffold(
    tasks: list[dict], registry: str, prefix: str, tag: str
) -> dict:
    """Generate a skaffold config dict."""
    artifacts = []
    for task in tasks:
        image = _image_name(task["name"], registry, prefix, tag)
        image_no_tag = image.rsplit(":", 1)[0]
        artifacts.append({"image": image_no_tag, "context": task["path"]})
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
    repo: Annotated[
        Path,
        typer.Option("--repo", help="Directory containing task subdirectories."),
    ],
    registry: Annotated[str, typer.Option(help="Registry prefix.")] = "opspai",
    prefix: Annotated[str, typer.Option(help="Image name prefix.")] = "longcli",
    tag: Annotated[str, typer.Option(help="Image tag.")] = "v0",
    push: Annotated[bool, typer.Option("--push", help="Push to registry after build.")] = False,
    base_dir: Annotated[
        Path | None,
        typer.Option("--base-dir", help="Directory with base Dockerfiles to build first."),
    ] = None,
    task: Annotated[
        list[str] | None,
        typer.Option("--task", "-t", help="Only build these tasks (repeatable)."),
    ] = None,
) -> None:
    """Build (and optionally push) task environment images."""

    repo = repo.expanduser().resolve()
    if not repo.is_dir():
        typer.echo(f"Error: {repo} not found.", err=True)
        raise typer.Exit(1)

    if base_dir:
        base_dir = base_dir.expanduser().resolve()
        typer.echo(f"Building base images from {base_dir}")
        _build_base_images(base_dir)

    tasks = _discover_tasks(repo)
    if task:
        tasks = [t for t in tasks if t["name"] in task]
    if not tasks:
        typer.echo("No tasks found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(tasks)} tasks in {repo}")

    skaffold_cfg = _generate_skaffold(tasks, registry, prefix, tag)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        import yaml

        yaml.dump(skaffold_cfg, f, default_flow_style=False)
        skaffold_path = f.name

    try:
        cmd = ["skaffold", "build", "-f", skaffold_path]
        if push:
            cmd.append("--push")
        typer.echo(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env={**os.environ, "TAG": tag})
        if result.returncode != 0:
            raise typer.Exit(result.returncode)
    finally:
        os.unlink(skaffold_path)

    typer.echo(f"\nBuilt {len(tasks)} images (prefix={registry}/{prefix}-*:{tag})")
    if push:
        typer.echo("Images pushed to registry.")


@app.command("list")
def list_tasks(
    repo: Annotated[
        Path,
        typer.Option("--repo", help="Directory containing task subdirectories."),
    ],
    registry: Annotated[str, typer.Option(help="Registry prefix.")] = "opspai",
    prefix: Annotated[str, typer.Option(help="Image name prefix.")] = "longcli",
    tag: Annotated[str, typer.Option(help="Image tag.")] = "v0",
    json_out: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """List discovered tasks and their image names."""

    repo = repo.expanduser().resolve()
    tasks = _discover_tasks(repo)
    if not tasks:
        typer.echo("No tasks found.", err=True)
        raise typer.Exit(1)

    if json_out:
        for t in tasks:
            t["image"] = _image_name(t["name"], registry, prefix, tag)
        typer.echo(json.dumps(tasks, indent=2, ensure_ascii=False))
        return

    typer.echo(f"{'Task':<30} {'Difficulty':<10} {'Category':<15} {'Image'}")
    typer.echo("-" * 100)
    for t in tasks:
        image = _image_name(t["name"], registry, prefix, tag)
        typer.echo(
            f"{t['name']:<30} {t.get('difficulty',''):<10} "
            f"{t.get('category',''):<15} {image}"
        )
    typer.echo(f"\n{len(tasks)} tasks")


@app.command()
def run(
    task: Annotated[str, typer.Option("--task", "-t", help="Task name.")],
    instruction: Annotated[
        str | None,
        typer.Option("--instruction", "-p", help="Prompt (default: from task.yaml)."),
    ] = None,
    repo: Annotated[
        Path | None,
        typer.Option("--repo", help="Task repo (to read task.yaml for instruction)."),
    ] = None,
    model: Annotated[str, typer.Option(help="AgentM model profile.")] = "glm47",
    gateway: Annotated[str, typer.Option(help="ARL gateway URL.")] = "http://localhost:28080",
    registry: Annotated[str, typer.Option(help="Registry prefix.")] = "opspai",
    prefix: Annotated[str, typer.Option(help="Image name prefix.")] = "longcli",
    tag: Annotated[str, typer.Option(help="Image tag.")] = "v0",
) -> None:
    """Run a single task via ARL sandbox."""

    image = _image_name(task, registry, prefix, tag)

    if instruction is None:
        if repo is None:
            typer.echo("Error: --instruction or --repo required.", err=True)
            raise typer.Exit(1)
        repo = repo.expanduser().resolve()
        task_yaml = repo / task / "task.yaml"
        if not task_yaml.is_file():
            typer.echo(f"Error: {task_yaml} not found.", err=True)
            raise typer.Exit(1)
        import yaml

        raw = yaml.safe_load(task_yaml.read_text())
        instruction = raw.get("instruction", "")
        if not instruction:
            typer.echo(f"Error: no instruction in {task_yaml}.", err=True)
            raise typer.Exit(1)

    typer.echo(f"Task:    {task}")
    typer.echo(f"Image:   {image}")
    typer.echo(f"Model:   {model}")
    typer.echo(f"Gateway: {gateway}")
    typer.echo("")

    env = {
        **os.environ,
        "AGENTM_AGENT_ENV_IMAGE": image,
        "AGENTM_AGENT_ENV_GATEWAY_URL": gateway,
        "AGENTM_AGENT_ENV_EXPERIMENT_ID": f"{prefix}-{task}",
    }
    cmd = [
        "uv", "run", "agentm",
        "--scenario", "terminal_bench_arl",
        "--model", model,
        "-p", instruction,
    ]
    result = subprocess.run(cmd, env=env)
    raise typer.Exit(result.returncode)


@app.command()
def batch(
    repo: Annotated[
        Path,
        typer.Option("--repo", help="Directory containing task subdirectories."),
    ],
    model: Annotated[str, typer.Option(help="AgentM model profile.")] = "glm47",
    gateway: Annotated[str, typer.Option(help="ARL gateway URL.")] = "http://localhost:28080",
    registry: Annotated[str, typer.Option(help="Registry prefix.")] = "opspai",
    prefix: Annotated[str, typer.Option(help="Image name prefix.")] = "longcli",
    tag: Annotated[str, typer.Option(help="Image tag.")] = "v0",
    concurrency: Annotated[int, typer.Option("-j", help="Parallel tasks.")] = 5,
    results_dir: Annotated[
        Path,
        typer.Option("--results", help="Results directory."),
    ] = Path("/tmp/longcli-results"),
    task: Annotated[
        list[str] | None,
        typer.Option("--task", "-t", help="Only run these tasks (repeatable)."),
    ] = None,
) -> None:
    """Run all tasks in parallel via ARL sandbox."""

    from concurrent.futures import ThreadPoolExecutor, as_completed

    repo = repo.expanduser().resolve()
    tasks = _discover_tasks(repo)
    if task:
        tasks = [t for t in tasks if t["name"] in task]
    if not tasks:
        typer.echo("No tasks found.", err=True)
        raise typer.Exit(1)

    out = results_dir / model
    out.mkdir(parents=True, exist_ok=True)

    def _run_one(t: dict) -> dict:
        name = t["name"]
        log = out / f"{name}.log"
        if log.is_file() and "session_id=" in log.read_text():
            return {"task": name, "status": "skip"}

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
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

        text = log.read_text()
        if "session_id=" in text:
            import re

            tools = ""
            m = re.search(r"tool_calls=(\d+)", text)
            if m:
                tools = m.group(1)
            return {"task": name, "status": "done", "tools": tools}
        return {"task": name, "status": "fail", "rc": result.returncode}

    typer.echo(
        f"Batch: {len(tasks)} tasks | model={model} | concurrency={concurrency}"
    )
    typer.echo(f"Results: {out}")
    typer.echo("")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run_one, t): t["name"] for t in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                r = future.result()
                status = r["status"]
                extra = f" tools={r['tools']}" if r.get("tools") else ""
                typer.echo(f"  [{status.upper()}] {name}{extra}")
            except Exception as e:
                typer.echo(f"  [ERROR] {name}: {e}")

    typer.echo("")
    typer.echo("Summary:")
    done = skip = fail = 0
    for t in sorted(tasks, key=lambda x: x["name"]):
        log = out / f"{t['name']}.log"
        if log.is_file() and "session_id=" in log.read_text():
            done += 1
            typer.echo(f"  ✅ {t['name']}")
        else:
            fail += 1
            typer.echo(f"  ❌ {t['name']}")
    typer.echo(f"\n{done} done, {fail} failed")


if __name__ == "__main__":
    app()
