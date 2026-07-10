"""ALE task-data images — bake each task's data into ONE private image.

Per task, a single small image
``<registry>/ale-<domain>-<task>-data:<tag>`` holds ``input/``,
``software/`` and ``reference/`` under ``/ale/<domain>/<task>/<variant>/``.
At run time it becomes an agent-env *private container* (``ale-data``) in
the sandbox pod: the agent's executor container shares only the workspace
volume with it and can never read its filesystem, so the answer key stays
hidden even though it ships in the same image. The harness copies data
into the workspace from inside the private container — ``input/`` +
``software/`` before the agent starts, ``reference/`` only after it
finishes (ALE's timing-based secrecy, unchanged).

The agent's own environment is a separate, *shared* runtime image passed
to ``run --image`` — one image for all tasks, so ARL warm pools stay
reusable. Tasks with ``requiredSystemPackages`` need those baked into
that runtime image (see the ALE repo's ``env/packages-linux``).

Build-time input: a task-data root laid out as
``<data_root>/<domain>/<task>/<variant>/{input,software,reference}``.
Data is needed once on the build host; runs never re-upload it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from loguru import logger

BAKED_DATA_ROOT = "/ale"
DATA_CONTAINER_NAME = "ale-data"
_BUILD_DIR = ".ale_build"


def _slug(domain: str, name: str) -> str:
    return f"ale-{domain}-{name}".lower().replace("_", "-").replace("/", "-")


def data_image_ref(registry: str, domain: str, name: str, tag: str) -> str:
    return f"{registry.rstrip('/')}/{_slug(domain, name)}-data:{tag}"


def _docker(*args: str) -> None:
    cmd = ["docker", *args]
    logger.info("$ {}", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:2])} failed (rc={proc.returncode})")


def build_data_image(
    *,
    data_root: Path,
    domain: str,
    name: str,
    registry: str,
    tag: str = "latest",
    variant: str = "base",
    base_image: str = "busybox:latest",
    venv_base_image: str = "python:3.12",
    pip_index: str | None = "https://pypi.tuna.tsinghua.edu.cn/simple",
    push: bool = False,
) -> str:
    """Build (and optionally push) the single data image for one task.

    If the task ships ``input/runtime_env`` (pyproject/uv.lock) the image is
    built FROM ``venv_base_image`` and its ``.venv`` is pre-populated at build
    time via ``uv sync`` against ``pip_index`` (a China mirror by default), so
    the agent's no-internet sandbox finds a ready venv and never runs uv sync.
    The runtime image the agent lives in must share the same interpreter path
    (``python:3.12`` → ``/usr/local/bin/python3.12``) for the relocated venv to
    resolve. Tasks without runtime_env use the tiny ``base_image``.
    """
    ctx = data_root / domain / name
    variant_dir = ctx / variant
    for required, label in (
        (variant_dir / "input", "input"),
        (variant_dir / "reference", "reference"),
    ):
        if not required.is_dir():
            raise FileNotFoundError(f"{domain}/{name}: missing {label} dir: {required}")
    has_software = (variant_dir / "software").is_dir()
    runtime_env = variant_dir / "input" / "runtime_env"
    bake_venv = (runtime_env / "pyproject.toml").is_file() or (
        runtime_env / "uv.lock"
    ).is_file()

    baked = f"{BAKED_DATA_ROOT}/{domain}/{name}/{variant}"
    lines = [f"FROM {bake_venv and venv_base_image or base_image}"]
    if bake_venv:
        idx = f" UV_DEFAULT_INDEX={pip_index}" if pip_index else ""
        pip_i = f" -i {pip_index}" if pip_index else ""
        lines += [
            f"RUN pip install --no-cache-dir{pip_i} uv",
            f"COPY {variant}/input {baked}/input",
            # Build the venv IN PLACE at the baked path; the wrapper skips uv
            # sync when .venv/bin/python already exists.
            f"RUN cd {baked}/input/runtime_env &&{idx} "
            f"UV_LINK_MODE=copy uv sync --frozen "
            f"|| ({idx} UV_LINK_MODE=copy uv sync)",
        ]
    else:
        lines += [f"COPY {variant}/input {baked}/input"]
    lines += [f"COPY {variant}/reference {baked}/reference"]
    if has_software:
        chmod = (
            f"RUN find {baked}/software -type f -exec chmod +x {{}} +"
        )
        lines += [f"COPY {variant}/software {baked}/software", chmod]

    build_dir = ctx / _BUILD_DIR
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        dockerfile = build_dir / "Dockerfile.data"
        dockerfile.write_text("\n".join(lines) + "\n")
        image = data_image_ref(registry, domain, name, tag)
        _docker("build", "-f", str(dockerfile), "-t", image, str(ctx))
        if push:
            _docker("push", image)
        return image
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
