"""LHTB adapter (Long-Horizon Terminal-Bench, github.com/zli12321/LHTB).

Standard Harbor task layout with two bench-specific conventions:
solved means reward >= 0.95 (not 1.0), and each task.toml declares a
``[verifier] timeout_sec`` that can exceed the CLI default (some
verifiers replay hundreds of authenticated moves). Point ``--repo``
at the checkout's ``tasks/`` directory — all 46 tasks ship prebuilt
docker.io/opspai images pulled via the Guangzhou mirror.

How to run::

    uv run agentm-eval sandbox batch \\
      --bench lhtb \\
      --repo /home/ddq/AoyangSpace/LHTB/tasks \\
      --model azure-gpt \\
      -j 5 -n 1

DEFAULT_REGISTRY and DEFAULT_SOURCE_IMAGES are set so ``--source-images``
and ``--registry pair-cn-guangzhou.cr.volces.com`` are not needed on the
CLI. Images resolve to ``pair-cn-guangzhou.cr.volces.com/opspai/lhtb-{task}:{tag}``.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from ..bench import TaskSpec, image_name
from .base import HarborAdapter


def _dockerfile_lines(dockerfile: Path) -> list[str]:
    """Dockerfile lines with backslash continuations joined."""
    joined: list[str] = []
    buf = ""
    for raw in dockerfile.read_text().splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        joined.append(buf + line)
        buf = ""
    if buf:
        joined.append(buf)
    return joined


def _dockerfile_env(dockerfile: Path) -> dict[str, str]:
    """Parse ``ENV k=v ...`` assignments (with line continuations)."""
    import shlex

    env: dict[str, str] = {}
    for line in _dockerfile_lines(dockerfile):
        stripped = line.strip()
        if not stripped.upper().startswith("ENV "):
            continue
        try:
            parts = shlex.split(stripped[4:])
        except ValueError:
            continue
        for part in parts:
            if "=" in part:
                key, _, value = part.partition("=")
                env[key] = value
    return env


def _dockerfile_entrypoint_cmd(dockerfile: Path) -> tuple[list[str], list[str]]:
    """Last ENTRYPOINT and CMD of a Dockerfile (exec or shell form)."""
    import json as _json

    entrypoint: list[str] = []
    cmd: list[str] = []
    for line in _dockerfile_lines(dockerfile):
        stripped = line.strip()
        upper = stripped.upper()
        for keyword, slot in (("ENTRYPOINT", "entrypoint"), ("CMD", "cmd")):
            if not upper.startswith(keyword + " ") and upper != keyword:
                continue
            rest = stripped[len(keyword):].strip()
            value: list[str]
            if rest.startswith("["):
                try:
                    parsed = _json.loads(rest)
                    value = [str(p) for p in parsed] if isinstance(parsed, list) else []
                except ValueError:
                    value = []
            elif rest:
                value = ["/bin/sh", "-c", rest]
            else:
                value = []
            if slot == "entrypoint":
                entrypoint = value
            else:
                cmd = value
    return entrypoint, cmd


class LhtbAdapter(HarborAdapter):
    """LHTB adapter.

    Compose sidecars (e.g. chess-mate's isolated Stockfish referee) map to
    ARL private containers: every non-``main`` service in the task's
    ``environment/docker-compose.yaml`` runs as a gateway-managed container
    in the same pod, named ``{prefix}-{task}-{service}:{tag}`` (build and
    push it from ``Dockerfile.{service}`` first). Pod containers share one
    network namespace, so compose service hostnames collapse to localhost;
    agent-Dockerfile ENV values that reference ``://{service}`` are
    rewritten accordingly and injected as main-container env overrides.
    """

    DEFAULT_REGISTRY = "pair-cn-guangzhou.cr.volces.com"
    DEFAULT_SOURCE_IMAGES = True

    SOLVED_THRESHOLD = 0.95

    def is_pass(self, result: dict) -> bool:
        reward = result.get("reward")
        return isinstance(reward, (int, float)) and reward >= self.SOLVED_THRESHOLD

    RESERVED_POD_PORTS = frozenset({8080})
    PORT_REMAP_OFFSET = 10000

    def _remapped_port(self, port: int) -> int:
        if port in self.RESERVED_POD_PORTS:
            return port + self.PORT_REMAP_OFFSET
        return port

    def _compose_sidecar_services(self, task: TaskSpec) -> list[str]:
        compose = Path(task.path) / "environment" / "docker-compose.yaml"
        if not compose.is_file():
            return []
        import yaml

        try:
            doc = yaml.safe_load(compose.read_text()) or {}
        except yaml.YAMLError as exc:
            logger.warning("lhtb: unparseable compose file for {}: {}", task.name, exc)
            return []
        services = doc.get("services", {})
        if not isinstance(services, dict):
            return []
        return [s for s in services if s != "main"]

    def _sidecar_image(
        self, task: TaskSpec, svc: str, registry: str, prefix: str, tag: str
    ) -> str:
        """Sidecar image reference.

        Derived from the main source image (repo+tag, ``-{svc}`` appended),
        not the build convention, so ``opspai/lhtb-chess-mate:20260713``
        yields sidecar ``opspai/lhtb-chess-mate-game:20260713``.
        """
        src = self.get_source_image(task)
        if not src:
            return image_name(f"{task.name}-{svc}", registry, prefix, tag)
        repo, sep, src_tag = src.rpartition(":")
        ref = f"{repo}-{svc}:{src_tag}" if sep and "/" not in src_tag else f"{src}-{svc}"
        return f"{registry}/{ref}" if registry else ref

    def get_sidecar_containers(
        self, task: TaskSpec, registry: str, prefix: str, tag: str
    ) -> list[dict]:
        sidecars: list[dict] = []
        for svc in self._compose_sidecar_services(task):
            spec: dict = {
                "name": svc,
                "image": self._sidecar_image(task, svc, registry, prefix, tag),
            }
            env_overrides: dict[str, str] = {}
            svc_dockerfile = Path(task.path) / "environment" / f"Dockerfile.{svc}"
            if svc_dockerfile.is_file():
                for key, value in _dockerfile_env(svc_dockerfile).items():
                    if not value.isdigit() or not key.upper().endswith("PORT"):
                        continue
                    remapped = self._remapped_port(int(value))
                    if remapped != int(value):
                        env_overrides[key] = str(remapped)
                entrypoint, cmd = _dockerfile_entrypoint_cmd(svc_dockerfile)
                if entrypoint:
                    spec["command"] = entrypoint
                    if cmd:
                        spec["args"] = cmd
                elif cmd:
                    spec["command"] = cmd
            if env_overrides:
                spec["env"] = env_overrides
            sidecars.append(spec)
        return sidecars

    def get_main_env(self, task: TaskSpec) -> dict[str, str]:
        """Rewrite compose-service URLs in agent-image ENV to localhost."""
        services = self._compose_sidecar_services(task)
        if not services:
            return {}
        dockerfile = Path(task.path) / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            return {}

        def rewrite_url(match: re.Match[str]) -> str:
            port = int(match.group("port"))
            return f"://localhost:{self._remapped_port(port)}"

        overrides: dict[str, str] = {}
        for key, value in _dockerfile_env(dockerfile).items():
            rewritten = value
            for svc in services:
                rewritten = re.sub(
                    rf"://{re.escape(svc)}:(?P<port>\d+)", rewrite_url, rewritten
                )
                rewritten = rewritten.replace(f"://{svc}/", "://localhost/")
            if rewritten != value:
                overrides[key] = rewritten
        return overrides
