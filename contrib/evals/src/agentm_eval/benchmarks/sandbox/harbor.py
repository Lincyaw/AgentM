"""Harbor adapter (formerly Terminal Bench 2.0): task.toml + instruction.md + tests/test.sh."""

from __future__ import annotations

import os
import re
import shlex
import tomllib
from pathlib import Path

from loguru import logger

from .bench import TaskSpec, image_name, upload_file_to_sandbox


def _patch_test_sh(content: bytes) -> bytes:
    """No-op: sandbox has network access, let test.sh install its own deps."""
    return content


def _source_image_from_dockerfile(task_dir: Path) -> str:
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return ""
    for raw_line in dockerfile.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line, comments=True)
        if not parts or parts[0].upper() != "FROM":
            continue
        image_idx = 1
        while image_idx < len(parts) and parts[image_idx].startswith("--"):
            image_idx += 1
        if image_idx < len(parts):
            return parts[image_idx]
    return ""


class HarborAdapter:
    """Harbor format: task.toml + instruction.md + environment/ dir."""

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

            image = ""
            env_section = meta.get("environment", {})
            if isinstance(env_section, dict):
                image = env_section.get("docker_image", "")
            if not image:
                image = _source_image_from_dockerfile(task_dir)

            metadata = meta.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            tasks.append(TaskSpec(
                name=task_dir.name,
                prompt=prompt,
                image=image,
                path=str(task_dir),
                difficulty=metadata.get("difficulty", meta.get("difficulty", "")),
                category=metadata.get("category", meta.get("category", "")),
                extra=meta,
            ))
        return tasks

    def get_image(self, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
        return image_name(task.name, registry, prefix, tag)

    def get_source_image(self, task: TaskSpec) -> str | None:
        return task.image or None

    def supports_build(self) -> bool:
        return True

    def _verifier_env(self, task: TaskSpec) -> dict[str, str]:
        """Env vars exported to test.sh: task.toml ``[verifier.env]``
        (standard Harbor schema field — e.g. chess-mate's RESULT_TOKEN gates
        the referee's /result endpoint). ``${VAR}``/``${VAR:-default}``
        references expand against the host environment; literals pass
        through unchanged."""
        raw = task.extra.get("verifier", {})
        declared = raw.get("env", {}) if isinstance(raw, dict) else {}
        env: dict[str, str] = {}
        for key, value in declared.items():
            expanded = _expand_host_env(str(value))
            if expanded:
                env[key] = expanded
        return env

    def eval_timeout_for(self, task: TaskSpec, timeout: int) -> int:
        """Effective verifier timeout: task.toml ``[verifier] timeout_sec``
        extends the CLI value when larger (standard Harbor schema field).
        The runner also uses this to size the eval HTTP request timeout."""
        verifier = task.extra.get("verifier", {})
        task_timeout = verifier.get("timeout_sec") if isinstance(verifier, dict) else None
        if isinstance(task_timeout, (int, float)) and task_timeout > timeout:
            return int(task_timeout)
        return timeout

    def get_sidecar_containers(
        self, task: TaskSpec, registry: str, prefix: str, tag: str
    ) -> list[dict]:
        """Extra pod containers to run beside the agent. Base Harbor: none."""
        return []

    def get_main_env(self, task: TaskSpec) -> dict[str, str]:
        """Env overrides for the agent (main) container. Base Harbor: none."""
        return {}

    def get_resources(self, task: TaskSpec) -> dict[str, str]:
        """Operations-config resource fields from task.toml ``[environment]``.

        Harbor tasks declare ``cpus``/``memory_mb`` sizing (compile-heavy
        tasks ask for 4 CPU / 8 GiB); without forwarding them the pod gets
        pool defaults and those tasks OOM or crawl. Requests and limits are
        set equal for run-to-run comparability.
        """
        env_section = task.extra.get("environment", {})
        if not isinstance(env_section, dict):
            return {}
        fields: dict[str, str] = {}
        cpus = env_section.get("cpus")
        if isinstance(cpus, (int, float)) and cpus > 0:
            fields["cpu_request"] = str(int(cpus))
            fields["cpu_limit"] = str(int(cpus))
        memory_mb = env_section.get("memory_mb")
        if isinstance(memory_mb, (int, float)) and memory_mb > 0:
            fields["memory_request"] = f"{int(memory_mb)}Mi"
            fields["memory_limit"] = f"{int(memory_mb)}Mi"
        return fields

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        timeout = self.eval_timeout_for(task, timeout)
        task_dir = Path(task.path)

        tests_dir = task_dir / "tests"
        if tests_dir.is_dir():
            session.execute([{  # type: ignore[attr-defined]
                "name": "prep",
                "command": ["bash", "-lc", "mkdir -p /tests /logs/verifier /app/_eval_staging"],
                "work_dir": "/app",
            }])
            for f in tests_dir.rglob("*"):
                if not f.is_file():
                    continue
                content = f.read_bytes()
                if f.name == "test.sh":
                    content = _patch_test_sh(content)
                rel = str(f.relative_to(tests_dir))
                upload_file_to_sandbox(session, f"_eval_staging/{rel}", content)
            session.execute([{  # type: ignore[attr-defined]
                "name": "mv-tests",
                "command": ["bash", "-lc",
                    'cd /app/_eval_staging && find . -type f | while read f; do '
                    'mkdir -p "/tests/$(dirname "$f")" && '
                    'mv "$f" "/tests/$f" && chmod +x "/tests/$f"; done && '
                    'rm -rf /app/_eval_staging'],
                "work_dir": "/app",
            }])

        session.execute([{  # type: ignore[attr-defined]
            "name": "prep-eval",
            "command": ["bash", "-lc", "mkdir -p /logs/verifier"],
            "work_dir": "/app",
        }])

        env_prefix = "".join(
            f"{k}={shlex.quote(str(v))} " for k, v in self._verifier_env(task).items()
        )
        # timeoutSeconds is REQUIRED for verifiers that run longer than the
        # gateway's default sidecar-call timeout (5 min): without it the
        # gateway cancels the exec mid-run and returns empty stdout with no
        # reward.txt, which looks like an invalid trial. Pass the effective
        # eval timeout so the gateway extends its own deadline to match.
        r = session.execute([{  # type: ignore[attr-defined]
            "name": "eval",
            "command": ["bash", "-lc",
                f"{env_prefix}timeout {timeout} bash /tests/test.sh 2>&1"],
            "work_dir": "/app",
            "timeoutSeconds": timeout,
        }], recover_timeout=timeout + 120)
        eval_out = r.results[0].output.stdout

        reward = None
        try:
            r2 = session.execute([{  # type: ignore[attr-defined]
                "name": "read-reward",
                "command": ["bash", "-lc", "cat /logs/verifier/reward.txt 2>/dev/null"],
                "work_dir": "/app",
            }])
            txt = r2.results[0].output.stdout.strip()
            if txt:
                reward = float(txt)
        except Exception as exc:  # noqa: S110
            logger.debug("Failed to read reward.txt: {}", exc)

        return {
            "reward": reward,
            "eval_output": eval_out or "",
        }

    def is_pass(self, result: dict) -> bool:
        reward = result.get("reward")
        return isinstance(reward, (int, float)) and reward >= 1.0

    def format_score_line(self, r: dict) -> str:
        name = r.get("task", "?")
        tools = r.get("tools", "?")
        status = r.get("status", "?").upper()
        reward = r.get("reward")
        reward_str = f"{reward:.2f}" if isinstance(reward, (int, float)) else "-"
        return f"  [{status}] {name} tools={tools} reward={reward_str}"

    def summary_header(self) -> str:
        return (
            f"  {'Task':<30} {'Reward':<12}\n"
            f"  {'-' * 45}"
        )

    def summary_row(self, name: str, r: dict) -> str:
        reward = r.get("reward")
        reward_str = f"{reward:.2f}" if isinstance(reward, (int, float)) else "-"
        return f"  {name:<30} {reward_str:<12}"

    def summary_footer(self, results: dict[str, dict]) -> str:
        scored: list[float] = [
            float(r["reward"]) for r in results.values()
            if isinstance(r.get("reward"), (int, float))
        ]
        if scored:
            avg = sum(scored) / len(scored)
            return f"\nAvg reward: {avg:.3f} ({len(scored)}/{len(results)} scored)"
        return ""

    def pass_at_k_header(self) -> str:
        return (
            f"  {'Task':<30} {'Best':<8} {'Pass':<8} {'Avg reward'}\n"
            f"  {'-' * 60}"
        )

    def pass_at_k_row(self, name: str, runs: list[dict]) -> tuple[str, dict]:
        rewards = [
            float(r["reward"]) for r in runs
            if isinstance(r.get("reward"), (int, float))
        ]
        best = max(rewards) if rewards else None
        avg = sum(rewards) / len(rewards) if rewards else None
        any_pass = any(self.is_pass(r) for r in runs)

        best_str = f"{best:.2f}" if best is not None else "-"
        pass_str = "YES" if any_pass else "no"
        avg_str = f"{avg:.3f}" if avg is not None else "-"
        line = f"  {name:<30} {best_str:<8} {pass_str:<8} {avg_str}"
        stats = {"any_pass": any_pass, "avg_reward": avg, "best_reward": best}
        return line, stats

    def pass_at_k_footer(self, all_stats: list[dict], n_tasks: int) -> str:
        pass_count = sum(1 for s in all_stats if s.get("any_pass"))
        avg_rewards = [s["avg_reward"] for s in all_stats if s.get("avg_reward") is not None]
        lines = [f"\n  Overall pass@k: {pass_count}/{n_tasks} = {pass_count / n_tasks:.1%}"]
        if avg_rewards:
            lines.append(f"  Avg reward:     {sum(avg_rewards) / len(avg_rewards):.3f}")
        return "\n".join(lines)


class LhtbAdapter(HarborAdapter):
    """LHTB (Long-Horizon Terminal-Bench, github.com/zli12321/LHTB).

    Standard Harbor task layout with two bench-specific conventions:
    solved means reward >= 0.95 (not 1.0), and each task.toml declares a
    ``[verifier] timeout_sec`` that can exceed the CLI default (some
    verifiers replay hundreds of authenticated moves). Point ``--repo``
    at the checkout's ``tasks/`` directory and run with
    ``--source-images`` — all 46 tasks ship prebuilt docker.io images.

    Compose sidecars (e.g. chess-mate's isolated Stockfish referee) map to
    ARL private containers: every non-``main`` service in the task's
    ``environment/docker-compose.yaml`` runs as a gateway-managed container
    in the same pod, named ``{prefix}-{task}-{service}:{tag}`` (build and
    push it from ``Dockerfile.{service}`` first). Pod containers share one
    network namespace, so compose service hostnames collapse to localhost;
    agent-Dockerfile ENV values that reference ``://{service}`` are
    rewritten accordingly and injected as main-container env overrides.
    """

    SOLVED_THRESHOLD = 0.95

    def is_pass(self, result: dict) -> bool:
        reward = result.get("reward")
        return isinstance(reward, (int, float)) and reward >= self.SOLVED_THRESHOLD

    # The pod's network namespace is shared with ARL infra containers; the
    # executor's HTTP API already listens on 8080, so a sidecar that binds a
    # reserved port silently loses the race. Remap it out of the way and keep
    # main-container URLs and the sidecar's own *_PORT env consistent.
    RESERVED_POD_PORTS = frozenset({8080})
    PORT_REMAP_OFFSET = 10000

    def _remapped_port(self, port: int) -> int:
        if port in self.RESERVED_POD_PORTS:
            return port + self.PORT_REMAP_OFFSET
        return port

    def _compose_sidecar_services(self, task: TaskSpec) -> list[str]:
        """Non-main services declared in the task's compose file, if any."""
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

        A compose sidecar is pushed alongside its main image and shares that
        image's repo and tag, with the service name appended before the tag:
        ``opspai/lhtb-chess-mate:20260713`` -> ``opspai/lhtb-chess-mate-game:
        20260713``. Deriving from the main source image (rather than the
        build-time ``{prefix}-{task}-{svc}:{tag}`` convention) keeps the
        sidecar pinned to the same registry namespace and rebuild tag as main,
        so an ``opspai/``-prefixed main image yields an ``opspai/`` sidecar.
        Falls back to the build convention when the task declares no source
        image.
        """
        src = self.get_source_image(task)
        if not src:
            return image_name(f"{task.name}-{svc}", registry, prefix, tag)
        repo, sep, src_tag = src.rpartition(":")
        # A ':' only delimits a tag when the trailing part has no '/' (else it
        # is a registry host:port, and the ref carries no tag).
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
                # The gateway replaces an unset container command with
                # `sleep infinity` (private containers were designed for
                # exec-style evaluators), which silently disables a serving
                # sidecar. Pass the image's own ENTRYPOINT/CMD explicitly.
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
        """Rewrite compose-service URLs in agent-image ENV to localhost.

        In-pod containers share the network namespace, so ``http://game:8080``
        from the compose network becomes ``http://localhost:18080`` (the port
        also remapped away from ports the ARL executor reserves).
        """
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


_ENV_REF = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def _expand_host_env(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` against the host environment."""
    return _ENV_REF.sub(
        lambda m: os.environ.get(m.group("name")) or (m.group("default") or ""),
        value,
    )


class SeniorSweAdapter(HarborAdapter):
    """senior-swe-bench (github.com/snorkel-ai/senior-swe-bench-v2026.06).

    Harbor layout, but task.toml declares no registry image — environments are
    built locally from ``environment/Dockerfile`` and pushed under the
    ``{registry}/{prefix}-{task}:{tag}`` convention (``--source-images`` is
    unsupported). The verifier is a multi-stage pipeline (tests + LLM rubric,
    taste, and validation-agent judges) that reads its judge credentials and
    model overrides from ``[verifier.env]`` in task.toml; values are expanded
    against the HOST environment at eval time, so export judge creds
    (e.g. ``OPENAI_API_KEY``/``OPENAI_BASE_URL`` + ``SSB_OVERRIDE_*``) before
    running. ``reward.txt`` is binary 0/1; an EMPTY reward.txt means the
    verifier infrastructure failed (invalid trial), which is surfaced as
    ``invalid_trial`` instead of a silent 0.
    """

    # Provider vars the judges' litellm needs but task.toml's [verifier.env]
    # does not declare (its allowlist predates non-big-three providers), or
    # declares as an empty ``${VAR:-}`` default that shadows the host value.
    # Forwarded from the host when the host value is non-empty.
    EXTRA_VERIFIER_ENV = (
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    )

    # llm_utils.have_credentials() gates the judges on one of these being set;
    # a DeepSeek-only setup would otherwise be silently skipped.
    _JUDGE_CREDENTIAL_VARS = ("PORTKEY_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")

    # The judge pipeline (workspace build + rubric + taste + optional
    # validation agent, each an LLM round-trip) routinely runs many minutes,
    # well past task.toml's timeout_sec. If the eval step's own timeout fires
    # first, ARL kills test.sh before it writes reward.txt and the empty file
    # reads as an invalid trial. Floor the eval timeout above the real cost.
    _MIN_EVAL_TIMEOUT = 2400

    def get_source_image(self, task: TaskSpec) -> str | None:
        return None

    def eval_timeout_for(self, task: TaskSpec, timeout: int) -> int:
        return max(super().eval_timeout_for(task, timeout), self._MIN_EVAL_TIMEOUT)

    def _verifier_env(self, task: TaskSpec) -> dict[str, str]:
        env = super()._verifier_env(task)
        # task.toml expands undeclared/optional keys to "" -- treat empty as
        # absent so a real host value wins.
        for key in self.EXTRA_VERIFIER_ENV:
            if not env.get(key) and os.environ.get(key):
                env[key] = os.environ[key]
        # The judges accept a DeepSeek slug via SSB_OVERRIDE_* but their
        # credential gate only recognizes PORTKEY/ANTHROPIC/OPENAI. Mirror the
        # DeepSeek key into OPENAI_API_KEY to pass the gate; litellm still
        # authenticates the deepseek/ model through DEEPSEEK_API_KEY.
        if env.get("DEEPSEEK_API_KEY") and not any(
            env.get(k) for k in self._JUDGE_CREDENTIAL_VARS
        ):
            env["OPENAI_API_KEY"] = env["DEEPSEEK_API_KEY"]
        return env

    # The judges' litellm code path imports fastapi + orjson (and more), none
    # of which are in the task image or the verifier-requirements.txt test.sh
    # installs — so every judge call dies with ModuleNotFoundError and the run
    # is scored as an invalid trial. Installing the litellm[proxy] extra pulls
    # the whole set in one shot; verified end-to-end that it makes both the
    # rubric and taste judges run. Pre-install before test.sh (tolerant; the
    # image's pip predates --break-system-packages, so fall back to plain).
    VERIFIER_PIP_DEPS = ("litellm[proxy]",)

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        if self.VERIFIER_PIP_DEPS:
            deps = " ".join(shlex.quote(d) for d in self.VERIFIER_PIP_DEPS)
            session.execute([{  # type: ignore[attr-defined]
                "name": "verifier-deps",
                "command": ["bash", "-lc",
                    f"python3 -m pip install --break-system-packages -q {deps} "
                    f"2>/dev/null || python3 -m pip install -q {deps} 2>/dev/null || true"],
                "work_dir": "/app",
                "timeoutSeconds": 600,
            }], recover_timeout=720)
        result = super().evaluate(session, task, timeout=timeout)
        if result.get("reward") is None:
            result["invalid_trial"] = True
        return result
