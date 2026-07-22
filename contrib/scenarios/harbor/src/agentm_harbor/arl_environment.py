"""ARL environment defaults tailored for AgentM's Harbor runs."""

from __future__ import annotations

from pathlib import Path

from arl.harbor import ArlEnvironment as BaseArlEnvironment
from harbor.environments.base import ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

_DEFAULT_EXEC_TIMEOUT_SECONDS = 1800


class ArlEnvironment(BaseArlEnvironment):
    """ARL environment with a longer default for unbounded Harbor exec calls.

    Harbor's verifier does not pass ``timeout_sec`` to the environment. The
    upstream ARL adapter otherwise limits those calls to a 300-second command
    window plus 120 seconds for recovery, even when Harbor's verifier timeout
    multiplier is much larger.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *,
        default_exec_timeout_seconds: int = _DEFAULT_EXEC_TIMEOUT_SECONDS,
        **kwargs: object,
    ) -> None:
        if default_exec_timeout_seconds <= 0:
            raise ValueError("default_exec_timeout_seconds must be positive")
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            **kwargs,  # type: ignore[arg-type]  # heterogeneous upstream kwargs
        )
        self._default_exec_timeout_seconds = default_exec_timeout_seconds

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        effective_timeout = (
            self._default_exec_timeout_seconds if timeout_sec is None else timeout_sec
        )
        return await super().exec(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=effective_timeout,
            user=user,
        )
