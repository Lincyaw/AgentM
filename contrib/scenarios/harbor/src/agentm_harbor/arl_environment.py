"""ARL environment defaults tailored for AgentM's Harbor runs."""

from __future__ import annotations

import time
from pathlib import Path

from arl.harbor import ArlEnvironment as BaseArlEnvironment
from harbor.environments.base import ExecResult
from harbor.models.task.config import EnvironmentConfig, TaskConfig
from harbor.models.trial.paths import TrialPaths
from loguru import logger

from agentm_harbor.trajectory_replay import read_actions, replay

#: Only reached when the task declares no verifier budget of its own. High
#: enough that it is never the thing that decides a benchmark verifier's fate.
_FALLBACK_EXEC_TIMEOUT_SECONDS = 7200

#: Added to the task's declared budget so this cap always sits outside Harbor's
#: own ``asyncio.wait_for``. Whichever of the two fires first decides how a
#: cut-off verifier is reported, and Harbor's is the one that reports it as a
#: timeout rather than as a missing file.
_EXEC_TIMEOUT_MARGIN_SECONDS = 300


class ArlEnvironment(BaseArlEnvironment):
    """ARL environment with a longer default for unbounded Harbor exec calls.

    Harbor's verifier does not pass ``timeout_sec`` to the environment. The
    upstream ARL adapter otherwise limits those calls to a 300-second command
    window plus 120 seconds for recovery, even when Harbor's verifier timeout
    multiplier is much larger.

    A flat default is the wrong shape for that gap. The gateway kills the step
    when the window closes and hands back an ordinary non-zero result, so a
    truncated ``test.sh`` is indistinguishable from one that ran to completion
    and scored badly -- except that the reward file it writes last is missing,
    which surfaces as ``RewardFileNotFoundError`` and reads like a broken task.
    The window is therefore taken from the same numbers Harbor bounds the
    verifier with, and a command that consumes the whole of it is logged.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *,
        default_exec_timeout_seconds: int | str | None = None,
        verifier_timeout_multiplier: float | str = 1.0,
        trajectory_dsn: str = "",
        trajectory_schema: str = "",
        trajectory_session: str = "",
        trajectory_up_to_turn: int | str | None = None,
        **kwargs: object,
    ) -> None:
        declared_timeout = _opt_int(default_exec_timeout_seconds)
        self._exec_timeout = (
            _task_exec_timeout(environment_dir, float(verifier_timeout_multiplier))
            if declared_timeout is None
            else declared_timeout
        )
        if self._exec_timeout <= 0:
            raise ValueError("default_exec_timeout_seconds must be positive")
        self._replay_dsn = trajectory_dsn
        self._replay_schema = trajectory_schema
        self._replay_session = trajectory_session
        self._replay_up_to_turn = _opt_int(trajectory_up_to_turn)
        if self._replay_up_to_turn is not None and not (
            self._replay_dsn and self._replay_schema and self._replay_session
        ):
            raise ValueError(
                "trajectory_up_to_turn needs trajectory_dsn, trajectory_schema "
                "and trajectory_session: without all three there is no recording "
                "to rebuild the workspace from"
            )
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            **kwargs,  # type: ignore[arg-type]  # heterogeneous upstream kwargs
        )

    async def start(self, force_build: bool) -> None:
        """Start the sandbox, then put the recorded work back into it.

        After ``super().start()`` rather than instead of it: the image and the
        task's environment directory are what the original attempt began from,
        so they have to be in place before its actions are re-applied.

        Any failure propagates. A trial that silently starts from a workspace
        that is not the one being studied still produces a score, and that score
        would be read as the effect of whatever the run was testing.
        """
        await super().start(force_build)
        if self._replay_up_to_turn is None:
            return

        actions = read_actions(
            self._replay_dsn,
            self._replay_schema,
            self._replay_session,
            up_to_turn=self._replay_up_to_turn,
        )
        logger.info(
            "agentm-harbor: rebuilding {} to turn {} ({} action(s))",
            self._replay_session,
            self._replay_up_to_turn,
            len(actions),
        )
        await replay(self, actions)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        effective_timeout = self._exec_timeout if timeout_sec is None else timeout_sec
        started = time.monotonic()
        result = await super().exec(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=effective_timeout,
            user=user,
        )
        elapsed = time.monotonic() - started
        if result.return_code != 0 and elapsed >= effective_timeout:
            # The gateway kills the step and reports it like any other failure,
            # so without this the caller sees only whatever the command had
            # managed to write. Said plainly here because the symptom it causes
            # downstream -- a missing artefact -- points at the wrong thing.
            logger.warning(
                "agentm-harbor: command cut off after {:.0f}s at the {}s window; "
                "anything it writes at the end is missing: {}",
                elapsed,
                effective_timeout,
                command[:120],
            )
        return result


def _opt_int(value: int | str | None) -> int | None:
    """A Harbor-supplied number, or ``None`` when it said nothing.

    Harbor passes task config through as strings, so every numeric knob arrives
    as ``int | str``, and an unset one as either ``None`` or empty. Said once
    here rather than at each parameter.
    """
    return None if value in (None, "") else int(value)  # type: ignore[arg-type]


def _task_exec_timeout(environment_dir: Path, multiplier: float) -> int:
    """The window a command gets, from what the task says its verifier needs.

    Harbor bounds the verifier at ``[verifier] timeout_sec`` times the run's
    multiplier; a shorter window here would end ``test.sh`` before Harbor has
    any say, and the two report the same event very differently.

    Read through Harbor's own model rather than parsed by hand: the budget this
    aligns with is whatever ``TaskConfig`` says it is, so reading it any other
    way is a second opinion about the same file -- and a second opinion that
    drifts low reintroduces exactly the misreport this window exists to remove.
    """
    config = environment_dir.parent / "task.toml"
    try:
        declared = TaskConfig.model_validate_toml(
            config.read_text(encoding="utf-8")
        ).verifier.timeout_sec
    except (OSError, ValueError) as exc:
        logger.warning(
            "agentm-harbor: no verifier budget in {} ({}); allowing {}s per command",
            config,
            exc,
            _FALLBACK_EXEC_TIMEOUT_SECONDS,
        )
        return _FALLBACK_EXEC_TIMEOUT_SECONDS

    if declared <= 0:
        return _FALLBACK_EXEC_TIMEOUT_SECONDS
    return int(declared * max(multiplier, 1.0)) + _EXEC_TIMEOUT_MARGIN_SECONDS
