# code-health: ignore-file[AM025] -- harbor and the verifier write untyped JSON;
# the isinstance checks here are the one place it becomes typed.
"""Senior SWE-Bench, run through Harbor on an ARL cluster.

Everything this benchmark knows about itself lives here: that a verifier writes
``verifier_results.json`` and ``validation_results.json``, that an assertion id
can be a wrapper around a whole crate suite whose per-test detail is only in
``runner_*.log``, that a story's real assertion is in its script rather than its
name, that re-running means stacking an ARL workspace fork under a Harbor
conversation fork.

Nothing outside this file and its environment adapter should learn any of it.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import tarfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from agentm.core.abi import JsonValue

from ..contracts import (
    Assertion,
    EvidenceRef,
    FailureCase,
    Metrics,
    json_int,
    json_str,
)
from ..environments.arl import (
    ArlFork,
    ArlImage,
    ArlSandbox,
    SandboxUnavailable,
    open_image_sandbox_async,
    open_sandbox_async,
)
from ..protocols import ReplayOutcome, RestoredEnvironment

NAME = "senior-swe"

#: Where these images keep the checkout -- one directory named after the
#: upstream project, which differs per task, so the exact path is found rather
#: than assumed.
REPO_ROOT = "/repo"

_BUNDLE_PATH = "/tmp/diagnosis-evidence.tar.gz"

#: Which metric decides whether the task was solved. The others are still
#: recorded: a check that moves the story pass rate without flipping this is
#: progress, and treating it as nothing retires working checks.
PRIMARY_METRIC = "reward"

DEFAULT_BENCH_ROOT = Path.home() / ".agentm" / "bench-repos" / "senior-swe" / "tasks"

#: Signatures of a measurement that did not happen, mapped to why. Order
#: matters: the first match wins, so the specific ones come first.
_LOST_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("DownloadVerifierDirError", "runtime_lost"),
    ("runtime lost", "runtime_lost"),
    ("source session", "source_expired"),
    ("VerifierTimeoutError", "verifier_timeout"),
    ("RewardFileNotFoundError", "verifier_incomplete"),
    ("RewardFileEmptyError", "verifier_incomplete"),
    ("GatewayError", "gateway_error"),
    ("TriggerTerminated", "agent_transport_error"),
    ("not found", "source_expired"),
)


# -- read side ----------------------------------------------------------------


def _batch_model(batch_dir: Path) -> str:
    """Which model made this batch's attempts, from the job's own config.

    Read once per batch rather than carried in a flag, so a case can never
    claim a model the run did not use. Absent, the model-scope notes simply
    have nothing to group on -- everything else is unaffected.
    """
    config = batch_dir / "config.json"
    try:
        raw = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.info("senior-swe: no model name in {}: {}", config, exc)
        return ""
    agents = raw.get("agents")
    if not isinstance(agents, list) or not agents:  # code-health: ignore[AM025]
        return ""
    first = agents[0]
    name = (
        first.get("model_name") if isinstance(first, dict) else None
    )  # code-health: ignore[AM025]
    return name if isinstance(name, str) else ""  # code-health: ignore[AM025]


@dataclass(slots=True)
class SeniorSweSource:
    """Discovery over a Harbor jobs directory."""

    bench_root: Path = DEFAULT_BENCH_ROOT
    trajectory_dsn: str = ""
    trajectory_schema: str = ""
    gateway_url: str = ""
    gateway_key: str = ""
    #: Where the task images live. Without these a diagnosis can only be made
    #: from the artefacts once the attempt's own session has expired.
    image_registry: str = ""
    image_tag: str = ""

    @property
    def name(self) -> str:
        return NAME

    def discover(self, batch: str, *, only: str = "") -> list[FailureCase]:
        # Absolute throughout: every locator these cases carry has to keep
        # meaning the same thing after the shell moves into a container, where
        # nothing shares this process's working directory.
        batch_dir = Path(batch).resolve()
        model_name = _batch_model(batch_dir)
        trials = sorted(p for p in batch_dir.iterdir() if (p / "result.json").is_file())
        if not trials:
            logger.warning("senior-swe: no trials with result.json under {}", batch_dir)
            return []

        parsed = [(trial, _read_trial(trial)) for trial in trials]
        cohort: dict[str, list[tuple[Path, _Trial]]] = {}
        for trial, record in parsed:
            if record is not None:
                cohort.setdefault(record.task_name, []).append((trial, record))

        cases: list[FailureCase] = []
        for trial, record in parsed:
            if record is None or (only and only not in record.task_name):
                continue
            if record.metrics.is_pass or not record.metrics.present:
                # No metric at all means the attempt was lost to the harness and
                # has nothing to diagnose; a full pass has nothing to learn.
                continue
            case = self._build(
                trial, record, cohort.get(record.task_name, []), model_name
            )
            if case is not None:
                cases.append(case)

        logger.info(
            "senior-swe: {} graded failure(s) from {} trial(s) in {}",
            len(cases),
            len(trials),
            batch_dir.name,
        )
        return cases

    async def open_environment(self, case: FailureCase) -> RestoredEnvironment | None:
        """The attempt's own machine if it still exists, a rebuild if not.

        Worth trying in that order even though the fork almost always fails: it
        succeeds during a batch, and while it does it is the only place the
        attempt's out-of-patch damage is visible.

        Either way the evidence goes in with it, at the same absolute paths it
        has on this host. That is what lets the locators written by ``discover``
        stay literally true once the shell is inside a container -- nothing has
        to be rewritten, and nothing silently points at a path that no longer
        resolves.
        """
        if not self.gateway_url:
            return None
        # Off the loop: building the bundle is a blocking trajectory fetch plus
        # a gzip of the task tree, seconds of it, and `diagnose` runs several
        # cases at once on this one loop. Left inline it stalls every other
        # in-flight session mid-token for the duration.
        bundle, listing = await asyncio.to_thread(self._evidence_bundle, case)
        image = self._image(case)
        return await self._fork(case, bundle, image) or await self._from_image(
            case, bundle, listing, image
        )

    def _image(self, case: FailureCase) -> str:
        task_dir = case.backend_ref.get("task_dir")
        if not self.image_registry or not isinstance(task_dir, str) or not task_dir:
            return ""
        return f"{self.image_registry}/{Path(task_dir).name}:{self.image_tag}"

    async def _fork(
        self, case: FailureCase, bundle: bytes, image: str
    ) -> RestoredEnvironment | None:
        session = case.backend_ref.get("arl_session_id")
        step = case.backend_ref.get("arl_step")
        if not isinstance(session, str) or not session:
            return None
        try:
            sandbox = await open_sandbox_async(
                ArlFork(
                    session_id=session,
                    step=step if isinstance(step, int) else 0,
                    # Lets a checkpoint outlive its session: the filesystem is
                    # still stored, but once the session record is gone nothing
                    # else records which image to restore it onto.
                    image=image,
                ),
                gateway_url=self.gateway_url,
                api_key=self.gateway_key,
            )
        except SandboxUnavailable as exc:
            logger.info("senior-swe: cannot fork {}: {}", case.case_id, exc)
            return None

        await _unpack(sandbox, bundle)
        checkout = await _prepare(sandbox)
        return RestoredEnvironment(
            sandbox,
            writer=sandbox.writer,
            work_dir=checkout,
            description=(
                "You are on the attempt's own machine, forked at the moment it "
                f"finished. {checkout} holds the repository with its changes "
                "applied, and whatever it installed or started is still running. "
                "What you see is what it left, which makes this the one place "
                "where damage it did outside its patch is visible."
            ),
        )

    async def _from_image(
        self, case: FailureCase, bundle: bytes, listing: str, image: str
    ) -> RestoredEnvironment | None:
        if not image:
            return None
        agent_patch = str(case.backend_ref.get("agent_patch", ""))
        try:
            sandbox = await open_image_sandbox_async(
                ArlImage(
                    image=image,
                    uploads={_BUNDLE_PATH: bundle},
                    setup=(f"tar -xzf {_BUNDLE_PATH} -C / && rm -f {_BUNDLE_PATH}",),
                ),
                gateway_url=self.gateway_url,
                api_key=self.gateway_key,
            )
        except SandboxUnavailable as exc:
            logger.info("senior-swe: no environment for {}: {}", case.case_id, exc)
            return None

        checkout = await _prepare(sandbox)
        return RestoredEnvironment(
            sandbox,
            writer=sandbox.writer,
            work_dir=checkout,
            description=(
                "The attempt's own machine is gone -- these expire a couple of "
                "hours after the batch. You are on a fresh one built from the "
                f"same image, so {checkout} is the repository as the attempt "
                "*found* it, before any of its work.\n\n"
                "What that costs you: anything it did outside its patch -- a "
                "package installed, a service started, a file written elsewhere "
                "-- is not here. Absence of those proves nothing about what it "
                "did.\n\n"
                "What you still have: the code it read, the toolchain, the "
                "graded tests, and the ability to run them. Everything named "
                "under 'Where to look' has been placed at exactly the path "
                f"given, including:\n{listing}\n\n"
                f"{checkout} is a git checkout, so put it in whichever state "
                "answers your question and reset between experiments. Applying "
                f"the agent's patch ({agent_patch or 'see above'}) reproduces "
                "what was graded; leaving it off shows what it was reading when "
                "it decided."
            ),
        )

    def _evidence_bundle(self, case: FailureCase) -> tuple[bytes, str]:
        """Everything the case points at, as a tar rooted at ``/``.

        Absolute paths are preserved deliberately. The alternative -- copying to
        a tidy ``/diagnosis`` and rewriting every locator -- puts a translation
        step between what the prompt says and what is on disk, and a stale
        translation is invisible until an agent reports that a file is missing.
        """
        paths: list[Path] = []
        task_dir = case.backend_ref.get("task_dir")
        if isinstance(task_dir, str) and task_dir:
            paths.append(Path(task_dir))
        verifier = case.backend_ref.get("verifier_dir")
        if isinstance(verifier, str) and verifier:
            paths.append(Path(verifier))

        extra: dict[Path, bytes] = {}
        transcript = self._transcript(case)
        if transcript:
            extra[Path(str(case.backend_ref.get("transcript_path", "")))] = transcript

        return _tar(paths, extra), "\n".join(
            f"  {p}" for p in [*paths, *extra] if str(p)
        )

    def _transcript(self, case: FailureCase) -> bytes:
        """The trajectory, dumped so it survives the trip into a container.

        It lives in a database on this host that nothing inside the sandbox can
        reach, and it is where the answer to "from which step was it already
        wrong" actually is. One line of JSON per turn, in order.
        """
        session = case.backend_ref.get("agentm_session_id")
        if not self.trajectory_dsn or not isinstance(session, str) or not session:
            return b""
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.warning("senior-swe: no psycopg, trajectory not bundled: {}", exc)
            return b""
        query = (
            f'SELECT turn_index, turn_json FROM "{self.trajectory_schema}".'
            "agentm_trajectory_turns WHERE session_id = %s ORDER BY turn_index"
        )
        try:
            with psycopg.connect(self.trajectory_dsn) as conn, conn.cursor() as cur:
                cur.execute(query, (session,))
                rows = cur.fetchall()
        except psycopg.Error as exc:
            logger.warning("senior-swe: cannot read trajectory {}: {}", session, exc)
            return b""
        lines = [
            json.dumps({"turn_index": index, "turn": turn}, ensure_ascii=False)
            for index, turn in rows
        ]
        return ("\n".join(lines) + "\n").encode("utf-8")

    # -- assembling one case --------------------------------------------------

    def _build(
        self,
        trial: Path,
        record: _Trial,
        cohort: Sequence[tuple[Path, _Trial]],
        model_name: str = "",
    ) -> FailureCase | None:
        short = record.task_name.split("/")[-1]
        task_dir = self.bench_root / short
        if not task_dir.is_dir():
            logger.warning("senior-swe: no task dir for {} at {}", short, task_dir)
            return None

        verifier = trial / "verifier"
        agent_patch = verifier / "agent_filtered.patch"
        if not agent_patch.is_file():
            agent_patch = verifier / "agent.patch"

        evidence = [
            EvidenceRef(
                label="reference solution",
                locator=str(task_dir / "tests" / "judge" / "oracle.patch"),
                note="Read it for its idea, not its diff.",
            ),
            EvidenceRef(label="the agent's patch", locator=str(agent_patch)),
            EvidenceRef(label="graded checks", locator=str(task_dir / "tests")),
            EvidenceRef(
                label="per-test detail",
                locator=str(verifier / "runner_*.log"),
                note="An assertion id can be a wrapper around a whole suite; "
                "the individual test results are only here.",
            ),
        ]
        if (verifier / "validation_scripts").is_dir():
            # Not every task has stories. Listing the directory unconditionally
            # sends the reader after something that was never there, and the
            # honest reading of a missing file is that the evidence was lost.
            evidence.append(
                EvidenceRef(
                    label="story scripts and inputs",
                    locator=str(verifier / "validation_scripts")
                    + ", "
                    + str(verifier / "validation_params"),
                    note="A story's name says little. Its script and params say "
                    "exactly what was asserted and with what inputs.",
                )
            )
        transcript = verifier / "trajectory.jsonl"
        if self.trajectory_dsn and record.session_id:
            evidence.append(
                EvidenceRef(
                    label="the trajectory",
                    locator=str(transcript),
                    note=(
                        "One JSON object per line, in order: {turn_index, turn}. "
                        "Inside a turn -- what it said: "
                        "turn.response.content[] with type='text'. What it ran: "
                        "the same array with type='tool_call', each with "
                        "name and arguments. What came back: turn.tool_results[]"
                        ", each result.extras.stdout. Large; read it with jq or "
                        "python rather than opening it whole."
                    ),
                )
            )

        return FailureCase(
            case_id=f"{short}:{record.session_id or trial.name}",
            task_name=short,
            model_name=model_name,
            instruction=_instruction(task_dir),
            failing_assertions=record.failing,
            evidence=tuple(evidence),
            cohort_note=_cohort_note(cohort),
            metrics=record.metrics,
            sibling_metrics=tuple(r.metrics for _, r in cohort),
            backend_ref={
                "task_dir": str(task_dir),
                "verifier_dir": str(verifier),
                "agent_patch": str(agent_patch),
                "transcript_path": str(transcript),
                "arl_session_id": record.arl_session_id,
                "arl_step": record.arl_step,
                "agentm_session_id": record.session_id,
            },
        )


# -- moving the evidence into a sandbox ---------------------------------------


def _tar(paths: Sequence[Path], extra: Mapping[Path, bytes]) -> bytes:
    """A gzipped tar whose members are absolute, so it unpacks at ``/`` onto the
    same paths the case's locators name."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in paths:
            if not path.exists():
                logger.warning("senior-swe: evidence missing: {}", path)
                continue
            archive.add(str(path), arcname=str(path).lstrip("/"))
        for path, content in extra.items():
            if not str(path) or not content:
                continue
            info = tarfile.TarInfo(name=str(path).lstrip("/"))
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


async def _prepare(sandbox: ArlSandbox) -> str:
    """Find the working tree, and point the writer at it.

    Both halves of one fact: the shell and the file tools have to agree on what
    a relative path means, or an edit lands somewhere the agent never looked.

    These images hold one project under ``/repo``, named for the upstream
    repository rather than the task, so the name differs every time and is not
    derivable from anything the case carries. Asking git is both shorter and
    correct if the layout ever changes.
    """
    result = await sandbox.bash.exec(
        f"git -C {REPO_ROOT}/*/ rev-parse --show-toplevel 2>/dev/null | head -1",
        cwd="/",
        timeout=60,
    )
    found = result.stdout.decode("utf-8", "replace").strip()
    if not found:
        logger.warning("senior-swe: no checkout under {}, using it directly", REPO_ROOT)
        found = REPO_ROOT
    sandbox.writer.work_dir = found
    return found


async def _unpack(sandbox: ArlSandbox, bundle: bytes) -> None:
    """Put the evidence on a sandbox that is already running.

    Used for the fork path, where the session exists before there is anywhere to
    hand uploads to. A failure is logged rather than raised: a machine with the
    repository and no artefacts still answers questions the artefacts alone
    cannot.
    """
    if not bundle:
        return
    try:
        await sandbox.client.upload_file(  # type: ignore[attr-defined]
            sandbox.session_id, _BUNDLE_PATH, bundle
        )
    except Exception as exc:  # noqa: BLE001 - any transport failure means the same
        logger.warning("senior-swe: could not upload evidence: {}", exc)
        return
    result = await sandbox.bash.exec(
        f"tar -xzf {_BUNDLE_PATH} -C / && rm -f {_BUNDLE_PATH}", cwd="/", timeout=300
    )
    if result.exit_code != 0:
        logger.warning(
            "senior-swe: could not unpack evidence: {}",
            result.stderr.decode("utf-8", "replace")[:300],
        )


# -- reading what harbor wrote ------------------------------------------------


@dataclass(slots=True)
class _Trial:
    task_name: str
    metrics: Metrics
    arl_session_id: str
    arl_step: int
    session_id: str
    #: Parsed once here, because both the case and the cohort comparison want
    #: them and re-reading the verifier's two JSON files per use kept the ids
    #: and the assertions they came from as two things to hold in step.
    failing: tuple[Assertion, ...] = ()

    @property
    def failing_ids(self) -> tuple[str, ...]:
        return tuple(sorted(a.assertion_id for a in self.failing))


def _read_trial(trial: Path) -> _Trial | None:
    path = trial / "result.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # An empty result.json is a real thing: a full disk truncated several
        # during one batch. Say which, rather than dropping it silently.
        logger.warning("senior-swe: unreadable {}: {}", path, exc)
        return None
    if not isinstance(raw, Mapping):
        return None

    verifier = raw.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    rewards = rewards if isinstance(rewards, Mapping) else {}

    agent = raw.get("agent_result")
    meta = agent.get("metadata") if isinstance(agent, Mapping) else None
    meta = meta if isinstance(meta, Mapping) else {}

    task = raw.get("task_name")
    return _Trial(
        task_name=task if isinstance(task, str) else "",
        metrics=_metrics(rewards),
        arl_session_id=json_str(meta, "arl_session_id"),
        arl_step=json_int(meta, "arl_step"),
        session_id=json_str(meta, "agentm_session_id"),
        failing=tuple(_failing(trial / "verifier")),
    )


def _metrics(rewards: Mapping[str, JsonValue]) -> Metrics:
    """The three graded numbers, keeping absent distinct from zero."""
    values: dict[str, float | None] = {}
    for key in ("reward", "verifier_score", "validation_score"):
        value = rewards.get(key)
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            values[key] = None
        else:
            values[key] = float(value)
    return Metrics(values=values, primary=PRIMARY_METRIC)


def _instruction(task_dir: Path) -> str:
    path = task_dir / "instruction.md"
    if not path.is_file():
        return ""
    # Everything after this heading is boilerplate repeated across every task.
    return path.read_text(encoding="utf-8").split("## General instructions")[0].strip()


def _failing(verifier_dir: Path) -> list[Assertion]:
    found: list[Assertion] = []
    found += _from(verifier_dir / "verifier_results.json", "tests", "test", "failure")
    found += _from(
        verifier_dir / "validation_results.json", "stories", "story", "reason"
    )
    return found


def _from(path: Path, container: str, kind: str, message_key: str) -> list[Assertion]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("senior-swe: unreadable {}: {}", path, exc)
        return []
    entries = raw.get(container) if isinstance(raw, Mapping) else None
    if not isinstance(entries, Mapping):
        return []
    out: list[Assertion] = []
    for name, entry in entries.items():
        if not isinstance(entry, Mapping) or entry.get("pass"):
            continue
        message = entry.get(message_key)
        out.append(
            Assertion(
                kind=kind,
                assertion_id=str(name),
                message=message if isinstance(message, str) else "",
            )
        )
    return out


def _cohort_note(cohort: Sequence[tuple[Path, _Trial]]) -> str:
    """How this task's attempts compare, in this benchmark's terms.

    Matching assertion ids are weaker evidence than they look here: one task's
    graded check is a single wrapper -- the crate's suite exited non-zero -- so
    every attempt reports the identical id while the tests failing underneath
    were four, six and three, in sets that do not contain one another. Where the
    ids agree this points at the runner logs rather than concluding.
    """
    if len(cohort) < 2:
        return ""
    groups = [set(record.failing_ids) for _, record in cohort]
    shared = set.intersection(*groups)
    union = set().union(*groups)

    lines = [
        f"  attempt {i}: {len(group)} failed -- "
        + (", ".join(sorted(group)[:6]) or "(none)")
        for i, group in enumerate(groups, start=1)
    ]
    if len(union) > len(shared):
        lines.append(
            f"  -> the attempts do not agree: {len(shared)} assertion(s) fail "
            f"every time, {len(union)} distinct across attempts. Anything "
            "outside the shared set is a candidate for a flaky harness rather "
            "than a defect; check whether the diff can even reach it."
        )
    elif shared:
        lines.append(
            f"  -> the same {len(shared)} assertion id(s) fail every time, which "
            "is consistent with a deterministic cause. But an assertion here can "
            "wrap a whole suite, in which case the id says nothing about which "
            "tests failed underneath. Before concluding, compare the per-test "
            "detail across attempts:"
        )
        lines += [f"       {trial}/verifier/runner_*.log" for trial, _ in cohort[:6]]
        lines.append(
            "     If the individual failures differ between attempts, or land in "
            "code the diff does not touch, the harness is the variable."
        )
    return "\n".join(lines)


# -- write side ---------------------------------------------------------------


@dataclass(slots=True)
class HarborReplay:
    """Re-run one attempt from its decision point, with a message injected.

    Two forks stacked: ARL restores the workspace at a checkpoint, Harbor
    restores the conversation at a turn, and the injected text is what the agent
    finds waiting when it resumes. Harbor is driven as a subprocess so this
    package never imports the scenario it measures.
    """

    gateway_url: str
    image_registry: str
    image_tag: str
    agentm_home: str
    trajectory_dsn: str
    trajectory_schema: str
    model_name: str = "azure-gpt"
    judge_model: str = "openai/DeepSeek-V4-pro"
    judge_base_url: str = ""
    judge_api_key: str = ""
    verifier_timeout_multiplier: float = 12.0
    agent_timeout_multiplier: float = 5.0
    jobs_root: str = "jobs"
    work_dir: str = ".policy-loop-work"

    @property
    def name(self) -> str:
        return NAME

    def rerun(
        self,
        case: FailureCase,
        *,
        injected: str,
        label: str,
        resume_at: int = -1,
    ) -> ReplayOutcome:
        # Everything that can fail is inside the try, including writing our own
        # scratch files: one arm failing must cost that arm, not the batch. A
        # full disk raising through the caller would discard the measurements
        # already taken.
        work = Path(self.work_dir)
        config_path = work / f"{label}.json"
        log_path = work / f"{label}.log"
        origin = self._origin(case, resume_at)
        if not origin:
            logger.warning(
                "senior-swe: cannot rebuild {} at turn {}: no trajectory to "
                "replay from",
                case.case_id,
                resume_at,
            )
            return ReplayOutcome(Metrics(), lost_reason="cannot_resume")
        try:
            work.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    self._job_config(case, injected, origin, resume_at), indent=2
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("senior-swe: could not write {}: {}", config_path, exc)
            return ReplayOutcome(Metrics(), lost_reason="harness_error")

        command = [
            "python",
            "-m",
            "harbor.cli.main",
            "run",
            "-c",
            str(config_path),
            "--job-name",
            label,
            "-q",
            "-y",
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=7200, check=False
            )
            log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("senior-swe: could not run harbor: {}", exc)
            return ReplayOutcome(
                Metrics(), lost_reason="harness_error", resumed_at=resume_at
            )

        job_dir = Path(self.jobs_root) / label
        metrics, lost = self._read_outcome(job_dir, log_path)
        return ReplayOutcome(
            metrics,
            lost_reason=lost,
            artifact_dir=str(job_dir),
            resumed_at=resume_at,
            review_report=self._review_report(job_dir),
        )

    def _review_report(self, job_dir: Path) -> str:
        """What the review said, if the scenario ran one.

        The run writes its session id into the trial metadata, and a review is
        a child of that session -- so the report is its last message. Read here
        rather than left for later because a job directory is not kept and a
        session id outside one is unusable.
        """
        if not self.trajectory_dsn or not self.trajectory_schema:
            return ""
        results = sorted(job_dir.glob("*/result.json"))
        if not results:
            return ""
        try:
            raw = json.loads(results[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        meta = raw.get("metadata") if isinstance(raw, Mapping) else None
        session = meta.get("agentm_session_id") if isinstance(meta, Mapping) else None
        if not isinstance(session, str) or not session:
            return ""
        try:
            return _last_review_message(
                self.trajectory_dsn, self.trajectory_schema, session
            )
        except Exception as exc:  # noqa: BLE001 -- a missing report is not a lost run
            logger.warning(
                "senior-swe: could not read the review for {}: {}", session, exc
            )
            return ""

    def _job_config(
        self,
        case: FailureCase,
        injected: str,
        origin: Mapping[str, JsonValue],
        resume_at: int,
    ) -> dict[str, JsonValue]:
        ref = case.backend_ref
        agent_env: dict[str, JsonValue] = {
            "AGENTM_HOME": self.agentm_home,
            "AGENTM_TRAJECTORY_DSN": self.trajectory_dsn,
            "AGENTM_TRAJECTORY_SCHEMA": self.trajectory_schema,
            "AGENTM_FORK_FROM_SESSION": str(ref.get("agentm_session_id", "")),
            "AGENTM_FORK_TURN": str(resume_at),
            "AGENTM_FORK_PROMPT": injected,
        }
        notes = self._review_notes(case)
        if notes:
            agent_env["AGENTM_REVIEW_NOTES"] = str(notes)

        verifier_env: dict[str, JsonValue] = {
            "SSB_OVERRIDE_ALL_JUDGE_MODEL": self.judge_model,
            "SSB_OVERRIDE_CLASSIFIER_MODEL": self.judge_model,
            "SSB_OVERRIDE_VA_MODEL": self.judge_model,
        }
        if self.judge_api_key:
            verifier_env["OPENAI_API_KEY"] = self.judge_api_key
            verifier_env["ARK_API_KEY"] = self.judge_api_key
        if self.judge_base_url:
            verifier_env["OPENAI_BASE_URL"] = self.judge_base_url
            verifier_env["OPENAI_API_BASE"] = self.judge_base_url

        return {
            "agent_timeout_multiplier": self.agent_timeout_multiplier,
            "verifier_timeout_multiplier": self.verifier_timeout_multiplier,
            "n_concurrent_trials": 1,
            "environment": {
                "import_path": "agentm_harbor:ArlEnvironment",
                # Keeping the environment is what makes a second arm possible;
                # deleting it would take the pool's pod with it.
                "delete": False,
                # The environment sizes its own command window from the task's
                # verifier budget, and that budget is this multiplier times what
                # the task declares. Passing the same number keeps the window
                # outside Harbor's timeout instead of inside it, where a killed
                # test.sh looks like a task that wrote no reward file.
                "kwargs": {
                    "verifier_timeout_multiplier": self.verifier_timeout_multiplier,
                    **dict(origin),
                },
            },
            "agents": (
                {
                    "name": "agentm_harbor:ExternalAgentMAgent",
                    "model_name": self.model_name,
                    "env": agent_env,
                },
            ),
            "verifier": {"env": verifier_env},
            "tasks": ({"path": str(ref.get("task_dir", ""))},),
        }

    def _review_notes(self, case: FailureCase) -> Path | None:
        """Where this repository's own notes for a reviewer live, if anywhere.

        Beside the task, named for it, because a note about how a database
        takes locks is noise to a syntax highlighter and neither belongs in a
        prompt that has to serve both.
        """
        task_dir = case.backend_ref.get("task_dir")
        if not isinstance(task_dir, str) or not task_dir:
            return None
        path = Path(task_dir) / "review-notes.md"
        return path if path.is_file() else None

    def _origin(self, case: FailureCase, resume_at: int) -> dict[str, JsonValue]:
        """Where the re-run's workspace comes from, or empty if nowhere.

        One answer only: the agent's own commands and edits, out of a database
        of ours, re-applied to a fresh sandbox in order. It costs a re-run of
        everything before the decision, and it works for any attempt whose rows
        we still hold, which is the property that matters.

        There is deliberately no fallback to the sandbox's stored checkpoint:
        that expires with the sandbox within hours, so it is never there for the
        attempts worth studying. Reporting that we cannot resume is more useful
        than a run that scores something else.
        """
        session = str(case.backend_ref.get("agentm_session_id", ""))
        if resume_at < 0 or not session or not self.trajectory_dsn:
            return {}
        return {
            "gateway_url": self.gateway_url,
            "image_registry": self.image_registry,
            "image_tag": self.image_tag,
            "trajectory_dsn": self.trajectory_dsn,
            "trajectory_schema": self.trajectory_schema,
            "trajectory_session": session,
            "trajectory_up_to_turn": resume_at,
        }

    def _read_outcome(self, job_dir: Path, log_path: Path) -> tuple[Metrics, str]:
        results = sorted(job_dir.glob("*/result.json"))
        if not results:
            return Metrics(), _sniff(log_path, "no_result_json")
        try:
            raw = json.loads(results[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("senior-swe: unreadable {}: {}", results[0], exc)
            return Metrics(), "unreadable_result"
        if not isinstance(raw, Mapping):
            return Metrics(), "unreadable_result"

        info = raw.get("exception_info")
        if isinstance(info, Mapping):
            kind = info.get("exception_type")
            if isinstance(kind, str) and kind:
                return Metrics(), _reason(kind) or _sniff(log_path, kind)

        verifier = raw.get("verifier_result")
        rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
        if not isinstance(rewards, Mapping):
            return Metrics(), _sniff(log_path, "no_rewards")
        metrics = _metrics(rewards)
        if not metrics.present:
            return Metrics(), _sniff(log_path, "empty_rewards")
        return metrics, ""


def _reason(text: str) -> str:
    for needle, reason in _LOST_SIGNATURES:
        if needle in text:
            return reason
    return ""


def _sniff(log_path: Path, fallback: str) -> str:
    """Why the sandbox went away, from the run log. Worth the read: a pool
    dropping the pod and a source that had already expired call for different
    responses."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback
    return _reason(text) or fallback


def judge_credentials(profile: str, user_config: str) -> tuple[str, str]:
    """The verifier runs inside the sandbox and cannot read the host config, so
    its key and base url are passed through explicitly."""
    if not profile or not user_config:
        return ("", "")
    path = Path(user_config)
    if not path.is_file():
        logger.warning("senior-swe: no user config at {}", path)
        return ("", "")
    try:
        models = tomllib.loads(path.read_text(encoding="utf-8")).get("models", {})
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("senior-swe: unreadable {}: {}", path, exc)
        return ("", "")
    entry = models.get(profile)
    if not isinstance(entry, dict):
        logger.warning("senior-swe: no [models.{}] in {}", profile, path)
        return ("", "")
    key = entry.get("api_key")
    base = entry.get("base_url") or entry.get("api_base")
    return (
        key if isinstance(key, str) else "",
        base if isinstance(base, str) else "",
    )


def default_agentm_home() -> str:
    return os.environ.get("AGENTM_HOME", "")


__all__ = [
    "DEFAULT_BENCH_ROOT",
    "NAME",
    "PRIMARY_METRIC",
    "HarborReplay",
    "SeniorSweSource",
    "default_agentm_home",
    "judge_credentials",
]


def _last_review_message(dsn: str, schema: str, parent_session: str) -> str:
    """The final thing a review said, out of the trajectory store.

    A review is a child session of the run, and what it produced is the last
    text its model emitted -- there is no separate result record, because the
    tool that started it returns that text to its caller and nothing persists
    it under a name.
    """
    import psycopg  # local: only this path needs the database

    query = (
        f'SELECT t.turn_json FROM "{schema}".agentm_trajectory_turns t '
        f'JOIN "{schema}".agentm_trajectory_sessions s ON s.id = t.session_id '
        "WHERE s.parent_id = %s AND s.purpose = 'review' "
        "ORDER BY t.turn_index DESC LIMIT 12"
    )
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(query, (parent_session,))
        rows = cur.fetchall()

    for (payload,) in rows:
        turn = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(turn, Mapping):
            continue
        response = turn.get("response")
        blocks = response.get("content") if isinstance(response, Mapping) else None
        for block in blocks if isinstance(blocks, list) else []:
            if not isinstance(block, Mapping) or block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and len(text) > 300:
                return text
    return ""
