"""The environment an attempt ran in, restored so an agent can look at it.

Reading a patch tells you what changed. It does not tell you what the agent was
looking at when it decided, and that is usually where the answer is. One
diagnosis run without the repository missed its own cause entirely: the failing
behaviour was endorsed by a test already in the repository, which appears in
neither the agent's diff nor the reference's, so nothing on disk could have
shown it.

There are two ways to get one, and the difference matters enough to be told to
whoever is looking.

**A fork of the attempt's own session** is the faithful one: the repository at
the moment it finished, its changes applied, whatever it installed still
installed. It is also the one that usually cannot be had. A fork lives on the
pool's pod, so it dies when any sibling session for the same image idles out,
and a source deleted more than two hours ago cannot be forked at all with no
persistent checkpoint store configured. In practice this works during a batch
and never after it.

**A fresh session from the task's own image** is always available, because the
image is immutable and still in the registry. It is the repository as the
attempt found it, not as it left it, so anything the attempt did outside its
patch is gone -- which is exactly the evidence an ``environment_self_break``
diagnosis needs. What it does give is the code, the toolchain, the graded tests,
and the ability to run them, with the patches uploaded alongside so any state
can be reconstructed deliberately.

So: fork when the window allows, image otherwise, and say which one it is.

Written against the ARL client rather than harbor's environment adapter, because
this package must not depend on the scenario it studies. The ports the SDK needs
are small -- one ``exec``, plus identity and teardown.
"""

from __future__ import annotations

import shlex
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from loguru import logger

from agentm.core.abi import (
    CancelSignal,
    EnvironmentRef,
    ExecResult,
    PathClass,
    WriterAuthor,
    WriteResult,
)

if TYPE_CHECKING:
    # Type-only: ``arl`` is optional at runtime and is imported inside the two
    # functions that build a client. Naming the real type rather than restating
    # its methods as a Protocol keeps the signatures the gateway actually has;
    # a hand-copied set drifts.
    from arl.async_client import AsyncGatewayClient
    from arl.types import SessionInfo

#: Where the agent's repository lives inside these images. Harbor mounts the
#: task at the filesystem root and the agent works from there.
DEFAULT_WORK_DIR = "/"


class SandboxUnavailable(RuntimeError):
    """The attempt's environment cannot be restored.

    Distinct from a failure inside it: this means the diagnosis has to fall back
    to patches on disk, and the caller should say so rather than quietly
    producing a shallower answer.
    """


@dataclass(slots=True, frozen=True)
class ArlFork:
    """Which checkpoint of which session to restore.

    Named here rather than in the loop's records: a workspace checkpoint is an
    ARL concept, and a benchmark on a different runtime has no use for one.
    """

    session_id: str
    step: int = 0
    #: The task's image. Needed when the source session has been deleted and
    #: the gateway no longer holds its metadata: the checkpoint is still there,
    #: but nothing left records which image to restore it onto.
    image: str = ""


@dataclass(slots=True, frozen=True)
class ArlImage:
    """A fresh session from a task's pre-built image.

    ``uploads`` are files placed inside before anyone looks, and ``setup`` runs
    after them. Between the two, whatever the attempt's own machine held that is
    still worth reading can be put back: the patches, the graded output, the
    trajectory. The image alone is the task as it started, which answers fewer
    questions than it appears to.
    """

    image: str
    uploads: Mapping[str, bytes] = field(default_factory=dict)
    setup: tuple[str, ...] = ()
    idle_timeout_seconds: int = 3600


@dataclass(slots=True)
class _ArlBash:
    """``BashOperations`` over one ARL session."""

    client: AsyncGatewayClient
    session_id: str
    default_timeout: float = 600.0

    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        on_data: object | None = None,
        signal: CancelSignal | None = None,
        log_path: str | None = None,
    ) -> ExecResult:
        from arl.types import StepRequest  # local: optional dependency

        seconds = int(timeout or self.default_timeout)
        step = StepRequest(
            name=f"diag-{uuid.uuid4().hex[:8]}",
            command=["bash", "-lc", cmd],
            env=dict(env) if env else None,
            workDir=cwd or DEFAULT_WORK_DIR,
            timeoutSeconds=max(1, seconds),
        )
        response = await self.client.execute(self.session_id, [step])
        results = response.results or []
        if not results:
            return ExecResult(
                stdout=b"", stderr=b"no result", exit_code=1, timed_out=False
            )
        output = results[0].output
        return ExecResult(
            stdout=_clean(output.stdout),
            stderr=_clean(output.stderr),
            exit_code=int(output.exit_code),
            timed_out=False,
        )


@dataclass(slots=True)
class ArlSandbox:
    """``EnvironmentOperations`` over a forked ARL session."""

    client: AsyncGatewayClient
    session_id: str
    _bash: _ArlBash = field(init=False)
    _writer: ArlResourceWriter = field(init=False)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._bash = _ArlBash(client=self.client, session_id=self.session_id)
        self._writer = ArlResourceWriter(
            client=self.client, session_id=self.session_id, bash=self._bash
        )

    @property
    def ref(self) -> EnvironmentRef:
        return EnvironmentRef(id=self.session_id, kind="sandbox")

    @property
    def bash(self) -> _ArlBash:
        return self._bash

    @property
    def writer(self) -> ArlResourceWriter:
        return self._writer

    async def snapshot(self) -> str | None:
        # Nothing here is meant to outlive the inspection, so there is nothing
        # worth checkpointing.
        return None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.client.delete_session(self.session_id)
            logger.debug("sandbox: deleted {}", self.session_id)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask a result
            logger.warning("sandbox: could not delete {}: {}", self.session_id, exc)
        try:
            await self.client.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("sandbox: could not close client: {}", exc)


@dataclass(slots=True)
class ArlResourceWriter:
    """``ResourceWriter`` over one ARL session.

    Needed because the file tools and the tool-output cap are written against
    this port rather than against bash, and binding an environment without one
    leaves them unsatisfied -- the session then refuses to start at all.

    Binding the *host's* writer instead would be worse than not starting: the
    agent would read and edit this machine's files while believing it was
    looking at the repository under investigation.

    Reads and writes go through the gateway's transfer API rather than through
    ``cat``, so file content never has to survive a shell quoting round-trip.
    The rest is bash, because it is one command each.
    """

    client: AsyncGatewayClient
    session_id: str
    bash: _ArlBash

    #: Where a relative path is taken from. The gateway's transfer API accepts
    #: absolute paths only, and the tools hand over workspace-relative ones --
    #: which the gateway rejects with a 500 that reads like a transfer failure
    #: rather than a path problem.
    work_dir: str = DEFAULT_WORK_DIR

    def _absolute(self, path: str) -> str:
        return (
            path if path.startswith("/") else str(PurePosixPath(self.work_dir) / path)
        )

    async def read(self, path: str) -> bytes:
        data = await self.client.download_file(self.session_id, self._absolute(path))
        return bytes(data)

    async def exists(self, path: str) -> bool:
        result = await self.bash.exec(
            f"test -e {shlex.quote(self._absolute(path))}", cwd="/"
        )
        return result.exit_code == 0

    async def list_dir(self, path: str) -> list[str]:
        result = await self.bash.exec(
            f"ls -1 {shlex.quote(self._absolute(path))}", cwd="/"
        )
        if result.exit_code != 0:
            return []
        text = result.stdout.decode("utf-8", "replace")
        return [line for line in text.splitlines() if line]

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        target = self._absolute(path)
        parent = str(PurePosixPath(target).parent)
        await self.bash.exec(f"mkdir -p {shlex.quote(parent)}", cwd="/")
        try:
            await self.client.upload_file(self.session_id, target, content)
        except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
            return WriteResult(path=path, path_class="unmanaged", error=str(exc))
        return WriteResult(path=path, path_class="unmanaged")

    async def replace(
        self,
        path: str,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        try:
            current = await self.read(path)
        except Exception as exc:  # noqa: BLE001
            return WriteResult(path=path, path_class="unmanaged", error=str(exc))
        # Compare-and-swap on the whole file, as the local and harbor writers do.
        # The one caller passes the content it last read as ``old``, so this is
        # the check that catches the file moving underneath it. Matching a single
        # occurrence instead accepted exactly that case -- a file appended to
        # since the read still contains its old self once, so the write went
        # through and silently folded the concurrent change into the result.
        if current != old:
            return WriteResult(
                path=path,
                path_class="unmanaged",
                error=f"replace precondition failed for {path!r}",
            )
        return await self.write(path, new, rationale=rationale, author=author)

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        result = await self.bash.exec(
            f"rm -rf {shlex.quote(self._absolute(path))}", cwd="/"
        )
        error = result.stderr.decode("utf-8", "replace").strip() or None
        return WriteResult(
            path=path,
            path_class="unmanaged",
            error=None if result.exit_code == 0 else error or "delete failed",
        )

    def classify(self, path: str) -> PathClass:
        # Nothing in a sandbox raised for one investigation is under this
        # project's management, and calling it managed would attach guarantees
        # about history that nothing here provides.
        return "unmanaged"


async def open_sandbox_async(
    fork: ArlFork,
    *,
    gateway_url: str,
    api_key: str = "",
) -> ArlSandbox:
    """A fork of the attempt's environment, ready to run commands in.

    Raises ``SandboxUnavailable`` when the source is gone, which is the common
    case more than two hours after a batch.
    """
    if not fork.session_id:
        raise SandboxUnavailable("case has no ARL session recorded")
    try:
        from arl.async_client import AsyncGatewayClient
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SandboxUnavailable(f"arl client not installed: {exc}") from exc

    client = AsyncGatewayClient(base_url=gateway_url, api_key=api_key or None)
    try:
        response = await client.fork_session(
            fork.session_id, fork.step, image=fork.image or None
        )
    except Exception as exc:
        await _quietly_close(client)
        raise SandboxUnavailable(
            f"cannot fork {fork.session_id} at step {fork.step}: {exc}"
        ) from exc

    session_id = _identity(response.session)
    if not session_id:
        await _quietly_close(client)
        raise SandboxUnavailable("fork returned no session id")

    logger.info(
        "sandbox: forked {} from {} at step {}",
        session_id,
        fork.session_id,
        fork.step,
    )
    return ArlSandbox(client=client, session_id=session_id)


def _clean(text: str | None) -> bytes:
    """Sandbox output, with the bytes that poison downstream storage removed.

    A shell in a repository will eventually cat something binary, and a NUL that
    reaches the trajectory writer fails the insert with "unsupported Unicode
    escape sequence" -- which kills the whole investigation, several minutes in,
    for a byte nobody wanted. Dropping it at the boundary keeps the blast radius
    at the one command that produced it.
    """
    return (text or "").replace("\x00", "").encode("utf-8", "replace")


def _identity(session: SessionInfo | None) -> str:
    """The gateway's id for a session, empty when it returned none.

    A gateway that answers without a session is a failure the caller has to
    turn into ``SandboxUnavailable``, so the empty string is a real answer here
    rather than something to raise on.
    """
    return str(session.id or "") if session is not None else ""


async def _quietly_close(client: AsyncGatewayClient) -> None:
    try:
        await client.aclose()
    except Exception as exc:  # noqa: BLE001
        logger.debug("sandbox: client close failed: {}", exc)


async def open_image_sandbox_async(
    spec: ArlImage,
    *,
    gateway_url: str,
    api_key: str = "",
) -> ArlSandbox:
    """A fresh session from ``spec.image``, with ``spec.uploads`` written in.

    Unlike a fork this does not expire with the batch, so it is what makes a
    diagnosis possible at all once the attempt's own session is gone.
    """
    if not spec.image:
        raise SandboxUnavailable("no image for this case")
    try:
        from arl.async_client import AsyncGatewayClient
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SandboxUnavailable(f"arl client not installed: {exc}") from exc

    client = AsyncGatewayClient(base_url=gateway_url, api_key=api_key or None)
    try:
        info = await client.create_session(
            spec.image, idle_timeout_seconds=spec.idle_timeout_seconds
        )
    except Exception as exc:
        await _quietly_close(client)
        raise SandboxUnavailable(f"cannot start {spec.image}: {exc}") from exc

    session_id = _identity(info)
    if not session_id:
        await _quietly_close(client)
        raise SandboxUnavailable("create returned no session id")

    for path, content in spec.uploads.items():
        try:
            await client.upload_file(session_id, path, content)
        except Exception as exc:  # noqa: BLE001 - a missing file is not fatal,
            # but a silently missing one would make the diagnosis quietly worse.
            logger.warning("sandbox: could not upload {}: {}", path, exc)

    sandbox = ArlSandbox(client=client, session_id=session_id)
    for command in spec.setup:
        result = await sandbox.bash.exec(command, cwd=DEFAULT_WORK_DIR, timeout=300)
        if result.exit_code != 0:
            # Not fatal: a session with half its evidence still answers some
            # questions. But it must be visible, because the shortfall shows up
            # later as a diagnosis that is merely shallow.
            logger.warning(
                "sandbox: setup failed ({}): {}",
                command,
                result.stderr.decode("utf-8", "replace")[:300],
            )

    logger.info("sandbox: started {} from {}", session_id, spec.image)
    return sandbox


__all__ = [
    "DEFAULT_WORK_DIR",
    "ArlFork",
    "ArlImage",
    "ArlResourceWriter",
    "ArlSandbox",
    "SandboxUnavailable",
    "open_image_sandbox_async",
    "open_sandbox_async",
]
