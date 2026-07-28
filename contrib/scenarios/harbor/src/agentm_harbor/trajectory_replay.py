# code-health: ignore-file[AM025] -- the trajectory arrives as untyped JSON from
# the database; every isinstance here is at that boundary, turning one recorded
# tool call into the typed Action below exactly once.
"""Rebuild a recorded attempt's workspace by re-running what the agent did.

An attempt is worth resuming from the middle rather than only from its end: the
choice that decides the outcome is usually made in the first half, and anything
said afterwards arrives after the work that would have to be undone. Resuming
needs the workspace as it stood at that moment, and there are three ways to get
one.

A stored checkpoint is the cheapest and the least available -- it lives with the
sandbox and is gone once the cluster forgets the session, which is hours. The
sandbox's own step log can be re-executed instead, but it is kept by the same
cluster and expires the same way, and it includes the harness's own steps as
well as the agent's.

This is the third: the agent's tool calls, out of the trajectory the run already
wrote to a database of ours. It is available for every recorded attempt, for as
long as we keep the rows, and it contains the agent's actions and nothing else.
The cost is re-running them, which is why it happens once per measurement rather
than per arm.

What is replayed is what changes the workspace -- commands, edits, writes.
Reads are skipped: they moved nothing, and skipping them removes most of the
calls.

**A failure here is fatal on purpose.** If an edit does not apply, the file was
not what the recording says it was, and the state has diverged. Continuing would
produce a plausible-looking workspace that is not the one being studied, and
every number measured against it would be attributed to the injected message.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from agentm_toolbox import FileToolbox, ShellStateStore
from harbor.environments.base import BaseEnvironment
from loguru import logger

#: Tool calls that change the workspace, and the argument each carries its
#: target in. Anything not named here is assumed to be a read.
_SHELL = "bash"
_EDIT = "edit"
_WRITE = "write"

#: The shell a command ran in when the recording does not name one.
_DEFAULT_SHELL = "sh"


class ReplayDiverged(RuntimeError):
    """The recording could not be reproduced.

    Raised rather than logged: a resumed run whose workspace does not match the
    recording measures something nobody asked about, and it looks exactly like a
    run that does match.
    """


@dataclass(slots=True, frozen=True)
class Action:
    """One recorded thing the agent did to the workspace."""

    turn: int
    kind: str
    args: Mapping[str, object]
    #: Whether the tool refused this call when it was recorded.
    #:
    #: A refused edit changed nothing, so replaying it must change nothing
    #: either. This is not a detail: an agent that asks for a replacement the
    #: file has twice gets an error and comes back with a line range instead, so
    #: the recording legitimately contains calls that were never applied.
    #: Applying them makes the workspace diverge from the run being reproduced,
    #: and the failure surfaces later, somewhere else, looking like a different
    #: problem.
    refused: bool = False

    @property
    def path(self) -> str:
        value = self.args.get("path")
        return value if isinstance(value, str) else ""

    @property
    def shell(self) -> str:
        """Which named shell ran this, since each keeps its own directory."""
        value = self.args.get("shell")
        return value if isinstance(value, str) and value else _DEFAULT_SHELL


def read_actions(
    dsn: str,
    schema: str,
    session_id: str,
    *,
    up_to_turn: int,
) -> list[Action]:
    """The agent's workspace-changing calls, in order, through ``up_to_turn``.

    Inclusive of ``up_to_turn`` itself, for two reasons that agree.

    The conversation fork is inclusive -- resuming at turn N restores the agent's
    context up to and including what it did at N -- so replaying only up to N-1
    would leave it reading that it made an edit while the file on disk does not
    have it. Measured: one resumed run spent thirty turns repairing an edit that
    had never been applied, because the recording said it had.

    And it is the right moment on its own terms. A decision is worth
    interrupting once it has been made and its consequences are still cheap to
    undo, not while the agent is still choosing: what we want to ask is "you
    have just done this -- what else does it touch", which is not a question
    that can be asked before the thing exists.
    """

    import psycopg  # local: only this path needs the database

    query = (
        f'SELECT turn_index, turn_json FROM "{schema}".agentm_trajectory_turns '
        "WHERE session_id = %s AND turn_index <= %s ORDER BY turn_index"
    )
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(query, (session_id, up_to_turn))
        rows = cur.fetchall()

    if not rows:
        raise ReplayDiverged(
            f"no recorded turns for session {session_id!r} in schema {schema!r} "
            f"through turn {up_to_turn}. Rebuilding would silently produce a "
            "pristine image, and the trial would score that as the effect of "
            "whatever was injected."
        )

    actions: list[Action] = []
    for turn_index, payload in rows:
        if not isinstance(payload, Mapping):
            continue
        refused = _refusals(payload)
        response = payload.get("response")
        content = response.get("content") if isinstance(response, Mapping) else None
        for item in content if isinstance(content, list) else []:
            if not isinstance(item, Mapping) or item.get("type") != "tool_call":
                continue
            name = item.get("name")
            if name not in {_SHELL, _EDIT, _WRITE}:
                continue
            args = item.get("arguments")
            call_id = item.get("id")
            actions.append(
                Action(
                    turn=int(turn_index),
                    kind=str(name),
                    args=args if isinstance(args, Mapping) else {},
                    refused=refused.get(call_id, False) if isinstance(call_id, str) else False,
                )
            )
    return actions


def _refusals(payload: Mapping[str, object]) -> dict[str, bool]:
    """Which of this turn's calls the tool rejected, by call id.

    Matched on the id rather than on position: a turn can issue several calls
    and nothing guarantees the results come back in the order they were asked
    for.
    """
    out: dict[str, bool] = {}
    results = payload.get("tool_results")
    for entry in results if isinstance(results, list) else []:
        if not isinstance(entry, Mapping):
            continue
        call = entry.get("call")
        result = entry.get("result")
        call_id = call.get("id") if isinstance(call, Mapping) else None
        if isinstance(call_id, str):
            out[call_id] = bool(isinstance(result, Mapping) and result.get("is_error"))
    return out


async def replay(
    env: BaseEnvironment,
    actions: Sequence[Action],
) -> None:
    """Apply ``actions`` to ``env`` in order.

    Commands whose recorded run failed are still replayed. They are part of what
    produced the state -- a failed ``mkdir`` may be why a later command took a
    different branch -- and skipping them would quietly write a different
    history than the one being reproduced.

    Each named shell keeps its own working directory across commands, exactly as
    the bash tool does, using the tool's own tracker so the two cannot drift
    apart. Almost every recorded command opens with its own ``cd``, which is why
    running them all from one fixed directory looked like it worked; the ones
    that do not are the ones that would have moved somewhere first, and they
    would have run in the wrong place with nothing to say so.
    """

    cwd = await _initial_cwd(env)
    shells = ShellStateStore(default_cwd=cwd)
    # Stateless on purpose, and ``require_read=False`` for the same reason:
    # replay skips the agent's reads, so the toolbox can never learn that a file
    # changed under it. In a live session a read after a command that rewrote a
    # file is what refreshes that knowledge; here there is no read, and the
    # toolbox would go on believing the file is whatever the last edit left --
    # then reject the next edit as "modified since you last read it". Measured:
    # one 124-turn rebuild died that way at turn 103, and the workspace was
    # fine. Nothing here needs the state anyway; the bytes are fetched and
    # written by this module, not by the toolbox.
    files = FileToolbox(cwd=cwd, require_read=False)
    applied = 0
    for position, action in enumerate(actions):
        if action.refused and action.kind in {_EDIT, _WRITE}:
            # It changed nothing when it was recorded, so it changes nothing
            # now. Shell commands are replayed even when they failed: a command
            # that exits non-zero has still run, and its side effects are part
            # of the state.
            logger.debug(
                "trajectory-replay: skipping refused {} at turn {}",
                action.kind,
                action.turn,
            )
            continue
        try:
            applied += 1
            if action.kind == _SHELL:
                await _run(env, action, shells)
            elif action.kind == _EDIT:
                await _edit(env, action, files)
            elif action.kind == _WRITE:
                await _write(env, action, files)
        except ReplayDiverged:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure means the same thing
            raise ReplayDiverged(
                f"replaying action {position} (turn {action.turn}, {action.kind}"
                f"{' ' + action.path if action.path else ''}) failed: {exc}"
            ) from exc

    logger.info(
        "trajectory-replay: applied {} of {} recorded action(s)",
        applied,
        len(actions),
    )


async def _initial_cwd(env: BaseEnvironment) -> str:
    """Where a shell starts, asked rather than assumed.

    The image decides this, and the recorded session started from the same
    answer, so taking it from the environment keeps replay and recording in the
    same place without either naming a path.
    """
    result = await env.exec(command="pwd")
    cwd = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not cwd.startswith("/"):
        raise ReplayDiverged(
            f"the environment did not say where a shell starts (pwd gave {cwd!r}); "
            "replaying commands from a guessed directory would put the workspace "
            "somewhere other than where the recording left it"
        )
    return cwd


async def _run(env: BaseEnvironment, action: Action, shells: ShellStateStore) -> None:
    cmd = action.args.get("cmd")
    if not isinstance(cmd, str) or not cmd:
        raise ReplayDiverged(f"command at turn {action.turn} has no text to run")
    timeout = action.args.get("timeout")
    # The tool's timeout is a float. Accepting only ``int`` quietly dropped
    # every one of them, so a command the original run killed at ten seconds
    # ran here to whatever the environment's default is -- and a command that
    # timed out is part of the state being reproduced.
    seconds = (
        int(round(timeout))
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
        else None
    )
    shell = action.shell
    result = await env.exec(
        command=shells.wrap_with_inline_cwd(cmd, shell),
        cwd=shells.effective_cwd(shell),
        timeout_sec=seconds,
    )
    # Reading the reply back is what advances this shell's directory: the
    # wrapper appends a ``pwd`` that the tracker consumes. Dropping the reply
    # would leave every shell pinned where it started.
    shells.strip_inline_cwd(result.stdout, shell)


async def _edit(env: BaseEnvironment, action: Action, files: FileToolbox) -> None:
    """Re-apply one edit by asking the edit tool what it does.

    Not by reimplementing it. The tool decides more than it looks like: which of
    its two forms the arguments select, whether a search string that differs
    from the file in smart quotes or in indentation still matches, where a line
    range clamps, whether a replacement gets a trailing newline, and when a
    replacement is refused for deleting more than it should. A second
    implementation agrees with the first until it does not, and it is wrong in
    the direction that costs most: the workspace ends up subtly different from
    the recording, no error is raised, and whatever is measured afterwards is
    attributed to the thing being tested.
    """
    path = action.path
    if not path:
        raise ReplayDiverged(f"edit at turn {action.turn} names no file")

    current = await _download(env, path)
    old = action.args.get("old_string")
    start = action.args.get("start_line")
    end = action.args.get("end_line")
    new = action.args.get("new_string")
    result, staged = files.plan_edit(
        path,
        current,
        old_string=old if isinstance(old, str) else None,
        new_string=new if isinstance(new, str) else "",
        start_line=start if isinstance(start, int) else None,
        end_line=end if isinstance(end, int) else None,
        replace_all=bool(action.args.get("replace_all")),
    )
    if result.is_error or staged is None:
        raise ReplayDiverged(
            f"{path}: the edit recorded at turn {action.turn} does not apply to "
            f"this workspace -- {result.text}"
        )
    await _upload(env, path, staged)


async def _write(env: BaseEnvironment, action: Action, files: FileToolbox) -> None:
    path = action.path
    content = action.args.get("content")
    if not path or not isinstance(content, str):
        raise ReplayDiverged(f"write at turn {action.turn} has no path or content")
    result, staged = files.plan_write(path, await _download(env, path), content)
    if result.is_error or staged is None:
        raise ReplayDiverged(
            f"{path}: the write recorded at turn {action.turn} does not apply to "
            f"this workspace -- {result.text}"
        )
    await _upload(env, path, staged)


async def _download(env: BaseEnvironment, path: str) -> bytes | None:
    """The file's current bytes, or ``None`` where the tool means "no file".

    ``plan_write`` distinguishes creating from overwriting by this, and
    ``plan_edit`` refuses outright, so a missing file has to arrive as a missing
    file rather than as empty content.
    """
    with TemporaryDirectory() as tmp:
        local = Path(tmp) / "file"
        try:
            await env.download_file(path, local)
        except Exception as exc:  # noqa: BLE001 - any failure means "not readable"
            logger.debug("trajectory-replay: {} is not there to read: {}", path, exc)
            return None
        return local.read_bytes()


async def _upload(env: BaseEnvironment, path: str, content: bytes) -> None:
    parent = str(Path(path).parent)
    await env.exec(command=f"mkdir -p {json.dumps(parent)}")
    with TemporaryDirectory() as tmp:
        local = Path(tmp) / "file"
        local.write_bytes(content)
        await env.upload_file(local, path)


__all__ = [
    "Action",
    "ReplayDiverged",
    "read_actions",
    "replay",
]
