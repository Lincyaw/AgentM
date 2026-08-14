"""What a session holds is a value, and uninstalling an atom restores it.

``ActiveSetFingerprint`` digests the composition plan; ``composition_digest``
digests the result, which is what makes "install an atom, uninstall it, and
find the session unchanged" a single assertion. Everything the composability
work does later is verified that way, so these tests are about the instrument
as much as about the atoms: two of them exist only to show that the digest
notices order and that ``assert_revertible`` can actually fail.

The builtin atoms here were chosen for the kinds of registration they exercise,
not for coverage — bus subscriptions, tools, a plain service, a context policy,
a provider, and the two atoms whose residue is deliberate.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from agentm import ExtensionSpec
from agentm.core.abi.messages import TextContent
from agentm.core.abi.roles import PROVIDER_SESSION_IDENTITY
from agentm.core.abi.termination import ModelEndTurn
from agentm.core.abi.trajectory import Outcome, TurnMeta
from agentm.core.abi.trigger import UserInput
from agentm.testing import assert_revertible, composition_digest, probe_session

_LEAKY_ATOM = """\
import asyncio

from agentm.core.abi.manifest import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="leaky_atom",
    description="Starts a background loop the install ledger cannot see.",
    registers=("service:leaky_probe",),
)


async def _poll():
    while True:
        await asyncio.sleep(3600)


def install(api, config):
    del config
    api.services.register("leaky_probe", object(), scope="session")
    asyncio.create_task(_poll(), name="leaky-probe-loop")
"""


def _file_atom(root: Path, name: str, source: str) -> ExtensionSpec:
    path = root / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return ExtensionSpec.from_file(str(path), digest=digest)


@pytest.mark.asyncio
async def test_the_digest_records_tools_in_advertised_order(tmp_path: Path) -> None:
    """Tools reach the model as a list, so the digest is a list too.

    A set-valued digest would call two different tool surfaces equal, and every
    revertibility check built on it would be blind to a reinstall that landed a
    tool in the wrong place.
    """

    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(
            ExtensionSpec.from_module("agentm.extensions.builtin.task_tracking")
        )
        digest = composition_digest(session)
        assert [entry.name for entry in digest.tools] == [
            tool.name for tool in session.tools
        ]
        assert len(digest.tools) > 1

        session.tools.reverse()
        assert composition_digest(session).tools != digest.tools


@pytest.mark.asyncio
async def test_taking_a_digest_does_not_freeze_the_provider_identity(
    tmp_path: Path,
) -> None:
    """A witness that changes what it measures is not a witness.

    The public provider accessors re-resolve the active provider, and one of
    them mints the ``ProviderSessionIdentity`` that a committed turn makes
    durable. Reading through them here would freeze an identity into any
    session anyone digested.
    """

    async with probe_session(str(tmp_path)) as session:
        session.trajectory.begin(
            UserInput(content=(TextContent(type="text", text="hi"),)),
            run_id="digest",
            run_step=0,
        )
        session.trajectory.commit(
            Outcome(cause=ModelEndTurn()),
            TurnMeta(model_id="probe-model"),
        )

        assert composition_digest(session) == composition_digest(session)
        assert session.services.get_role(PROVIDER_SESSION_IDENTITY) is None


@pytest.mark.asyncio
async def test_bus_subscriptions_revert(tmp_path: Path) -> None:
    """``system_prompt`` is nothing but two subscriptions, at two priorities."""

    async with probe_session(str(tmp_path)) as session:
        await assert_revertible(
            session,
            ExtensionSpec.from_module(
                "agentm.extensions.builtin.system_prompt",
                config={"prompt": "probe", "include_tool_index": True},
            ),
        )


@pytest.mark.asyncio
async def test_tools_revert(tmp_path: Path) -> None:
    """``task_tracking`` registers four tools and nothing else."""

    async with probe_session(str(tmp_path)) as session:
        await assert_revertible(
            session,
            ExtensionSpec.from_module("agentm.extensions.builtin.task_tracking"),
        )


@pytest.mark.asyncio
async def test_a_plain_service_reverts(tmp_path: Path) -> None:
    """``loop_budget`` registers a service without binding it to a role.

    Plain registrations are the ones attribution has to catch at the mutation
    site rather than at a role boundary, so they are worth witnessing directly.
    """

    async with probe_session(str(tmp_path)) as session:
        await assert_revertible(
            session,
            ExtensionSpec.from_module("agentm.extensions.builtin.loop_budget"),
        )


@pytest.mark.asyncio
async def test_a_context_policy_reverts(tmp_path: Path) -> None:
    """``llm_compaction`` registers a context policy plus three services.

    Context policies are the one registration kind ordered by a priority the
    ledger holds separately from the list itself, so restoring the list without
    restoring the priority would leave the digest able to tell.
    """

    async with probe_session(str(tmp_path)) as session:
        await assert_revertible(
            session,
            ExtensionSpec.from_module(
                "agentm.extensions.builtin.llm_compaction",
                config={"keep_last_turns": 4},
            ),
        )


@pytest.mark.asyncio
async def test_plan_mode_reverts_apart_from_its_trigger_codec(tmp_path: Path) -> None:
    """The codec residue is deliberate, and naming it here keeps it honest.

    ``remove_atom_registrations`` leaves an atom's trigger codecs registered on
    purpose: a committed turn names its trigger source, and a session that
    could no longer decode that source would fail to resume. Everything else
    ``plan_mode`` writes — two tools, a renderer, a permission policy, two
    services and a subscription — comes back.

    ``residue`` is an assertion in both directions. If the codec ever does come
    back this test fails as loudly as if a tool had leaked, because an
    equivalence nobody re-checks is how a stale excuse survives.
    """

    async with probe_session(str(tmp_path)) as session:
        await assert_revertible(
            session,
            ExtensionSpec.from_module("agentm.extensions.builtin.plan_mode"),
            residue=["trigger_codecs"],
        )


@pytest.mark.asyncio
async def test_a_provider_atom_leaves_the_model_it_named(tmp_path: Path) -> None:
    """The second deliberate residue, and the one with teeth.

    ``ProviderRegistry.unregister`` drops the registration, the ownership
    record and the active provider *name*, but deliberately leaves the active
    ``stream_fn``/``model`` pair: it is what a running driver is already
    streaming through, and a session that lost its model mid-turn would be
    worse off than one whose model outlives the atom that named it.

    The consequence is only visible once something digests the result rather
    than the plan, which is the point of writing it down here: after this
    uninstall the session still streams through a provider no registration
    names. That is the documented behaviour, not a defect this branch fixes.
    """

    async with probe_session(str(tmp_path)) as session:
        await assert_revertible(
            session,
            ExtensionSpec.from_module(
                "agentm.extensions.builtin.llm_anthropic",
                # Explicit so the test never consults ANTHROPIC_API_KEY; install
                # does not authenticate and no turn is ever run.
                config={"api_key": "probe-key"},
            ),
            residue=["active_model_id", "active_stream_fn"],
        )


@pytest.mark.asyncio
async def test_a_background_task_is_caught_as_a_leak(tmp_path: Path) -> None:
    """The instrument has to be able to fail, and this is the case it must catch.

    A background task is the archetypal write no ledger slot describes: the
    atom is fully detached and the loop is still running. This atom is written
    for the test rather than picked from ``builtin/`` because no builtin starts
    a task at install time — the ones that do start theirs from an event, which
    a probe session never reaches.
    """

    async with probe_session(str(tmp_path)) as session:
        spec = _file_atom(tmp_path, "leaky_atom", _LEAKY_ATOM)
        with pytest.raises(AssertionError, match="background_tasks"):
            await assert_revertible(session, spec)

        leaked = [
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "leaky-probe-loop"
        ]
        assert leaked
        for task in leaked:
            task.cancel()
        await asyncio.gather(*leaked, return_exceptions=True)
