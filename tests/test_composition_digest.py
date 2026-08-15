"""What a session holds is a value, and uninstalling an atom restores it.

``ActiveSetFingerprint`` digests the composition plan; ``composition_digest``
digests the result, which is what makes "install an atom, uninstall it, and
find the session unchanged" a single assertion. Everything the composability
work does later is verified that way, so these tests are about the instrument
as much as about the atoms: two of them exist only to show that the digest
notices order and that ``assert_revertible`` can actually fail.

The named atom tests were chosen for the kinds of registration they exercise —
bus subscriptions, tools, a plain service, a context policy, a provider, and
the atoms whose residue is deliberate. The sweep at the bottom is the coverage
claim: every atom this repository ships reverts, and the three residue entries
that survive across all of them are declared beside the atom that leaves them.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentm import ExtensionSpec
from agentm.core.abi.messages import TextContent
from agentm.core.abi.provider import ProviderConfig
from agentm.core.abi.roles import PROVIDER_SESSION_IDENTITY
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.termination import ModelEndTurn
from agentm.core.abi.trajectory import Outcome, TurnMeta
from agentm.core.abi.trigger import UserInput
from agentm.core.runtime.composition_digest import _service_value
from agentm.extensions import builtin
from agentm.extensions.builtin.llm_anthropic import AnthropicStreamFn
from agentm.extensions.builtin.plan_mode import MODE_TRIGGER_SOURCE, ModeChange
from agentm.testing import (
    assert_revertible,
    composition_digest,
    digest_differences,
    probe_session,
)

_LEAKY_ATOM = """\
import asyncio

from agentm.core.abi.manifest import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="leaky_atom",
    description="Starts a background loop no registration table can see.",
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


_CODEC_ATOM = """\
from dataclasses import dataclass

from agentm.core.abi.manifest import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="codec_atom",
    description="Registers a trigger source a committed turn can name.",
    registers=("service:codec_marker",),
)


@dataclass(frozen=True, slots=True)
class Marker:
    note: str
    source: str = "codec_probe"


class MarkerCodec:
    def serialize(self, trigger):
        return {"__source__": "codec_probe", "note": trigger.note}

    def deserialize(self, data):
        return Marker(note=data["note"])


def install(api, config):
    del config
    api.services.register("codec_marker", Marker(note="kept"), scope="session")
    api.register_trigger_codec("codec_probe", MarkerCodec())
"""


def _file_atom(root: Path, name: str, source: str) -> ExtensionSpec:
    path = root / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return ExtensionSpec.from_file(str(path), digest=digest)


@dataclass(frozen=True, slots=True)
class _Budget:
    """A service shaped the way real services are: a dataclass of values."""

    limit: int
    label: str


class _StatefulBudget(_Budget):
    """A subclass that is a dataclass by inheritance and carries more state."""

    __slots__ = ("spent",)

    def __init__(self, limit: int, label: str, spent: int) -> None:
        super().__init__(limit=limit, label=label)
        object.__setattr__(self, "spent", spent)


class _MutableProbe:
    """A service that satisfies its consumers by holding attributes."""

    __slots__ = ("state",)

    def __init__(self, state: str) -> None:
        self.state = state


def _probe_stream_fn(base_url: str) -> StreamFn:
    """A stream function of the shape a shipped provider actually registers.

    The real one, not a stand-in: both provider atoms in this repository build
    their ``StreamFn`` as a ``@dataclass(slots=True)`` carrying the base URL and
    the key, so this is the shape the digest has to be able to open. A closure
    written here instead would witness a branch nothing in the repository
    reaches, and would say nothing about the providers that ship.
    """

    return AnthropicStreamFn(api_key="probe-key", base_url=base_url)


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

        # Reordered where the tools actually live. ``session.tools`` is a live
        # view over the host's table plus every linked atom context, so there
        # is no session-level list to reverse -- which is the point: a tool
        # belongs to the context that registered it.
        context = session.context_for(
            ExtensionSpec.from_module(
                "agentm.extensions.builtin.task_tracking"
            ).module_path
        )
        assert context is not None
        context.tables.tools.reverse()
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
async def test_the_digest_reads_what_a_service_resolves_to(tmp_path: Path) -> None:
    """A service is data as often as it is behaviour, and data has content.

    Digesting a service by its type name makes every string service equal to
    every other, which would let an atom rewrite a session boundary — the
    allowlist the driver reads each turn is one — and revert to a digest that
    says nothing moved. Content is compared structurally, so equal values
    written twice still agree and unequal ones do not.
    """

    async with probe_session(str(tmp_path)) as session:
        session.services.register("probe", "v1", scope="session")
        first = composition_digest(session)

        session.services.register("probe", "v2", scope="session")
        assert digest_differences(first, composition_digest(session)) != ()

        session.services.register("probe", "v1", scope="session")
        assert digest_differences(first, composition_digest(session)) == ()

        session.services.register("probe", {"keep": 4}, scope="session")
        mapping = composition_digest(session)
        session.services.register("probe", {"keep": 99}, scope="session")
        assert digest_differences(mapping, composition_digest(session)) != ()

        session.services.register("probe", ("a",), scope="session")
        allowlist = composition_digest(session)
        session.services.register("probe", ("b", "c"), scope="session")
        assert digest_differences(allowlist, composition_digest(session)) != ()


@pytest.mark.asyncio
async def test_the_digest_opens_a_dataclass_service_and_says_what_it_cannot(
    tmp_path: Path,
) -> None:
    """The structural walk has to reach where real services live.

    Almost nothing a composition registers is a bare ``str`` or ``dict``: the
    services in a realistic session are dataclasses and Protocol
    implementations, and a walk that opened only the JSON-ish types was
    structural in principle and opaque on every value that exists in practice.
    Dataclasses are opened by their fields.

    The second half is the limit stated as a test rather than as a sentence: an
    object that satisfies a Protocol by holding mutable attributes has no
    fields to enumerate, so mutating it in place is invisible here. That is the
    known coarseness of this equivalence, and a caller that needs it narrower
    has to make the value a dataclass.
    """

    async with probe_session(str(tmp_path)) as session:
        session.services.register("probe", _Budget(limit=4, label="a"), scope="session")
        first = composition_digest(session)

        session.services.register("probe", _Budget(limit=4, label="a"), scope="session")
        assert digest_differences(first, composition_digest(session)) == ()

        session.services.register("probe", _Budget(limit=9, label="a"), scope="session")
        assert [
            difference.field
            for difference in digest_differences(first, composition_digest(session))
        ] == ["services"]

        opaque = _MutableProbe("before")
        session.services.register("probe", opaque, scope="session")
        blind = composition_digest(session)
        opaque.state = "after"
        assert digest_differences(blind, composition_digest(session)) == ()

        # A subclass that inherits its dataclass-ness carries state ``fields()``
        # does not enumerate, so opening it by its base's fields would report a
        # value that changed as unchanged. It is named instead -- the same
        # answer the walk gives every other type it cannot open, and the reason
        # this branch tests the exact type rather than asking ``is_dataclass``.
        session.services.register(
            "probe", _StatefulBudget(limit=4, label="a", spent=0), scope="session"
        )
        inherited = composition_digest(session)
        assert [
            entry.value for entry in inherited.services if entry.key == "probe"
        ] == [_service_value(_StatefulBudget(limit=9, label="z", spent=7))]


@pytest.mark.asyncio
async def test_two_providers_differing_only_in_credential_are_not_equal(
    tmp_path: Path,
) -> None:
    """The registration tables name a provider; the credential is in its fields.

    ``ProviderEntry`` carries the name, the owner and the model id, and the
    active pair is the same type of stream function either way, so a provider
    rebuilt against a different base URL or key used to digest identically in
    every table at once. The base URL is a field of the stream function, so
    that is where the digest reads it.
    """

    model = Model(
        id="probe-model",
        provider="probe",
        context_window=1000,
        max_output_tokens=100,
    )

    async with probe_session(str(tmp_path)) as session:
        session.register_provider(
            "probe",
            ProviderConfig(
                stream_fn=_probe_stream_fn("https://one.invalid"),
                model=model,
                name="probe",
            ),
        )
        first = composition_digest(session)

        session.register_provider(
            "probe",
            ProviderConfig(
                stream_fn=_probe_stream_fn("https://two.invalid"),
                model=model,
                name="probe",
            ),
            replace=True,
        )
        moved = {
            difference.field
            for difference in digest_differences(first, composition_digest(session))
        }
        assert {"services", "active_stream_fn"} <= moved
        # The tables that name a provider are the ones this change is invisible
        # in, which is why it has to be visible in the two above. Asserted as an
        # absence rather than left implicit: if a later field started carrying
        # the credential, the two that carry it now could stop and nothing here
        # would say so.
        assert {"providers", "active_model_id", "provider_identity"} & moved == set()


@pytest.mark.asyncio
async def test_the_digest_reads_the_attribution_of_later_writes(
    tmp_path: Path,
) -> None:
    """Silencing the write observer is itself a write, and the loudest kind.

    ``set_write_observer`` is public on the registry an atom is handed. An atom
    that clears it keeps every service it registers afterwards out of the
    register event stream, so the digest has to see the observer itself —
    otherwise the one write that silences all later writes is the one write
    nothing witnesses.
    """

    async with probe_session(str(tmp_path)) as session:
        before = composition_digest(session)
        session.services.set_write_observer(None)
        assert [
            difference.field
            for difference in digest_differences(before, composition_digest(session))
        ] == ["service_write_observer"]


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

    Context policies are the one registration kind whose position is decided by
    a priority rather than by the write order alone, so a restore that put the
    list back at the wrong priority would leave the digest able to tell.
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
async def test_a_mode_change_turn_round_trips_through_plan_modes_codec(
    tmp_path: Path,
) -> None:
    """The retained-codec rule, exercised through the atom that declares it.

    ``plan_mode`` is the one shipped atom whose residue is a trigger codec, so
    it is the atom the retention rule is argued from — and until now nothing
    put a turn through it. ``_ModeChangeCodec.serialize`` omitted
    ``__source__``, which ``serialize_trigger`` refuses rather than fills in,
    so every ModeChange-triggered turn failed to persist and the residue's own
    justification could not be reached from the atom that claims it.
    """

    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(
            ExtensionSpec.from_module("agentm.extensions.builtin.plan_mode")
        )
        session.trajectory.begin(
            ModeChange(mode="plan", reason="host toggled"),
            run_id="mode",
            run_step=0,
        )
        turn = session.trajectory.commit(
            Outcome(cause=ModelEndTurn()),
            TurnMeta(model_id="probe-model"),
        )
        restored = session.codec.deserialize_turn(session.codec.serialize_turn(turn))
        assert isinstance(restored.trigger, ModeChange)
        assert restored.trigger.source == MODE_TRIGGER_SOURCE
        assert restored.trigger.mode == "plan"
        assert restored.trigger.reason == "host toggled"


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

    A background task is the archetypal write no registration table describes:
    the atom is fully detached and the loop is still running. This atom is written
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


@dataclass(frozen=True, slots=True)
class _Builtin:
    """One shipped atom and what it takes to install and revert it.

    ``prerequisites`` are composed into the probe session before the atom under
    test, so they are part of the before picture rather than part of what has
    to come back. ``residue`` names the digest fields this atom is known not to
    restore; each entry carries its reason beside it in ``_BUILTIN_ATOMS``.
    """

    name: str
    config: dict[str, object] | None = None
    prerequisites: tuple[str, ...] = ()
    residue: tuple[str, ...] = ()


_LOCAL_BACKEND = "local_backend"
"""The one atom the others need: it binds the resource writer and bash."""

_PROVIDER_RESIDUE = ("active_model_id", "active_stream_fn")
"""What ``ProviderRegistry.unregister`` leaves on purpose.

The registration, the ownership record and the active provider *name* all go;
the active ``stream_fn``/``model`` pair does not, because it is what a running
driver is already streaming through. The consequence is only visible to
something that digests the result rather than the plan: after this uninstall
the session still streams through a provider no registration names.
"""

_BUILTIN_ATOMS: tuple[_Builtin, ...] = (
    _Builtin("atom_watch"),
    _Builtin("background_exec"),
    _Builtin("file_tools", prerequisites=(_LOCAL_BACKEND,)),
    _Builtin("goal", config={"condition": "the probe is done"}),
    _Builtin(
        "llm_anthropic",
        # Explicit so the test never consults ANTHROPIC_API_KEY; install does
        # not authenticate and no turn is ever run.
        config={"api_key": "probe-key"},
        residue=_PROVIDER_RESIDUE,
    ),
    _Builtin("llm_compaction", config={"keep_last_turns": 4}),
    _Builtin("llm_openai", config={"api_key": "probe-key"}, residue=_PROVIDER_RESIDUE),
    _Builtin(_LOCAL_BACKEND),
    _Builtin("loop_budget"),
    _Builtin("memory", prerequisites=(_LOCAL_BACKEND,)),
    _Builtin("message_patterns"),
    _Builtin("observability"),
    _Builtin(
        "plan_mode",
        # A committed turn names its trigger source, and a session that could
        # no longer decode it would fail to resume, so an uninstalled atom's
        # trigger codec stays registered. See the codec test below, which
        # asserts the thing this residue exists for.
        residue=("trigger_codecs",),
    ),
    _Builtin(
        "read_history",
        config={"tool_result_max_tokens": 1000, "total_max_tokens": 4000},
    ),
    _Builtin("retry_policy"),
    _Builtin("skill_loader"),
    _Builtin(
        "structured_output",
        config={"schema": {"type": "object", "properties": {}}},
    ),
    _Builtin("sub_agent"),
    _Builtin("system_prompt", config={"prompt": "probe", "include_tool_index": True}),
    _Builtin("task_tracking"),
    _Builtin("thinking_retry"),
    _Builtin("tool_authoring"),
    _Builtin("tool_bash", prerequisites=(_LOCAL_BACKEND,)),
    _Builtin("tool_error_messages"),
    _Builtin("tool_policy"),
    _Builtin("tool_purpose"),
    _Builtin("tool_result_cap", prerequisites=(_LOCAL_BACKEND,)),
    _Builtin("trace_query"),
    _Builtin("workflow"),
)


def _builtin_spec(name: str, config: dict[str, object] | None = None) -> ExtensionSpec:
    return ExtensionSpec.from_module(
        f"agentm.extensions.builtin.{name}",
        config=config or None,
    )


def test_every_shipped_atom_is_listed_for_the_revertibility_sweep() -> None:
    """A new builtin has to say what it takes to revert, or fail here.

    The sweep below is only a claim about every shipped atom for as long as the
    list is every shipped atom. Reading the package directory is what keeps
    that true without anyone remembering.
    """

    shipped = {
        path.stem
        for path in Path(builtin.__file__ or "").parent.glob("*.py")
        if path.stem != "__init__"
    }
    assert shipped == {atom.name for atom in _BUILTIN_ATOMS}


@pytest.mark.parametrize("atom", _BUILTIN_ATOMS, ids=lambda atom: atom.name)
@pytest.mark.asyncio
async def test_every_shipped_atom_reverts(tmp_path: Path, atom: _Builtin) -> None:
    """Install each shipped atom into a probe session, detach it, and diff.

    The obligation the whole context model rests on, discharged against the
    atoms that actually ship rather than against a chosen few. Three residue
    entries survive across twenty-nine atoms, each justified where it is
    declared; everything else comes back.
    """

    async with probe_session(
        str(tmp_path),
        extensions=[_builtin_spec(name) for name in atom.prerequisites],
    ) as session:
        await assert_revertible(
            session,
            _builtin_spec(atom.name, atom.config),
            residue=list(atom.residue),
        )


@pytest.mark.asyncio
async def test_a_superseded_atoms_codec_still_decodes_a_committed_turn(
    tmp_path: Path,
) -> None:
    """What the retained codec is for, asserted on a turn rather than a table.

    A committed turn names its trigger source by name, so an atom's codec is
    registered on the session's shared registry and deliberately left there
    when the atom goes. That is why ownership of a trigger source is the one
    attribution the context tree cannot hold: the registration outlives the
    context, and a record that has to outlive the context cannot be the
    context.

    Both halves are here. A replacement takes the source over rather than
    colliding with it, which only works because the source is still recorded
    against an atom that has left the installed set; and an outright detach
    leaves the committed turn decodable, still attributed to the atom that
    registered it.
    """

    spec = _file_atom(tmp_path, "codec_atom", _CODEC_ATOM)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        marker = session.services.require("codec_marker", object)
        session.trajectory.begin(marker, run_id="codec", run_step=0)
        turn = session.trajectory.commit(
            Outcome(cause=ModelEndTurn()),
            TurnMeta(model_id="probe-model"),
        )
        recorded = session.codec.serialize_turn(turn)

        # The supersede. The source is registered and its owner has left the
        # installed set, which is what lets the replacement take it over
        # instead of colliding with a codec nobody can remove.
        await session.install_extension(spec, replace=True)
        superseded = session.codec.deserialize_turn(recorded)
        assert type(superseded.trigger).__name__ == "Marker"
        assert superseded.trigger.note == "kept"

        # And the detach, where nothing re-registers it at all.
        assert session.uninstall_extension(spec)
        detached = session.codec.deserialize_turn(recorded)
        assert type(detached.trigger).__name__ == "Marker"
        assert detached.trigger.note == "kept"
        assert [
            entry.owner
            for entry in composition_digest(session).trigger_codecs
            if entry.source == "codec_probe"
        ] == [spec.module_path]
