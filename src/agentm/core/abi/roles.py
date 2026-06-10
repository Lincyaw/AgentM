"""Singleton-role identifiers + cross-boundary config keys.

These constants name the runtime "slots" an atom can fulfil. Each role
is filled by exactly one atom per session; the discovery layer indexes
atoms by ``MANIFEST.provides_role`` so the runtime asks for a role
("who is today's command parser?") rather than a specific atom name.

Constants live in ``agentm.core.abi`` — *not* in any individual atom —
because they cross the SDK/scenario boundary. A scenario that ships its
own command parser needs the same role string the runtime checks for.
"""

from __future__ import annotations

from typing import Final

# --- Singleton roles -------------------------------------------------------

COMMAND_PARSER: Final = "command_parser"
"""Atom that parses slash commands and dispatches registered handlers.

Default fulfiller: ``agentm.extensions.builtin.slash_commands``.
Resolved by the session factory to back the ``slash_commands`` service
and the dispatcher's ``fallback_owner``."""

COMPACTION_PROMPTS: Final = "compaction_prompts"
"""Atom that registers compaction prompt bodies + entry materializers.

Floor atom: present in every session even when not listed in a scenario
manifest so the ``llm_compaction`` atom always finds the default English
prompts.
Default fulfiller: ``agentm.extensions.builtin.compaction_prompts``."""

PROMPT_REGISTRY: Final = "prompt_registry"
"""Atom that publishes the in-memory prompt registry (under service key
``"prompt_templates"``) and the on-disk slash-template loader.

Floor atom: present in every session because ``compaction_prompts`` and
``llm_compaction`` resolve their bodies through it. Default fulfiller:
``agentm.extensions.builtin.prompt_templates``."""

SYSTEM_PROMPT_PROVIDER: Final = "system_prompt_provider"
"""Atom that prepends a system prompt at ``before_agent_start``.

Required whenever ``SUB_AGENT_RUNTIME`` is loaded — sub-agents inject
inherited prompt text and need this hook in the chain. Default
fulfiller: ``agentm.extensions.builtin.system_prompt``."""

SUB_AGENT_RUNTIME: Final = "sub_agent_runtime"
"""Atom that exposes ``dispatch_agent`` / ``check_tasks`` and owns the
nested-session lifecycle. Default fulfiller:
``agentm.extensions.builtin.sub_agent``."""

PROVIDER_INHERITOR: Final = "provider_inheritor"
"""Atom that re-publishes a parent session's :class:`ProviderConfig` to
a child session. The session factory installs whichever atom claims this
role when a child config arrives with ``provider=None``. Default
fulfiller: ``agentm.extensions.builtin.inherit_provider``."""


# --- Cross-boundary config keys -------------------------------------------

PARENT_PROVIDER_CONFIG_KEY: Final = "provider"
"""Key under which :func:`default_child_provider_factory` hands the
parent :class:`ProviderConfig` to the ``PROVIDER_INHERITOR`` atom's
install config. Lives here (rather than on the atom itself) so the
session factory can build the install spec without importing a specific
builtin module."""


# --- Service registry keys -------------------------------------------------

SLASH_COMMAND_DISPATCHER_SERVICE: Final = "slash_commands"
"""``service_registry`` key under which the runtime publishes the
:class:`HarnessCommandDispatcher`. Atoms that need to invoke registered
slash commands programmatically look it up via
``api.get_service(SLASH_COMMAND_DISPATCHER_SERVICE)``."""

SESSION_STORE_SERVICE: Final = "session_store"
"""``service_registry`` key under which the session factory publishes
the :class:`~agentm.core.abi.session_store.SessionStore`. Atoms that
need to resume existing sessions (e.g. workflow ``agent(session_id=)``)
look it up via ``api.get_service(SESSION_STORE_SERVICE)``."""

LOOP_BUDGET_SERVICE: Final = "loop_budget"
"""``service_registry`` key under which the ``loop_budget`` atom publishes a
:class:`~agentm.core.abi.loop.LoopConfig`. The session factory reads it just
before constructing the :class:`AgentLoop` to set the scenario's turn / tool
budget. Absent ⇒ the substrate falls back to ``LoopConfig()`` (no cap). An
explicit caller override (CLI ``--max-turns`` / SDK ``loop_config=``) takes
precedence over whatever the atom registered."""

MODEL_RESOLVER_SERVICE: Final = "model_resolver"
"""``service_registry`` key under which the session factory publishes a
callable ``(model_name: str) -> tuple[str, dict[str, Any]] | None`` that
resolves a ``config.toml`` profile name to a provider tuple suitable for
``AgentSessionConfig.provider``.  Atoms that need model resolution (e.g.
``workflow`` for its ``model=`` parameter) use this service instead of
importing presenter-layer modules directly."""


__all__ = [
    "COMMAND_PARSER",
    "COMPACTION_PROMPTS",
    "LOOP_BUDGET_SERVICE",
    "MODEL_RESOLVER_SERVICE",
    "PARENT_PROVIDER_CONFIG_KEY",
    "PROMPT_REGISTRY",
    "PROVIDER_INHERITOR",
    "SESSION_STORE_SERVICE",
    "SLASH_COMMAND_DISPATCHER_SERVICE",
    "SUB_AGENT_RUNTIME",
    "SYSTEM_PROMPT_PROVIDER",
]
