"""AgentM CLI (typer-based).

Single runtime: load the ``general_purpose`` scenario by default (a
curated minimal atom set). A different curated list is opted into via
``--scenario X``. Subsystems are turned off via ``--no-*`` flags.
Failures during construction emit diagnostics through the EventBus rather
than raising; only a missing provider is fatal.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, Any, TextIO, cast

import typer
from dotenv import load_dotenv

from agentm.ai import DEFAULT_PROVIDER_REGISTRY, ProviderRegistry
from agentm.core.abi.events import DiagnosticEvent, EventBus
from agentm.core.abi.session_store import SessionState, SessionStore
from agentm.core.lib.render import final_summary
from agentm.core.abi.events import ExtensionInstallEvent


# Default scenario when the user does not pass ``--scenario`` and does
# not opt out via ``--no-extensions``. Module-level constant so it can
# be patched from tests and referenced by env-var fallthrough below.
DEFAULT_SCENARIO = "general_purpose"

_PACKAGE_WALK_DEPTH = 8


def autoload_dotenv(cwd: Path | None = None) -> None:
    """Load ``.env`` files for the ``agentm`` CLI.

    Honoured order (each call uses ``override=False`` so the first value
    wins across both files and the already-set process env):

    1. Process env (always wins — explicit ``KEY=... agentm`` keeps top
       priority because ``load_dotenv(..., override=False)`` is a no-op
       on already-set names).
    2. ``<cwd>/.env`` — cwd-local file wins on conflict with the
       workspace-root file because it is loaded first.
    3. Workspace-root ``.env`` — the ``.env`` next to the nearest
       ``[tool.uv.workspace]`` pyproject, walked up at most
       ``_PACKAGE_WALK_DEPTH`` levels from ``cwd``.

    Disabled entirely when ``AGENTM_SKIP_DOTENV`` is truthy. The env-var
    check happens **before** any filesystem call (including ``Path.cwd()``
    resolution of the caller-supplied path) so tests can short-circuit
    without leaking cwd-noise into the candidate list.
    """

    if os.environ.get("AGENTM_SKIP_DOTENV"):
        return
    base = (cwd if cwd is not None else Path.cwd()).resolve()
    candidates: list[Path] = [base / ".env"]
    walker = base
    for _ in range(_PACKAGE_WALK_DEPTH):
        manifest = walker / "pyproject.toml"
        if manifest.exists():
            try:
                if "[tool.uv.workspace]" in manifest.read_text(encoding="utf-8"):
                    workspace_env = walker / ".env"
                    if workspace_env != candidates[0]:
                        candidates.append(workspace_env)
                    break
            except OSError:
                pass
        if walker.parent == walker:
            break
        walker = walker.parent
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)


def _print_final(
    final_messages: list[Any],
    *,
    cost_service: Any | None = None,
    provider: str | None = None,
    output: TextIO = sys.stdout,
) -> None:
    report = final_summary(final_messages)
    print("\n" + "=" * 60, file=output)
    print("AGENT FINAL OUTPUT", file=output)
    print("=" * 60, file=output)
    print(report.text if report.text else "<no text output>", file=output)
    print("=" * 60, file=output)
    print(f"messages={report.message_count} tool_calls={report.tool_calls}", file=output)
    usage = report.usage
    if usage.assistant_turns:
        estimate = getattr(cost_service, "estimate", None)
        cost = estimate(usage, provider=provider) if callable(estimate) else None
        suffix = f" cost={cost.currency} {cost.amount:.6f}" if cost is not None else ""
        turns = "turn" if usage.assistant_turns == 1 else "turns"
        print(
            f"tokens: in={usage.input_tokens} out={usage.output_tokens} "
            f"cache_r={usage.cache_read} cache_w={usage.cache_write} "
            f"(over {usage.assistant_turns} {turns}){suffix}",
            file=output,
        )

def _parse_tools(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if value == "":
        return []
    return [t.strip() for t in value.split(",") if t.strip()]


def _parse_extensions(values: list[str] | None) -> list[tuple[str, dict[str, Any]]]:
    """Parse repeated ``--extension MODULE[:JSON]`` flags.

    Each entry is either a bare dotted module path (config = {}) or
    ``module.path:{"key":value}`` where the JSON object is parsed verbatim.
    The first colon splits module from config so module paths themselves
    cannot contain a colon (they are dotted Python imports — colons would
    already be invalid).
    """

    if not values:
        return []
    import json

    out: list[tuple[str, dict[str, Any]]] = []
    for raw in values:
        spec = raw.strip()
        if not spec:
            continue
        if ":" in spec:
            module, _, cfg_raw = spec.partition(":")
            module = module.strip()
            try:
                cfg = json.loads(cfg_raw)
            except json.JSONDecodeError as exc:
                raise typer.BadParameter(
                    f"--extension {raw!r}: config after ':' must be valid JSON ({exc})"
                ) from exc
            if not isinstance(cfg, dict):
                raise typer.BadParameter(
                    f"--extension {raw!r}: config must be a JSON object"
                )
        else:
            module = spec
            cfg = {}
        if not module:
            raise typer.BadParameter(f"--extension {raw!r}: module path is empty")
        out.append((module, cfg))
    return out


def _resolve_provider_model_cwd(
    *,
    provider_flag: str | None,
    model_flag: str | None,
    cwd_flag: str | None,
    registry: ProviderRegistry = DEFAULT_PROVIDER_REGISTRY,
) -> tuple[str, str, str]:
    """Apply ``CLI flag > env var > built-in default`` for provider/model/cwd.

    Resolved at command-invocation time, NOT at module import — so:

    * ``--provider openai`` (without ``AGENTM_MODEL``) picks the OpenAI
      registry's ``default_model`` rather than inheriting whichever
      provider happened to win at import time;
    * ``--cwd`` honours the directory the user names, not the process
      cwd when ``agentm`` was first imported.
    """

    provider = (
        provider_flag
        or os.environ.get("AGENTM_PROVIDER")
        or registry.default_provider().id
    )
    model = (
        model_flag
        or os.environ.get("AGENTM_MODEL")
        or registry.default_model(provider)
    )
    cwd = cwd_flag or os.environ.get("AGENTM_CWD") or os.getcwd()
    return provider, model, cwd


def _auto_commit_default() -> bool:
    """Read `AGENTM_AUTO_COMMIT` env var; truthy → True, 0/false/no → False.

    Default when the var is unset: True (preserves existing sandbox
    workflows where agentm auto-commits during sessions).
    """
    raw = os.environ.get("AGENTM_AUTO_COMMIT")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _make_default_session_store(cwd: str) -> SessionStore:
    from agentm.core.runtime.session_bootstrap import make_default_session_store

    return make_default_session_store(cwd)


def _resolve_session_state(
    *,
    cwd: str,
    resume: str | None,
    continue_recent: bool,
    session_store: SessionStore,
) -> SessionState:
    from agentm.core.runtime.session_bootstrap import resolve_session_state

    try:
        return resolve_session_state(
            cwd=cwd,
            resume=resume,
            continue_recent=continue_recent,
            session_store=session_store,
        )
    except FileNotFoundError as exc:
        # Translate into Typer's parameter-validation error so the CLI
        # surfaces it as "bad --resume" instead of an opaque traceback.
        raise typer.BadParameter(
            f"--resume {resume!r}: no session found for cwd {cwd!r}"
        ) from exc


def _make_install_warner() -> Any:
    def _on_install(event: ExtensionInstallEvent) -> None:
        if event.phase != "error":
            return
        print(
            f"WARNING: [extension_install] {event.module_path}: {event.error}",
            file=sys.stderr,
        )

    return _on_install


def _attach_default_diagnostics(bus: EventBus) -> dict[str, bool]:
    state = {"error_seen": False}

    def _on_diagnostic(event: DiagnosticEvent) -> None:
        prefix = {
            "info": "INFO",
            "warning": "WARNING",
            "error": "ERROR",
        }.get(event.level, event.level.upper())
        print(f"{prefix}: [{event.source}] {event.message}", file=sys.stderr)
        if event.level == "error":
            state["error_seen"] = True

    bus.on(DiagnosticEvent.CHANNEL, _on_diagnostic)
    bus.on(ExtensionInstallEvent.CHANNEL, _make_install_warner())
    return state


def _build_session_config(
    *,
    scenario: str | None,
    extra_extensions: list[tuple[str, dict[str, Any]]],
    no_extensions: bool,
    no_skills: bool,
    no_prompt_templates: bool,
    tool_allowlist: list[str] | None,
    provider: str,
    model: str,
    cwd: str,
    bus: EventBus,
    resume: str | None = None,
    continue_recent: bool = False,
    session_store: SessionStore | None = None,
    provider_registry: ProviderRegistry = DEFAULT_PROVIDER_REGISTRY,
    auto_commit: bool = True,
) -> tuple[Any, SessionState]:
    from agentm.core.abi.session_config import AgentSessionConfig

    store = session_store or _make_default_session_store(cwd)
    session_state = _resolve_session_state(
        cwd=cwd,
        resume=resume,
        continue_recent=continue_recent,
        session_store=store,
    )
    try:
        provider_spec = provider_registry.build(provider, {"model": model})
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    # CLI explicitly opts in to disk-resident project context (CLAUDE.md /
    # AGENTS.md / skills / prompt templates). Embedded SDK callers get an
    # empty resource loader by default — see session_factory.py.
    from agentm.core.runtime.resource_loader import DefaultResourceLoader

    resource_loader = DefaultResourceLoader(
        cwd=Path(cwd),
        no_skills=no_skills,
        no_prompt_templates=no_prompt_templates,
    )

    return (
        AgentSessionConfig(
            cwd=cwd,
            provider=provider_spec,
            scenario=scenario,
            extra_extensions=extra_extensions,
            no_extensions=no_extensions,
            no_skills=no_skills,
            no_prompt_templates=no_prompt_templates,
            tool_allowlist=tool_allowlist,
            session_manager=cast(Any, session_state),
            resource_loader=resource_loader,
            bus=bus,
            auto_commit=auto_commit,
        ),
        session_state,
    )


async def run(
    *,
    prompt: str,
    scenario: str | None,
    extra_extensions: list[tuple[str, dict[str, Any]]],
    no_extensions: bool,
    no_skills: bool,
    no_prompt_templates: bool,
    tool_allowlist: list[str] | None,
    provider: str,
    model: str,
    cwd: str,
    quiet: bool,
    resume: str | None,
    continue_recent: bool,
    auto_commit: bool = True,
    output: TextIO = sys.stdout,
) -> int:
    from agentm.core.runtime.session import AgentSession

    bus = EventBus()
    diagnostic_state = _attach_default_diagnostics(bus)

    config, session_manager = _build_session_config(
        scenario=scenario,
        extra_extensions=extra_extensions,
        no_extensions=no_extensions,
        no_skills=no_skills,
        no_prompt_templates=no_prompt_templates,
        tool_allowlist=tool_allowlist,
        provider=provider,
        model=model,
        cwd=cwd,
        bus=bus,
        resume=resume,
        continue_recent=continue_recent,
        auto_commit=auto_commit,
    )
    if not quiet and session_manager.session_file is not None:
        print(f"INFO: session log: {session_manager.session_file}", file=sys.stderr)
        print(f"INFO: session id: {session_manager.get_session_id()}", file=sys.stderr)

    session = await AgentSession.create(config)
    try:
        final = await session.prompt(prompt)
        cost_service = session.get_service("cost_query")
        provider_name = session.model.provider if session.model is not None else None
        if not quiet:
            _print_final(final, cost_service=cost_service, provider=provider_name, output=output)
    finally:
        await session.shutdown()

    if not quiet:
        sid = session_manager.get_session_id()
        if sid:
            print(
                f'session_id={sid}  (resume with: agentm --resume {sid} "<prompt>")',
                file=output,
            )
    return 1 if diagnostic_state["error_seen"] else 0


def run_cmd(
    prompt: Annotated[
        str,
        typer.Argument(
            help=(
                "User prompt to send to the agent. For multi-turn / TUI "
                "use, run the channels gateway + agentm-terminal --format textual."
            ),
        ),
    ] = "",
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            help=(
                "Opt-in curated extension list. Bare name resolves under "
                "<cwd>/contrib/scenarios/<name>/manifest.yaml. An absolute "
                "path is also accepted. When unset, falls back to the "
                "``general_purpose`` scenario (minimal default tool set)."
            ),
        ),
    ] = None,
    extension: Annotated[
        list[str] | None,
        typer.Option(
            "--extension",
            "-e",
            help=(
                "Mount an extra atom on top of --scenario / auto-discovery. "
                "Repeatable. Form: 'dotted.module.path' or "
                '\'dotted.module.path:{"key":"value"}\' for inline JSON '
                "config. Example: -e llmharness.adapters.agentm "
                "-e some.atom:'{\"k\":3}'. Use `agentm list-extensions` "
                "to browse available atoms."
            ),
        ),
    ] = None,
    no_extensions: Annotated[
        bool,
        typer.Option(
            "--no-extensions",
            help="Skip atom discovery and loading; agent runs with no tools.",
        ),
    ] = False,
    no_skills: Annotated[
        bool,
        typer.Option(
            "--no-skills",
            help="Skip skill discovery in the resource loader.",
        ),
    ] = False,
    no_prompt_templates: Annotated[
        bool,
        typer.Option(
            "--no-prompt-templates",
            help="Skip prompt-template discovery in the resource loader.",
        ),
    ] = False,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            help="Comma-separated allowlist applied to atom-registered tools.",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help=(
                "LLM provider to register. Defaults (via AGENTM_PROVIDER or "
                "the provider registry) include 'anthropic' (respects "
                "ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY) or 'openai' (respects "
                "OPENAI_BASE_URL/OPENAI_API_KEY plus WARPGATE_TICKET and "
                "OPENAI_VERIFY_SSL for self-signed gateways like Warpgate)."
            ),
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=(
                "Model id passed to the active provider. Defaults (via "
                "AGENTM_MODEL or the registry's default_model for the "
                "resolved provider) include 'claude-sonnet-4-6' / 'gpt-4o'; "
                "override for alternates (e.g. 'Kimi-K2', 'deepseek-chat')."
            ),
        ),
    ] = None,
    cwd: Annotated[
        str | None,
        typer.Option(
            "--cwd",
            help=(
                "Working directory exposed to extensions. Defaults to "
                "AGENTM_CWD when set, otherwise the process cwd at "
                "invocation time."
            ),
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress final-output printout."),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            help=(
                "Resume an existing session. Accepts a session id (hex, as "
                "printed at the end of a previous run) or an explicit path "
                "to a .jsonl session log. The new --prompt is appended to "
                "that session's history, enabling multi-turn from the CLI."
            ),
        ),
    ] = None,
    continue_recent: Annotated[
        bool,
        typer.Option(
            "--continue",
            help=(
                "Resume the most recent session for the current --cwd. "
                "Convenient shortcut for --resume <id> when you just ran "
                "agentm in this directory."
            ),
        ),
    ] = False,
    auto_commit: Annotated[
        bool,
        typer.Option(
            "--auto-commit/--no-auto-commit",
            help=(
                "Whether the resource writer should auto-commit managed "
                "writes to the working tree. When disabled the writer falls "
                "back to advisory mode (bytes-on-disk, no `git commit`). "
                "When enabled (default) the writer still refuses to commit "
                "to protected branches ('main', 'master') in the user's real "
                "repo. Reads `AGENTM_AUTO_COMMIT` when the flag is unset."
            ),
        ),
    ] = _auto_commit_default(),
) -> None:
    """Send a single prompt and print the agent's final text."""

    # Resolve provider/model/cwd at invocation time so ``--cwd /b`` actually
    # selects ``/b/.env`` (rather than whichever directory the process was
    # launched from), and so ``--provider openai`` picks OpenAI's default
    # model instead of inheriting another provider's default that happened
    # to be frozen into the typer signature at import.
    provider, model, cwd = _resolve_provider_model_cwd(
        provider_flag=provider,
        model_flag=model,
        cwd_flag=cwd,
    )
    # Run dotenv autoload AFTER --cwd is known: ``cd /a && agentm --cwd /b``
    # must consult ``/b/.env``, not ``/a/.env``. Honours AGENTM_SKIP_DOTENV.
    autoload_dotenv(Path(cwd))

    extra_extensions = _parse_extensions(extension)
    # No --scenario? Resolve via AGENTM_SCENARIO, falling back to the
    # curated minimal set under ``contrib/scenarios/general_purpose/``
    # rather than auto-discovering every builtin atom. Self-modify,
    # evolution/query, and other specialised atoms live in dedicated
    # scenarios that compose on top.
    if scenario is None and not no_extensions:
        scenario = os.environ.get("AGENTM_SCENARIO") or DEFAULT_SCENARIO

    if not prompt:
        print(
            "ERROR: prompt is required.\n"
            "       The in-process Textual TUI (--interactive) was removed —\n"
            "       use the channels gateway instead:\n"
            "         agentm-gateway --bind unix:///tmp/agentm/gw.sock\n"
            "         agentm-worker  --connect unix:///tmp/agentm/gw.sock\n"
            "         agentm-terminal --connect unix:///tmp/agentm/gw.sock --format textual\n"
            "       Cross-host (WebSocket + token):\n"
            "         agentm-gateway --bind ws://0.0.0.0:7777/agentm --bind-token-file /etc/agentm/tokens\n"
            "         agentm-worker  --connect ws://gw.example.com:7777/agentm --token \"$AGENTM_TOKEN\"",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    rc = asyncio.run(
        run(
            prompt=prompt,
            scenario=scenario,
            extra_extensions=extra_extensions,
            no_extensions=no_extensions,
            no_skills=no_skills,
            no_prompt_templates=no_prompt_templates,
            tool_allowlist=_parse_tools(tools),
            provider=provider,
            model=model,
            cwd=cwd,
            quiet=quiet,
            resume=resume,
            continue_recent=continue_recent,
            auto_commit=auto_commit,
        )
    )
    raise typer.Exit(code=rc)


_run = run


def list_extensions_cmd(
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help=(
                "Which discovery source to list: 'builtin' (src/agentm/"
                "extensions/builtin/), 'contrib' (<repo>/contrib/extensions/),"
                " 'user' (<cwd>/.agentm/atoms/), or 'all'."
            ),
        ),
    ] = "all",
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'text' for humans, 'json' for scripts.",
        ),
    ] = "text",
    filter_substr: Annotated[
        str | None,
        typer.Option(
            "--filter",
            help="Substring filter applied to extension name or description.",
        ),
    ] = None,
    cwd: Annotated[
        str,
        typer.Option(
            "--cwd",
            help="Working directory used for 'user' source discovery.",
        ),
    ] = os.getcwd(),
) -> None:
    """List discoverable extensions (atoms) by source.

    Builtins ship under ``src/agentm/extensions/builtin/``; contrib atoms
    live at ``<repo>/contrib/extensions/<name>.py``; user atoms are
    committed by ``api.install_atom`` to ``<cwd>/.agentm/atoms/``. Mount
    any of them via ``agentm -e <module.path>`` (or stack on top of a
    ``--scenario``).
    """

    import json

    from agentm.extensions.discover import (
        BuiltinEntry,
        discover_builtin,
        discover_contrib_atoms,
        discover_user_atoms,
    )

    valid_sources = {"all", "builtin", "contrib", "user"}
    if source not in valid_sources:
        raise typer.BadParameter(
            f"--source {source!r}: must be one of {sorted(valid_sources)}"
        )
    if output_format not in {"text", "json"}:
        raise typer.BadParameter(
            f"--format {output_format!r}: must be 'text' or 'json'"
        )

    buckets: list[tuple[str, dict[str, BuiltinEntry]]] = []
    try:
        if source in {"all", "builtin"}:
            buckets.append(("builtin", discover_builtin()))
        if source in {"all", "contrib"}:
            buckets.append(("contrib", discover_contrib_atoms()))
        if source in {"all", "user"}:
            buckets.append(("user", discover_user_atoms(Path(cwd))))
    except Exception as exc:
        print(f"ERROR: discovery failed: {exc}", file=sys.stderr)
        raise typer.Exit(code=1) from exc

    needle = filter_substr.lower() if filter_substr else None

    def _matches(entry: BuiltinEntry) -> bool:
        if needle is None:
            return True
        return (
            needle in entry.name.lower()
            or needle in entry.manifest.description.lower()
            or needle in entry.module_path.lower()
        )

    if output_format == "json":
        payload = {
            origin: [
                {
                    "name": e.name,
                    "module_path": e.module_path,
                    "description": e.manifest.description,
                    "registers": list(e.manifest.registers),
                    "tier": e.manifest.tier,
                    "api_version": e.manifest.api_version,
                    "affects": list(e.manifest.affects),
                }
                for e in entries.values()
                if _matches(e)
            ]
            for origin, entries in buckets
        }
        print(json.dumps(payload, indent=2, sort_keys=False))
        return

    total = 0
    for origin, entries in buckets:
        filtered = [e for e in entries.values() if _matches(e)]
        total += len(filtered)
        print(f"# {origin} ({len(filtered)})")
        if not filtered:
            print("  (none)")
            continue
        for entry in sorted(filtered, key=lambda e: e.name):
            tier = entry.manifest.tier
            tags = ",".join(entry.manifest.registers) or "-"
            print(
                f"  {entry.name:32s} tier={tier} [{tags}]\n"
                f"    {entry.manifest.description}\n"
                f"    -e {entry.module_path}"
            )
    print(f"\n{total} extension(s) shown.", file=sys.stderr)


_BUILTIN_SUBCOMMANDS: dict[str, Any] = {
    "list-extensions": list_extensions_cmd,
}


def _discover_external_subcommands() -> dict[str, Any]:
    """Scan ``importlib.metadata`` entry points for additional subcommands.

    Each contrib package registers itself via::

        [project.entry-points."agentm.subcommands"]
        gateway = "agentm_channels.cli:main"

    Returns ``name → EntryPoint``; the EP is lazily ``.load()``-ed only
    when the user actually invokes the subcommand, so ``agentm --help``
    and the default prompt path stay fast.
    """

    from importlib.metadata import entry_points

    out: dict[str, Any] = {}
    try:
        eps = entry_points(group="agentm.subcommands")
    except Exception:
        return out
    for ep in eps:
        if ep.name in _BUILTIN_SUBCOMMANDS:
            print(
                f"WARNING: external subcommand {ep.name!r} from "
                f"{ep.value!r} shadowed by builtin",
                file=sys.stderr,
            )
            continue
        out[ep.name] = ep
    return out


def _print_subcommand_help(external: dict[str, Any]) -> None:
    names = sorted([*_BUILTIN_SUBCOMMANDS.keys(), *external.keys()])
    if not names:
        return
    print("\nSubcommands:", file=sys.stderr)
    for name in names:
        if name in _BUILTIN_SUBCOMMANDS:
            origin = "builtin"
        else:
            origin = external[name].value
        print(f"  agentm {name:24s}  [{origin}]", file=sys.stderr)
    print(
        "\nRun `agentm <subcommand> --help` for subcommand options.",
        file=sys.stderr,
    )


def main() -> None:
    """Entry point referenced by the ``agentm`` console script.

    Dispatch order:

    1. ``agentm <subcommand> [args...]`` — builtin introspection
       (``list-extensions``) or any subcommand registered via the
       ``agentm.subcommands`` entry-point group (contrib clients like
       ``gateway`` / ``worker`` / ``terminal`` / ``feishu``).
    2. ``agentm [options] [PROMPT]`` — the legacy single-shot prompt
       runner. ``agentm "<prompt>"`` stays backwards compatible.
    """

    argv = sys.argv[1:]
    external = _discover_external_subcommands()

    if argv and argv[0] in {"--help", "-h"}:
        try:
            typer.run(run_cmd)
        except SystemExit:
            _print_subcommand_help(external)
            raise
        return

    if argv:
        sub = argv[0]
        if sub in _BUILTIN_SUBCOMMANDS:
            sys.argv = [f"{sys.argv[0]} {sub}", *argv[1:]]
            typer.run(_BUILTIN_SUBCOMMANDS[sub])
            return
        if sub in external:
            target = external[sub].load()
            sys.argv = [f"{sys.argv[0]} {sub}", *argv[1:]]
            target()
            return

    typer.run(run_cmd)
