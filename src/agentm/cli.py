"""AgentM CLI (typer-based).

Single runtime: load the ``local`` scenario by default (a curated minimal
atom set). A different curated list is opted into via ``--scenario X``.
Subsystems are turned off via ``--no-*`` flags. Failures during
construction emit diagnostics through the EventBus rather than raising;
only a missing provider is fatal.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any, TextIO, cast

import typer
from dotenv import load_dotenv

from agentm.ai import DEFAULT_PROVIDER_REGISTRY, ProviderRegistry
from agentm.core.abi import LoopConfig
from agentm.core.abi.events import DiagnosticEvent, EventBus, MessagePersistedEvent
from agentm.core.abi.session_store import SessionState, SessionStore
from agentm.core.lib.render import final_summary
from agentm.core.lib.user_config import (
    ModelProfile,
    apply_reasoning_effort,
    resolve_model_profile,
)
from agentm.core.abi.events import ExtensionInstallEvent

from dataclasses import dataclass, field


# Default scenario when the user does not pass ``--scenario`` and does
# not opt out via ``--no-extensions``. Module-level constant so it can
# be patched from tests and referenced by env-var fallthrough below.
DEFAULT_SCENARIO = "local"

# #201: defense-in-depth bound on the one-shot ``idle()`` wait. The unbounded
# wait makes one-shot exit hinge entirely on ``track_background`` accounting
# being perfect; a leaked counter or a stuck background tool would hang
# ``agentm -p`` silently forever. This generous ceiling (well above any normal
# late-completion window — late completions land in seconds, not minutes)
# converts that into a visible warning + clean exit. Long-lived hosts (gateway
# / worker / TUI) own their loop and never call ``idle()``.
ONESHOT_IDLE_TIMEOUT_SECONDS = 120.0

_PACKAGE_WALK_DEPTH = 8


# Single command tree for the ``agentm`` console script. The default prompt
# runner is the root callback (``run_cmd``); ``list-extensions`` is a sibling
# command; ``trace`` and ``gateway`` are mounted from their own Typer apps.
# ``trace`` / ``gateway`` live in the same ``agentm`` package and dependency
# closure as this module, so they compose as ordinary subcommands — there is
# no vendor-SDK isolation reason to dispatch them out of band. (The peer CLIs
# ``agentm-terminal`` / ``agentm-feishu`` are the genuinely isolated binaries;
# they ship their own console scripts and are not ``agentm`` subcommands.)
app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@dataclass
class CliRunConfig:
    """Resolved CLI flags for session construction.

    Bundles every knob that flows from ``run_cmd`` through ``run`` into
    ``_build_session_config``, replacing the ~16-parameter signatures.
    """

    provider: str
    model: str
    cwd: str
    prompt: str = ""
    scenario: str | None = None
    extra_extensions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    no_extensions: bool = False
    no_skills: bool = False
    no_prompt_templates: bool = False
    tool_allowlist: list[str] | None = None
    quiet: bool = False
    resume: str | None = None
    continue_recent: bool = False
    max_turns: int | None = None
    max_tool_calls: int | None = None
    atom_config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    profile: ModelProfile | None = None
    reasoning_effort: str | None = None


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
    # Body text was already streamed live via ``_attach_streaming_presenter``;
    # only emit the summary line so we don't double-print the trajectory.
    report = final_summary(final_messages)
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


def _parse_traceparent(value: str | None) -> tuple[str, str] | None:
    """Parse a W3C ``traceparent`` header into ``(trace_id, span_id)``.

    Format: ``<version>-<trace-id:32hex>-<parent-id:16hex>-<flags:2hex>``.
    Returns ``None`` for absent/malformed/all-zero values so the caller falls
    back to a fresh trace. Used to continue a parent trace handed to AgentM via
    the ``TRACEPARENT`` env var (e.g. by a workbuddy dispatch) so AgentM's spans
    land in the SAME OTel trace as the caller — root_session_id maps to the OTel
    trace_id and parent_session_id to the caller's span_id.
    """
    if not value:
        return None
    parts = value.strip().split("-")
    if len(parts) != 4:
        return None
    _version, trace_id, span_id, _flags = parts
    trace_id = trace_id.lower()
    span_id = span_id.lower()
    if len(trace_id) != 32 or len(span_id) != 16:
        return None
    try:
        int(trace_id, 16)
        int(span_id, 16)
    except ValueError:
        return None
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return trace_id, span_id


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


def _parse_set_overrides(values: list[str] | None) -> dict[str, dict[str, Any]]:
    """Parse repeated ``--set <atom>.<key>=<value>`` flags.

    Returns ``{atom_name: {key: raw_value}}``. Values stay as raw strings —
    type coercion is deferred to ``resolve_atom_configs``, which reads the
    atom's ``config_schema`` (the single source of truth for key types). The
    value split is on the first ``=`` so values may themselves contain ``=``.
    Atom names are ``[a-z_]+`` (no dots, per the MANIFEST contract), so the
    atom/key split is on the first ``.`` and the key is everything after it;
    surrounding whitespace on atom and key is trimmed so ``a.b = c`` keys on
    ``b`` rather than ``b ``.
    """

    if not values:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for raw in values:
        spec = raw.strip()
        if not spec:
            continue
        lhs, sep, value = spec.partition("=")
        if not sep:
            raise typer.BadParameter(
                f"--set {raw!r}: expected '<atom>.<key>=<value>'"
            )
        atom, dot, key = lhs.partition(".")
        atom, key = atom.strip(), key.strip()
        if not dot or not atom or not key:
            raise typer.BadParameter(
                f"--set {raw!r}: left of '=' must be '<atom>.<key>'"
            )
        out.setdefault(atom, {})[key] = value
    return out


def _resolve_provider_model_cwd(
    *,
    provider_flag: str | None,
    model_flag: str | None,
    cwd_flag: str | None,
    registry: ProviderRegistry = DEFAULT_PROVIDER_REGISTRY,
) -> tuple[str, str, str, ModelProfile | None]:
    """Apply ``CLI flag > env var > config.toml profile > built-in default``.

    Resolved at command-invocation time, NOT at module import — so:

    * ``--provider openai`` (without ``AGENTM_MODEL``) picks the OpenAI
      registry's ``default_model`` rather than inheriting whichever
      provider happened to win at import time;
    * ``--cwd`` honours the directory the user names, not the process
      cwd when ``agentm`` was first imported.

    Returns ``(provider, model, cwd, profile)`` where *profile* is the
    matched :class:`ModelProfile` if the resolved model name corresponds
    to a ``~/.agentm/config.toml`` profile, or ``None`` otherwise.
    """

    raw_model = model_flag or os.environ.get("AGENTM_MODEL")

    # Check whether the model name matches a config.toml profile.
    # When raw_model is None, resolve_model_profile falls back to
    # config.default_model (if any).
    profile = resolve_model_profile(raw_model)

    if profile is not None:
        # Explicit --provider overrides the profile's provider.
        provider = provider_flag or os.environ.get("AGENTM_PROVIDER") or profile.provider
        model = profile.model
    else:
        provider = (
            provider_flag
            or os.environ.get("AGENTM_PROVIDER")
            or registry.default_provider().id
        )
        model = raw_model or registry.default_model(provider)

    cwd = cwd_flag or os.environ.get("AGENTM_CWD") or os.getcwd()
    return provider, model, cwd, profile


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


def _attach_streaming_presenter(bus: EventBus, output: TextIO) -> None:
    """Stream agent activity to ``output`` as the loop produces it.

    Subscribes to :class:`MessagePersistedEvent` (one event per durable
    message append in the loop) and renders:

    * ``assistant`` text blocks verbatim, tool calls as ``→ name(args)``
    * ``tool_result`` blocks as ``⇐ name: text`` (truncated to 400 chars)
    * ``injected`` messages as ``[injected role] text``

    Thinking blocks are dropped — they are noisy on terminal and the
    provider often strips them anyway. Use ``agentm trace messages``
    against the JSONL for the full trajectory after the fact.
    """

    def _truncate(text: str, limit: int = 400) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit] + f"... <+{len(text) - limit} chars>"

    def _short_args(args: Any) -> str:
        if isinstance(args, dict):
            parts = []
            for k, v in args.items():
                vs = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
                parts.append(f"{k}={_truncate(str(vs), 80)}")
            return ", ".join(parts)
        return _truncate(str(args), 160)

    use_color = hasattr(output, "isatty") and output.isatty() and not os.environ.get("NO_COLOR")
    DIM = "\x1b[2m" if use_color else ""
    RESET = "\x1b[0m" if use_color else ""

    def _on_message(event: MessagePersistedEvent) -> None:
        content = getattr(event.message, "content", None) or []
        if event.source == "assistant":
            for block in content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text = getattr(block, "text", "") or ""
                    if text.strip():
                        print(text, file=output, flush=True)
                        print(file=output, flush=True)
                elif btype == "thinking":
                    # Render thinking dimmed (live only — not re-encoded into
                    # next turn unless thinking_round_trip='system_note'; see
                    # llm_openai.py INFO message). One `💭 ` per line so long
                    # reasoning stays scannable.
                    text = getattr(block, "text", "") or ""
                    if text.strip():
                        for line in text.splitlines():
                            print(f"{DIM}💭 {line}{RESET}", file=output, flush=True)
                        print(file=output, flush=True)
                elif btype == "tool_call":
                    name = getattr(block, "name", "?")
                    args = getattr(block, "input", None) or getattr(block, "arguments", None)
                    print(f"→ {name}({_short_args(args)})", file=output, flush=True)
        elif event.source == "tool_result":
            for block in content:
                if getattr(block, "type", None) != "tool_result":
                    continue
                sub = getattr(block, "content", None) or []
                text = "".join(getattr(c, "text", "") or "" for c in sub if getattr(c, "type", None) == "text")
                is_error = getattr(block, "is_error", False)
                prefix = "⇐ ERROR" if is_error else "⇐"
                print(f"  {prefix} {_truncate(text)}", file=output, flush=True)
            print(file=output, flush=True)
        elif event.source == "injected":
            role = getattr(event.message, "role", "?")
            for block in content:
                if getattr(block, "type", None) == "text":
                    print(f"[injected:{role}] {_truncate(getattr(block, 'text', ''))}", file=output, flush=True)

    bus.on(MessagePersistedEvent.CHANNEL, _on_message)


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
    config: CliRunConfig,
    *,
    bus: EventBus,
    session_store: SessionStore | None = None,
    provider_registry: ProviderRegistry = DEFAULT_PROVIDER_REGISTRY,
    loop_config: LoopConfig | None = None,
) -> tuple[Any, SessionState]:
    from agentm.core.abi.session_config import AgentSessionConfig

    store = session_store or _make_default_session_store(config.cwd)
    session_state = _resolve_session_state(
        cwd=config.cwd,
        resume=config.resume,
        continue_recent=config.continue_recent,
        session_store=store,
    )
    try:
        if config.profile is not None:
            build_config = config.profile.to_build_config()
        else:
            build_config = {"model": config.model}
        apply_reasoning_effort(build_config, config.reasoning_effort)
        provider_spec = provider_registry.build(config.provider, build_config)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    from agentm.core.runtime.resource_loader import DefaultResourceLoader

    resource_loader = DefaultResourceLoader(
        cwd=Path(config.cwd),
        no_skills=config.no_skills,
        no_prompt_templates=config.no_prompt_templates,
    )

    # Continue a caller-supplied OTel trace (e.g. a workbuddy dispatch) when a
    # W3C TRACEPARENT is present in the env: root_session_id == OTel trace_id so
    # AgentM's spans land in the caller's trace; parent_session_id == the
    # caller's span_id. Absent/malformed → fresh trace (root/parent left None).
    root_session_id: str | None = None
    parent_session_id: str | None = None
    traceparent = _parse_traceparent(os.environ.get("TRACEPARENT"))
    if traceparent is not None:
        root_session_id, parent_session_id = traceparent

    return (
        AgentSessionConfig(
            cwd=config.cwd,
            provider=provider_spec,
            scenario=config.scenario,
            extra_extensions=config.extra_extensions,
            atom_config_overrides=config.atom_config_overrides,
            no_extensions=config.no_extensions,
            no_skills=config.no_skills,
            no_prompt_templates=config.no_prompt_templates,
            tool_allowlist=config.tool_allowlist,
            session_manager=cast(Any, session_state),
            resource_loader=resource_loader,
            bus=bus,
            loop_config=loop_config,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
        ),
        session_state,
    )


async def run(
    config: CliRunConfig,
    *,
    output: TextIO = sys.stdout,
) -> int:
    from agentm.core.runtime.session import AgentSession

    bus = EventBus()
    diagnostic_state = _attach_default_diagnostics(bus)
    if not config.quiet:
        _attach_streaming_presenter(bus, output)

    loop_config = (
        LoopConfig(max_turns=config.max_turns, max_tool_calls=config.max_tool_calls)
        if (config.max_turns is not None or config.max_tool_calls is not None)
        else None
    )

    session_config, session_manager = _build_session_config(
        config, bus=bus, loop_config=loop_config,
    )
    if not config.quiet and session_manager.session_file is not None:
        print(f"INFO: session log: {session_manager.session_file}", file=sys.stderr)
        print(f"INFO: session id: {session_manager.get_session_id()}", file=sys.stderr)

    session = await AgentSession.create(session_config)
    try:
        # The prompt/tick return value is intentionally discarded: the final
        # message list is re-fetched AFTER ``idle()`` below so it reflects any
        # late background completion delivered between agent_end and idle.
        if config.prompt:
            await session.prompt(config.prompt)
        else:
            await session.tick()
        # #179: a one-shot CLI owns its event loop only for the duration of
        # this coroutine — once it returns, ``asyncio.run`` tears the loop down.
        # An auto-backgrounded tool / child subagent that finishes AFTER the
        # agent's last turn would post its completion into a dead loop and be
        # lost. Wait for the session to be truly idle (driver parked, inbox
        # empty, no tracked background unit running) so the persistent driver
        # delivers any late completion first. Long-lived hosts (gateway / worker
        # / feishu / TUI) keep their own loop alive and never call this.
        # #201: bound the wait so a leaked ``track_background`` counter or a
        # stuck background unit cannot hang the one-shot CLI forever. On trip we
        # warn and exit anyway (still exit 0 for an otherwise-completed prompt);
        # any work that finishes after this point will not be delivered.
        if not await session.idle(timeout=ONESHOT_IDLE_TIMEOUT_SECONDS):
            print(
                "WARNING: timed out waiting for background work to finish after "
                f"{ONESHOT_IDLE_TIMEOUT_SECONDS:.0f}s; exiting anyway. Any "
                "late background completion will not be delivered.",
                file=sys.stderr,
            )
        final = session.session_manager.get_messages()
        cost_service = session.get_service("cost_query")
        provider_name = session.model.provider if session.model is not None else None
        if not config.quiet:
            _print_final(final, cost_service=cost_service, provider=provider_name, output=output)
    finally:
        await session.shutdown()

    if not config.quiet:
        sid = session_manager.get_session_id()
        if sid:
            print(
                f'session_id={sid}  (resume with: agentm --resume {sid} -p "<prompt>")',
                file=output,
            )
    return 1 if diagnostic_state["error_seen"] else 0


@app.callback(invoke_without_command=True)
def run_cmd(
    ctx: typer.Context,
    prompt: Annotated[
        str,
        typer.Option(
            "--prompt",
            "-p",
            help=(
                "User prompt to send to the agent. Optional when --resume / "
                "--continue is set — in that case extensions on "
                "decide_turn_action (e.g. llmharness.replay.reminder_seed) "
                "supply the first message via Inject. For multi-turn / TUI "
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
                "``local`` scenario (minimal default tool set)."
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
                "LLM provider to register. Defaults via AGENTM_PROVIDER, "
                "~/.agentm/config.toml profiles, or the provider registry. "
                "Built-ins include 'anthropic' and 'openai'."
            ),
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=(
                "Model id or ~/.agentm/config.toml profile name. Defaults "
                "via AGENTM_MODEL, config.toml default_model, or the "
                "registry default_model for the resolved provider."
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
    max_turns: Annotated[
        int | None,
        typer.Option(
            "--max-turns",
            min=1,
            help=(
                "Hard ceiling on agent-loop turns. Unset = no cap (the agent "
                "runs until it stops on its own / hits --max-tool-calls / is "
                "aborted). Overrides the scenario's `loop_budget` atom."
            ),
        ),
    ] = None,
    max_tool_calls: Annotated[
        int | None,
        typer.Option(
            "--max-tool-calls",
            min=1,
            help=(
                "Hard ceiling on total tool calls across the run. Unset = no "
                "cap. Overrides the scenario's `loop_budget` atom."
            ),
        ),
    ] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            "--reasoning-effort",
            envvar="AGENTM_REASONING_EFFORT",
            help=(
                "Provider reasoning-effort hint (e.g. 'high', 'max'). "
                "Maps to the OpenAI 'reasoning_effort' param and the "
                "Anthropic 'output_config.effort' slot. Precedence: this "
                "flag > AGENTM_REASONING_EFFORT > config.toml profile."
            ),
        ),
    ] = None,
    set_config: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            "-S",
            help=(
                "Override one atom config key. Repeatable. Form: "
                "'<atom>.<key>=<value>' where <atom> is the atom's "
                "MANIFEST.name. The value is coerced to the key's declared "
                "type from config_schema (JSON for array/object). Wins over "
                "AGENTM_<ATOM>_<KEY> env and the scenario's config:. Example: "
                "-S cost_budget.limit=5 -S permission.deny='[\"bash\"]'."
            ),
        ),
    ] = None,
) -> None:
    """Send a single prompt and print the agent's final text."""

    # When a subcommand (``trace`` / ``gateway`` / ``list-extensions``) is
    # being dispatched, this root callback fires first but must defer to the
    # subcommand rather than treat its absent ``--prompt`` as an error.
    if ctx.invoked_subcommand is not None:
        return

    # ``.env`` must load BEFORE provider/model resolution: if the user
    # only set AGENTM_PROVIDER / AGENTM_MODEL in ``.env`` (not in the
    # shell), ``_resolve_provider_model_cwd`` would otherwise read an
    # unset env and fall through to the registry default (anthropic),
    # silently picking the wrong backend. Compute cwd inline using the
    # same precedence ``_resolve_provider_model_cwd`` will apply, so
    # ``--cwd /b`` still consults ``/b/.env`` not the process cwd.
    pre_cwd = cwd or os.environ.get("AGENTM_CWD") or os.getcwd()
    autoload_dotenv(Path(pre_cwd))
    provider, model, cwd, profile = _resolve_provider_model_cwd(
        provider_flag=provider,
        model_flag=model,
        cwd_flag=cwd,
    )

    extra_extensions = _parse_extensions(extension)
    if scenario is None and not no_extensions:
        scenario = os.environ.get("AGENTM_SCENARIO") or DEFAULT_SCENARIO

    if not prompt and not resume and not continue_recent:
        print(
            "ERROR: prompt is required for a fresh session.\n"
            "       Pass --resume <sid> (or --continue) to advance an existing\n"
            "       session whose first message is supplied by an extension.\n"
            "       For multi-turn / chat use, run the single-process gateway\n"
            "       and connect a chat client:\n"
            "         agentm gateway --bind unix:///tmp/agentm/gw.sock\n"
            "         agentm-terminal --connect unix:///tmp/agentm/gw.sock --format textual\n"
            "       Cross-host (WebSocket + token):\n"
            "         agentm gateway --bind ws://0.0.0.0:7777/agentm --bind-token-file /etc/agentm/tokens\n"
            "         agentm-terminal --connect ws://gw.example.com:7777/agentm --token \"$AGENTM_TOKEN\"",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    rc = asyncio.run(
        run(CliRunConfig(
            prompt=prompt,
            provider=provider,
            model=model,
            cwd=cwd,
            scenario=scenario,
            extra_extensions=extra_extensions,
            no_extensions=no_extensions,
            no_skills=no_skills,
            no_prompt_templates=no_prompt_templates,
            tool_allowlist=_parse_tools(tools),
            quiet=quiet,
            resume=resume,
            continue_recent=continue_recent,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            atom_config_overrides=_parse_set_overrides(set_config),
            profile=profile,
            reasoning_effort=reasoning_effort,
        ))
    )
    raise typer.Exit(code=rc)


_run = run


@app.command(name="list-extensions")
def list_extensions_cmd(
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help=(
                "Which discovery source to list: 'builtin' (src/agentm/"
                "extensions/builtin/), 'contrib' (<repo>/contrib/extensions/),"
                " 'home' (~/.agentm/contrib/extensions/),"
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
    live at ``<repo>/contrib/extensions/<name>.py``; home atoms are
    user-installed at ``~/.agentm/contrib/extensions/<name>.py``; user
    atoms are committed by ``api.install_atom`` to ``<cwd>/.agentm/atoms/``.
    Mount any of them via ``agentm -e <module.path>`` (or stack on top of a
    ``--scenario``).
    """

    import json

    from agentm.extensions.discover import (
        BuiltinEntry,
        discover_builtin,
        discover_contrib_atoms,
        discover_home_atoms,
        discover_user_atoms,
    )

    valid_sources = {"all", "builtin", "contrib", "home", "user"}
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
        if source in {"all", "home"}:
            buckets.append(("home", discover_home_atoms()))
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


def _trace_main() -> None:
    from agentm.cli_trace import main as trace_main

    trace_main()


def _gateway_main() -> None:
    from agentm.gateway.cli import main as gateway_main

    gateway_main()


# Subcommands whose own Typer app owns argv parsing, help, and exit codes.
# The importer is called only when that subcommand is actually dispatched, so
# the default prompt path and ``agentm --help`` never import the ``trace`` /
# ``gateway`` dependency closures (gateway pulls in the websockets server).
# The short-help string mirrors each app's own short help so ``agentm --help``
# can list the subcommand without importing it.
_LAZY_SUBCOMMANDS: dict[str, tuple[Any, str]] = {
    "trace": (
        _trace_main,
        "Query an OTLP/JSON session log written by the observability atom.",
    ),
    "gateway": (
        _gateway_main,
        "Single-process gateway: hold all chat sessions and serve chat clients.",
    ),
}


def _build_command() -> Any:
    """Compile the Click command tree for the prompt / ``list-extensions`` path.

    Only the default prompt runner (root callback) and ``list-extensions`` are
    real commands here. ``trace`` and ``gateway`` are added as import-free
    placeholder commands purely so ``agentm --help`` lists them with their
    short help — they are never executed through this tree. Their real Typer
    apps are dispatched out of band by :func:`main`, which imports the chosen
    one lazily; so neither this build nor ``--help`` pays for their dependency
    closures.
    """

    import click

    root = cast(click.Group, typer.main.get_command(app))
    for name, (_importer, short_help) in _LAZY_SUBCOMMANDS.items():
        root.add_command(click.Command(name=name, help=short_help), name)
    return root


def main() -> None:
    """Entry point referenced by the ``agentm`` console script.

    * ``agentm`` (no args) — show help with the subcommand list (exit 0).
    * ``agentm -p "prompt" [options]`` — default single-shot prompt runner.
    * ``agentm {trace,gateway} ...`` — lazily handed to that app (its own argv
      parsing / help / exit codes); the other subcommand stays unimported.
    * ``agentm list-extensions ...`` — runs in the compiled command tree.
    """

    argv = sys.argv[1:]
    if argv and argv[0] in _LAZY_SUBCOMMANDS:
        sub = argv[0]
        # Rewrite argv so the subcommand's app sees ``<prog> <sub>`` as prog
        # name and its own flags as args, matching standalone invocation.
        sys.argv = [f"{sys.argv[0]} {sub}", *argv[1:]]
        importer = _LAZY_SUBCOMMANDS[sub][0]
        importer()
        return

    root = _build_command()
    if not argv:
        # Bare ``agentm`` shows help and exits 0 — without this the root
        # callback would fire with an empty prompt and exit 2.
        root(args=["--help"], standalone_mode=True)
    root()
