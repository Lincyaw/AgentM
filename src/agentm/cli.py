"""AgentM CLI (typer-based).

Single runtime: load the ``chatbot`` scenario by default (the curated
conversation-oriented atom set). A different curated list is opted into via
``--scenario X``.
Subsystems are turned off via ``--no-*`` flags. Failures during
construction emit diagnostics through the EventBus rather than raising;
only a missing provider is fatal.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TextIO, cast

import typer
from loguru import logger

from agentm.ai import DEFAULT_PROVIDER_REGISTRY, ProviderRegistry
from agentm.cli_daemon import app as _daemon_app
from agentm.cli_trace import app as _trace_app
from agentm.cli_validate import app as _validate_app
from agentm.cli_workflow import app as _workflow_app
from agentm.code_health import app as _lint_app
from agentm.contrib_sync import app as _contrib_app
from agentm.env import autoload_dotenv, resolve_cli_cwd
from agentm.core.abi import LoopConfig
from agentm.core.abi.events import (
    DiagnosticEvent,
    EventBus,
    ExtensionInstallEvent,
    MessagePersistedEvent,
)
from agentm.core.abi.session_store import SessionState, SessionStore
from agentm.core.lib.render import final_summary
from agentm.core.lib.user_config import (
    ModelBuildConfig,
    ModelProfile,
    apply_reasoning_effort,
)
from agentm.gateway.cli import app as _gateway_app


# Default scenario when the user does not pass ``--scenario`` and does
# not opt out via ``--no-extensions``. Module-level constant so it can
# be patched from tests and referenced by env-var fallthrough below.
DEFAULT_SCENARIO = "chatbot"

# #201: defense-in-depth bound on the one-shot ``idle()`` wait. The unbounded
# wait makes one-shot exit hinge entirely on ``track_background`` accounting
# being perfect; a leaked counter or a stuck background tool would hang
# ``agentm -p`` silently forever. This generous ceiling (well above any normal
# late-completion window — late completions land in seconds, not minutes)
# converts that into a visible warning + clean exit. Long-lived hosts (gateway
# / worker / TUI) own their loop and never call ``idle()``.
ONESHOT_IDLE_TIMEOUT_SECONDS = 120.0

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False
)

# -- Shared Typer option aliases ------------------------------------------------
# Used by both ``run_cmd`` and ``fork_cmd`` so each option is defined once.
# Pattern follows ``cli_trace.py`` (``SessionOpt``, ``FileOpt``, etc.).

ScenarioOpt = Annotated[
    str | None,
    typer.Option(
        "--scenario",
        help=(
            "Opt-in curated extension list. Bare names resolve from "
            "$AGENTM_PROJECT_ROOT, the process cwd, $AGENTM_HOME/contrib "
            "(default ~/.agentm/contrib), the AgentM checkout, or packaged "
            "portable scenarios. An absolute path is also accepted. When "
            "unset, falls back to the ``chatbot`` scenario. Use "
            "`agentm list-scenarios` to browse available scenarios."
        ),
    ),
]
ExtensionOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--extension",
        "-e",
        help=(
            "Mount an extra atom on top of --scenario / auto-discovery. "
            "Repeatable. Form: 'dotted.module.path' or "
            '\'dotted.module.path:{"key":"value"}\' for inline JSON '
            "config. Example: -e llmharness.atom "
            "-e some.atom:'{\"k\":3}'. Use `agentm list-extensions` "
            "to browse available atoms."
        ),
    ),
]
NoExtensionsOpt = Annotated[
    bool,
    typer.Option(
        "--no-extensions",
        help="Skip atom discovery and loading; agent runs with no tools.",
    ),
]
NoSkillsOpt = Annotated[
    bool,
    typer.Option(
        "--no-skills",
        help="Skip skill discovery in the resource loader.",
    ),
]
NoPromptTemplatesOpt = Annotated[
    bool,
    typer.Option(
        "--no-prompt-templates",
        help="Skip prompt-template discovery in the resource loader.",
    ),
]
ToolsOpt = Annotated[
    str | None,
    typer.Option(
        "--tools",
        help="Comma-separated allowlist applied to atom-registered tools.",
    ),
]
ProviderOpt = Annotated[
    str | None,
    typer.Option(
        "--provider",
        help=(
            "LLM provider to register. Defaults via AGENTM_PROVIDER, "
            "$AGENTM_HOME/config.toml profiles, or the provider registry. "
            "Built-ins include 'anthropic' and 'openai'."
        ),
    ),
]
ModelOpt = Annotated[
    str | None,
    typer.Option(
        "--model",
        help=(
            "Model id or $AGENTM_HOME/config.toml profile name. Defaults "
            "via AGENTM_MODEL, config.toml default_model, or the "
            "registry default_model for the resolved provider."
        ),
    ),
]
CwdOpt = Annotated[
    str | None,
    typer.Option(
        "--cwd",
        help=(
            "Working directory exposed to extensions. Defaults to "
            "AGENTM_CWD when set, otherwise the process cwd at "
            "invocation time."
        ),
    ),
]
QuietOpt = Annotated[
    bool,
    typer.Option("--quiet", help="Suppress final-output printout."),
]
MaxTurnsOpt = Annotated[
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
]
MaxToolCallsOpt = Annotated[
    int | None,
    typer.Option(
        "--max-tool-calls",
        min=1,
        help=(
            "Hard ceiling on total tool calls across the run. Unset = no "
            "cap. Overrides the scenario's `loop_budget` atom."
        ),
    ),
]
ReasoningEffortOpt = Annotated[
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
]
SetConfigOpt = Annotated[
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
]


@dataclass(slots=True)
class CliRunConfig:
    """Resolved CLI flags for session construction.

    Bundles every knob that flows from ``run_cmd`` through ``run`` into
    ``_build_session_config``, replacing the ~16-parameter signatures.
    """

    provider: str
    model: str
    cwd: str
    prompt: str = ""
    provider_explicit: bool = False
    model_explicit: bool = False
    scenario: str | None = None
    extra_extensions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    no_extensions: bool = False
    no_skills: bool = False
    no_prompt_templates: bool = False
    tool_allowlist: list[str] | None = None
    quiet: bool = False
    resume: str | None = None
    continue_recent: bool = False
    fork: str | None = None
    fork_up_to: int | None = None
    fork_message_id: str | None = None
    fork_turn_id: int | None = None
    fork_turn_index: int | None = None
    max_turns: int | None = None
    max_tool_calls: int | None = None
    atom_config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    profile: ModelProfile | None = None
    reasoning_effort: str | None = None


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

    Special forms:

    - ``atom=@path.json`` (no dot in LHS): load the entire atom config from
      the JSON/YAML file at *path*. The file must parse to a mapping.
    - ``atom.key=@path.json``: load the value for one key from a file.
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

        # Whole-atom config from file: ``atom=@path.json``
        if not dot and value.startswith("@"):
            file_path = Path(value[1:])
            if not file_path.is_file():
                raise typer.BadParameter(
                    f"--set {raw!r}: file not found: {file_path}"
                )
            content = file_path.read_text(encoding="utf-8")
            if file_path.suffix in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(content)
            else:
                data = json.loads(content)
            if not isinstance(data, dict):
                raise typer.BadParameter(
                    f"--set {raw!r}: file must contain a JSON/YAML mapping"
                )
            out.setdefault(atom, {}).update(data)
            continue

        if not dot or not atom or not key:
            raise typer.BadParameter(
                f"--set {raw!r}: left of '=' must be '<atom>.<key>'"
            )

        # Single-key from file: ``atom.key=@path.json``
        if value.startswith("@"):
            file_path = Path(value[1:])
            if not file_path.is_file():
                raise typer.BadParameter(
                    f"--set {raw!r}: file not found: {file_path}"
                )
            content = file_path.read_text(encoding="utf-8")
            if file_path.suffix in (".yaml", ".yml"):
                import yaml
                value = yaml.safe_load(content)
            else:
                try:
                    value = json.loads(content)
                except json.JSONDecodeError as exc:
                    # Non-JSON file content is kept as a raw string value.
                    logger.debug("cli: --set file {} not JSON, keeping raw: {}", file_path, exc)
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

    Returns ``(provider, model, cwd, profile)``.
    """
    from agentm.core.lib.user_config import resolve_provider_model

    provider, model, profile = resolve_provider_model(
        provider_flag=provider_flag,
        model_flag=model_flag,
        registry=registry,
    )
    cwd = str(resolve_cli_cwd(cwd_flag))
    return provider, model, cwd, profile


def _make_default_session_store(cwd: str) -> SessionStore:
    from agentm.core.runtime.session_bootstrap import make_default_session_store

    return make_default_session_store(cwd)


def _resolve_session_state(
    *,
    cwd: str,
    resume: str | None,
    continue_recent: bool,
    fork: str | None = None,
    fork_up_to: int | None = None,
    fork_message_id: str | None = None,
    fork_turn_id: int | None = None,
    fork_turn_index: int | None = None,
    session_store: SessionStore,
) -> SessionState:
    from agentm.core.runtime.session_bootstrap import resolve_session_state

    try:
        return resolve_session_state(
            cwd=cwd,
            resume=resume,
            continue_recent=continue_recent,
            fork=fork,
            fork_up_to=fork_up_to,
            fork_message_id=fork_message_id,
            fork_turn_id=fork_turn_id,
            fork_turn_index=fork_turn_index,
            session_store=session_store,
        )
    except FileNotFoundError as exc:
        flag = f"--fork {fork!r}" if fork else f"--resume {resume!r}"
        raise typer.BadParameter(
            f"{flag}: no session found for cwd {cwd!r}"
        ) from exc
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _get_stored_session_config(session_state: Any) -> dict[str, Any] | None:
    """Read the persisted session config from a resumed/forked session."""
    header = getattr(session_state, "get_header", lambda: None)()
    if header is None:
        return None
    stored_config = getattr(header, "config", None)
    return stored_config if isinstance(stored_config, dict) else None


def _stored_metadata(
    stored: dict[str, Any] | None, key: str
) -> dict[str, Any] | None:
    value = stored.get(key) if stored else None
    return dict(value) if isinstance(value, dict) else None


def _build_cli_lineage(
    config: CliRunConfig,
    stored: dict[str, Any] | None,
) -> dict[str, Any]:
    if config.fork:
        fork_point: dict[str, Any] = {}
        if config.fork_up_to is not None:
            fork_point["up_to"] = config.fork_up_to
        if config.fork_message_id is not None:
            fork_point["message_id"] = config.fork_message_id
        if config.fork_turn_id is not None:
            fork_point["turn_id"] = config.fork_turn_id
        if config.fork_turn_index is not None:
            fork_point["turn_index"] = config.fork_turn_index
        if not fork_point:
            fork_point["up_to"] = "end"
        lineage: dict[str, Any] = {
            "kind": "fork",
            "entrypoint": "agentm.cli",
            "source_session_id": config.fork,
            "fork_point": fork_point,
        }
        source_lineage = _stored_metadata(stored, "lineage")
        if source_lineage is not None:
            lineage["source_lineage"] = source_lineage
        return lineage

    stored_lineage = _stored_metadata(stored, "lineage")
    if stored_lineage is not None:
        return stored_lineage
    return {
        "kind": "root",
        "entrypoint": "agentm.cli",
        "prompt": bool(config.prompt),
    }


def _make_install_warner() -> Any:
    def _on_install(event: ExtensionInstallEvent) -> None:
        if event.phase != "error":
            return
        logger.warning(
            "[extension_install] {path}: {error}",
            path=event.module_path,
            error=event.error,
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
        log_fn = {
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
        }.get(event.level, logger.info)
        log_fn("[{source}] {message}", source=event.source, message=event.message)
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
        fork=config.fork,
        fork_up_to=config.fork_up_to,
        fork_message_id=config.fork_message_id,
        fork_turn_id=config.fork_turn_id,
        fork_turn_index=config.fork_turn_index,
        session_store=store,
    )

    # Restore config from source session for fork/resume when not
    # explicitly overridden on the CLI.
    stored = _get_stored_session_config(session_state)
    scenario = config.scenario
    if scenario is None and stored:
        scenario = stored.get("scenario")
    if scenario is None and not config.no_extensions:
        scenario = DEFAULT_SCENARIO
    if scenario is not None and not config.no_extensions:
        _validate_scenario_arg(scenario)
    stored_provider = stored.get("provider") if stored else None
    provider_spec: Any
    if (
        isinstance(stored_provider, list)
        and len(stored_provider) == 2
        and isinstance(stored_provider[0], str)
        and not config.provider_explicit
        and not (config.model_explicit and config.profile is not None)
    ):
        stored_cfg = stored_provider[1] if isinstance(stored_provider[1], dict) else {}
        provider_cfg = cast(ModelBuildConfig, dict(stored_cfg))
        if config.model_explicit:
            provider_cfg["model"] = config.model
        apply_reasoning_effort(provider_cfg, config.reasoning_effort)
        provider_spec = (stored_provider[0], provider_cfg)
    else:
        build_config: ModelBuildConfig
        if config.profile is not None:
            build_config = config.profile.to_build_config()
        else:
            build_config = {"model": config.model}
        apply_reasoning_effort(build_config, config.reasoning_effort)
        try:
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
    if config.fork is not None and parent_session_id is None:
        parent_session_id = config.fork

    lineage = _build_cli_lineage(config, stored)
    experiment = _stored_metadata(stored, "experiment")

    return (
        AgentSessionConfig(
            cwd=config.cwd,
            provider=provider_spec,
            scenario=scenario,
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
            lineage=lineage,
            experiment=experiment,
        ),
        session_state,
    )


def _validate_scenario_arg(scenario: str) -> None:
    from agentm.extensions.loader import ScenarioLoadError, validate_scenario

    try:
        validate_scenario(scenario)
    except ScenarioLoadError as exc:
        raise typer.BadParameter(f"--scenario {scenario!r}: {exc}") from exc


async def run(
    config: CliRunConfig,
    *,
    output: TextIO = sys.stdout,
) -> int:
    from agentm.core.observability.otel_export import attach_loguru_otel_sink
    from agentm.core.runtime.session import AgentSession

    # Ship operational logs into the OTel logs pipeline (→ collector →
    # ClickHouse) when an OTLP endpoint is configured. No-op otherwise.
    attach_loguru_otel_sink()

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
    if not config.quiet:
        sid = session_manager.get_session_id()
        if sid:
            if session_manager.session_file is not None:
                logger.info("session log: {path}", path=session_manager.session_file)
            logger.info("session id: {sid}", sid=sid)
            # The ``agentm trace`` command is logged centrally by
            # create_agent_session for every session, so it is not repeated here.

    session = await AgentSession.create(session_config)
    try:
        # The prompt/resume return value is intentionally discarded: the final
        # message list is re-fetched AFTER ``idle()`` below so it reflects any
        # late background completion delivered between agent_end and idle.
        if config.prompt:
            await session.prompt(config.prompt)
        else:
            # No prompt (``--resume`` / ``--fork`` / ``--continue``): continue the
            # agent on its current context. ``resume()`` runs one more round (the
            # model is re-invoked on the existing/forked messages) — unlike
            # ``tick()``, which only advances when a resume-atom injects and so
            # leaves a plain forked trajectory parked with no new turns.
            await session.resume()
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
            logger.warning(
                "timed out waiting for background work to finish after "
                "{timeout:.0f}s; exiting anyway. Any late background "
                "completion will not be delivered.",
                timeout=ONESHOT_IDLE_TIMEOUT_SECONDS,
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
                "--continue is set and initial messages already exist. For multi-turn / TUI "
                "use, run `agentm terminal`."
            ),
        ),
    ] = "",
    scenario: ScenarioOpt = None,
    extension: ExtensionOpt = None,
    no_extensions: NoExtensionsOpt = False,
    no_skills: NoSkillsOpt = False,
    no_prompt_templates: NoPromptTemplatesOpt = False,
    tools: ToolsOpt = None,
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    cwd: CwdOpt = None,
    quiet: QuietOpt = False,
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
    fork: Annotated[
        str | None,
        typer.Option(
            "--fork",
            help=(
                "Fork from an existing session. Creates a new session seeded "
                "with the source session's trajectory (or a prefix via "
                "--from-turn). The source session is not modified."
            ),
        ),
    ] = None,
    from_turn: Annotated[
        int | None,
        typer.Option(
            "--from-turn",
            min=1,
            help=(
                "Used with --fork: copy only the first N messages from the "
                "source session. Without this flag, all messages are copied."
            ),
        ),
    ] = None,
    message_id: Annotated[
        str | None,
        typer.Option(
            "--message-id",
            help=(
                "Used with --fork: fork at an exact persisted message entry id. "
                "Copy the id from `agentm trace messages`."
            ),
        ),
    ] = None,
    turn_id: Annotated[
        int | None,
        typer.Option(
            "--turn-id",
            help=(
                "Used with --fork: fork at the unique trace turn_id. "
                "If ambiguous, use --message-id."
            ),
        ),
    ] = None,
    turn_index: Annotated[
        int | None,
        typer.Option(
            "--turn-index",
            help=(
                "Used with --fork: fork at the unique trace turn_index. "
                "If ambiguous, use --message-id."
            ),
        ),
    ] = None,
    max_turns: MaxTurnsOpt = None,
    max_tool_calls: MaxToolCallsOpt = None,
    reasoning_effort: ReasoningEffortOpt = None,
    set_config: SetConfigOpt = None,
) -> None:
    """Send a single prompt and print the agent's final text."""

    # ``.env`` must load BEFORE provider/model resolution: if the user
    # only set AGENTM_PROVIDER / AGENTM_MODEL in ``.env`` (not in the
    # shell), ``_resolve_provider_model_cwd`` would otherwise read an
    # unset env and fall through to the registry default (anthropic),
    # silently picking the wrong backend. Compute cwd inline using the
    # same precedence ``_resolve_provider_model_cwd`` will apply, so
    # ``--cwd /b`` still consults ``/b/.env`` not the process cwd.
    pre_cwd = resolve_cli_cwd(cwd)
    autoload_dotenv(pre_cwd)

    # When a subcommand (``trace`` / ``daemon`` / ``gateway`` /
    # ``list-extensions`` / ``list-scenarios``) is
    # being dispatched, this root callback fires first but must defer to the
    # subcommand rather than treat its absent ``--prompt`` as an error. The
    # dotenv load above is intentionally shared with subcommands so trace
    # backends see repo-local observability env.
    if ctx.invoked_subcommand is not None:
        return
    provider_was_explicit = provider is not None
    model_was_explicit = model is not None
    provider, model, cwd, profile = _resolve_provider_model_cwd(
        provider_flag=provider,
        model_flag=model,
        cwd_flag=cwd,
    )

    extra_extensions = _parse_extensions(extension)
    if scenario is None and not no_extensions:
        env_scenario = os.environ.get("AGENTM_SCENARIO")
        if env_scenario is not None:
            scenario = env_scenario
        elif not (resume or continue_recent or fork):
            scenario = DEFAULT_SCENARIO

    if not prompt and not resume and not continue_recent and not fork:
        print(ctx.get_help())
        raise typer.Exit(code=0)
    fork_selectors = [
        from_turn is not None,
        message_id is not None,
        turn_id is not None,
        turn_index is not None,
    ]
    if sum(fork_selectors) > 1:
        raise typer.BadParameter(
            "--from-turn, --message-id, --turn-id, and --turn-index are mutually exclusive"
        )
    if not fork and any(fork_selectors):
        raise typer.BadParameter(
            "--from-turn, --message-id, --turn-id, and --turn-index require --fork"
        )

    rc = asyncio.run(
        run(CliRunConfig(
            prompt=prompt,
            provider=provider,
            model=model,
            cwd=cwd,
            provider_explicit=provider_was_explicit,
            model_explicit=model_was_explicit,
            scenario=scenario,
            extra_extensions=extra_extensions,
            no_extensions=no_extensions,
            no_skills=no_skills,
            no_prompt_templates=no_prompt_templates,
            tool_allowlist=_parse_tools(tools),
            quiet=quiet,
            resume=resume,
            continue_recent=continue_recent,
            fork=fork,
            fork_up_to=from_turn,
            fork_message_id=message_id,
            fork_turn_id=turn_id,
            fork_turn_index=turn_index,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            atom_config_overrides=_parse_set_overrides(set_config),
            profile=profile,
            reasoning_effort=reasoning_effort,
        ))
    )
    raise typer.Exit(code=rc)


@app.command(name="fork")
def fork_cmd(
    source: Annotated[
        str,
        typer.Argument(
            help=(
                "Source session id or .jsonl path. Use `agentm trace messages` "
                "or `agentm trace turns` to choose the fork point."
            ),
        ),
    ],
    prompt: Annotated[
        str,
        typer.Option(
            "--prompt",
            "-p",
            help="User prompt to send after creating the fork.",
        ),
    ],
    message_id: Annotated[
        str | None,
        typer.Option(
            "--message-id",
            help="Fork at an exact persisted message entry id.",
        ),
    ] = None,
    turn_id: Annotated[
        int | None,
        typer.Option(
            "--turn-id",
            help="Fork at a unique trace turn_id.",
        ),
    ] = None,
    turn_index: Annotated[
        int | None,
        typer.Option(
            "--turn-index",
            help="Fork at a unique trace turn_index.",
        ),
    ] = None,
    scenario: ScenarioOpt = None,
    extension: ExtensionOpt = None,
    no_extensions: NoExtensionsOpt = False,
    no_skills: NoSkillsOpt = False,
    no_prompt_templates: NoPromptTemplatesOpt = False,
    tools: ToolsOpt = None,
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    cwd: CwdOpt = None,
    quiet: QuietOpt = False,
    max_turns: MaxTurnsOpt = None,
    max_tool_calls: MaxToolCallsOpt = None,
    reasoning_effort: ReasoningEffortOpt = None,
    set_config: SetConfigOpt = None,
) -> None:
    """Fork a session at a message/turn boundary and continue with a prompt."""

    selectors = [
        message_id is not None,
        turn_id is not None,
        turn_index is not None,
    ]
    if sum(selectors) != 1:
        raise typer.BadParameter(
            "specify exactly one of --message-id, --turn-id, or --turn-index"
        )

    pre_cwd = resolve_cli_cwd(cwd)
    autoload_dotenv(pre_cwd)
    provider_was_explicit = provider is not None
    model_was_explicit = model is not None
    provider, model, cwd, profile = _resolve_provider_model_cwd(
        provider_flag=provider,
        model_flag=model,
        cwd_flag=cwd,
    )
    if scenario is None and not no_extensions:
        scenario = os.environ.get("AGENTM_SCENARIO")

    rc = asyncio.run(
        run(CliRunConfig(
            prompt=prompt,
            provider=provider,
            model=model,
            cwd=cwd,
            provider_explicit=provider_was_explicit,
            model_explicit=model_was_explicit,
            scenario=scenario,
            extra_extensions=_parse_extensions(extension),
            no_extensions=no_extensions,
            no_skills=no_skills,
            no_prompt_templates=no_prompt_templates,
            tool_allowlist=_parse_tools(tools),
            quiet=quiet,
            fork=source,
            fork_message_id=message_id,
            fork_turn_id=turn_id,
            fork_turn_index=turn_index,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            atom_config_overrides=_parse_set_overrides(set_config),
            profile=profile,
            reasoning_effort=reasoning_effort,
        ))
    )
    raise typer.Exit(code=rc)


@app.command(name="list-extensions")
def list_extensions_cmd(
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help=(
                "Which discovery source to list: 'builtin' (src/agentm/"
                "extensions/builtin/), 'contrib' (<repo>/contrib/extensions/),"
                " 'home' ($AGENTM_HOME/contrib/extensions; default "
                "~/.agentm/contrib/extensions/),"
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
        str | None,
        typer.Option(
            "--cwd",
            help="Working directory used for 'user' source discovery.",
        ),
    ] = None,
) -> None:
    """List discoverable extensions (atoms) by source.

    Builtins ship under ``src/agentm/extensions/builtin/``; contrib atoms
    live at ``<repo>/contrib/extensions/<name>.py``; home atoms are
    user-installed at ``$AGENTM_HOME/contrib/extensions/<name>.py`` (default
    ``~/.agentm/contrib/extensions/<name>.py``); user atoms are committed by
    ``api.install_atom`` to ``<cwd>/.agentm/atoms/``.
    Mount any of them via ``agentm -e <module.path>`` (or stack on top of a
    ``--scenario``).
    """

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
            buckets.append(("user", discover_user_atoms(resolve_cli_cwd(cwd))))
    except Exception as exc:
        logger.error("discovery failed: {exc}", exc=exc)
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
    logger.info("{total} extension(s) shown.", total=total)


@app.command(name="list-scenarios")
def list_scenarios_cmd(
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
            help="Substring filter applied to scenario name, source, path, or description.",
        ),
    ] = None,
) -> None:
    """List installed home-contrib scenarios usable with ``--scenario``."""

    from agentm.extensions.loader import list_scenarios

    if output_format not in {"text", "json"}:
        raise typer.BadParameter(
            f"--format {output_format!r}: must be 'text' or 'json'"
        )

    entries = list_scenarios()
    needle = filter_substr.lower() if filter_substr else None

    def matches(entry: Any) -> bool:
        if needle is None:
            return True
        return (
            needle in entry.name.lower()
            or needle in entry.source.lower()
            or needle in entry.manifest_path.lower()
            or needle in entry.description.lower()
        )

    filtered = [entry for entry in entries if matches(entry)]
    if output_format == "json":
        print(
            json.dumps(
                [
                    {
                        "name": entry.name,
                        "source": entry.source,
                        "manifest_path": entry.manifest_path,
                        "description": entry.description,
                    }
                    for entry in filtered
                ],
                indent=2,
            )
        )
        return

    print(f"# scenarios ({len(filtered)})")
    if not filtered:
        print("  (none)")
        print("\nInstalled scenarios live under `$AGENTM_HOME/contrib/scenarios`.")
        return
    name_width = min(max(32, max(len(entry.name) for entry in filtered)), 48)
    for entry in filtered:
        summary = _compact_text(entry.description, 96)
        suffix = f"  {summary}" if summary else ""
        print(f"  {entry.name:{name_width}s} {entry.source:12s}{suffix}")
    print(
        "\nInstalled scenarios live under `$AGENTM_HOME/contrib/scenarios` "
        "(default: `~/.agentm/contrib/scenarios`). Use with `--scenario <name>` "
        "or `/scenario <name>` in terminal."
    )


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


@app.command(
    name="terminal",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def terminal_cmd(
    ctx: typer.Context,
    connect: Annotated[
        str | None,
        typer.Option(
            "--connect",
            help=(
                "Existing gateway URL. Omit this to start or reuse the local "
                "gateway daemon."
            ),
        ),
    ] = None,
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            help=(
                "Scenario for the local gateway daemon or the first terminal "
                "message. Defaults to chatbot in one-command mode."
            ),
        ),
    ] = None,
    cwd: CwdOpt = None,
    state_dir: Annotated[
        Path | None,
        typer.Option(
            "--state-dir",
            envvar="AGENTM_STATE_DIR",
            help="Persistent gateway state dir. Only used when starting a local gateway.",
        ),
    ] = None,
    terminal_bin: Annotated[
        str,
        typer.Option(
            "--terminal-bin",
            envvar="AGENTM_TERMINAL_BIN",
            help="Terminal peer executable.",
        ),
    ] = "agentm-terminal",
    terminal_log: Annotated[
        Path | None,
        typer.Option(
            "--terminal-log",
            help="Terminal peer log file. Extra TUI flags may also be passed after --.",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Reconnect to a known terminal session id. Default: new session per terminal.",
        ),
    ] = None,
    simple: Annotated[
        bool,
        typer.Option(
            "--simple",
            help="Use the compact terminal layout.",
        ),
    ] = False,
    theme: Annotated[
        str | None,
        typer.Option(
            "--theme",
            help="Terminal theme: dark or light.",
        ),
    ] = None,
    gateway_log: Annotated[
        Path | None,
        typer.Option(
            "--gateway-log",
            envvar="AGENTM_TERMINAL_GATEWAY_LOG",
            help="Gateway daemon/supervisor log file. Default: $AGENTM_HOME/logs/terminal-gateway.log.",
        ),
    ] = None,
    private_gateway: Annotated[
        bool,
        typer.Option(
            "--private-gateway",
            help="Start a gateway only for this terminal and stop it when the TUI exits.",
        ),
    ] = False,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload/--no-reload",
            help=(
                "Enable daemon supervisor source watching and worker restarts. "
                "Default: disabled."
            ),
        ),
    ] = False,
    gateway_command: Annotated[
        str,
        typer.Option(
            "--gateway-command",
            help="Command used to launch a --private-gateway worker.",
            hidden=True,
        ),
    ] = "agentm",
    startup_timeout: Annotated[
        float,
        typer.Option(
            "--startup-timeout",
            min=0.1,
            help="Seconds to wait for the local gateway endpoint.",
        ),
    ] = 10.0,
) -> None:
    """Open the terminal UI, starting/reusing the local gateway daemon by default."""

    from agentm.terminal_launcher import (
        TerminalLaunchConfig,
        TerminalLaunchError,
        run_terminal,
    )

    resolved_cwd = resolve_cli_cwd(cwd)
    autoload_dotenv(resolved_cwd)
    resolved_scenario = (
        scenario if scenario is not None else os.environ.get("AGENTM_SCENARIO")
    )
    if not connect and resolved_scenario is None:
        resolved_scenario = DEFAULT_SCENARIO

    if connect and (
        state_dir is not None
        or gateway_log is not None
        or private_gateway
        or reload
        or gateway_command != "agentm"
    ):
        raise typer.BadParameter(
            "--state-dir, --gateway-log, --private-gateway, --reload, and "
            "--gateway-command only apply when agentm terminal starts the local gateway"
        )

    try:
        rc = run_terminal(
            TerminalLaunchConfig(
                cwd=resolved_cwd,
                connect=connect,
                scenario=resolved_scenario,
                state_dir=state_dir,
                terminal_bin=terminal_bin,
                terminal_log=terminal_log,
                session_id=session_id,
                simple=simple,
                theme=theme,
                gateway_log=gateway_log,
                gateway_command=gateway_command,
                terminal_args=list(ctx.args),
                startup_timeout=startup_timeout,
                use_daemon=not private_gateway,
                reload=reload,
            )
        )
    except TerminalLaunchError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=7) from exc
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    raise typer.Exit(code=rc)


@app.command(name="setup")
def setup_cmd(
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help="config.toml profile name to create or update.",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="Provider id, e.g. openai or anthropic. Defaults from env/config.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Model id or existing config.toml profile to use.",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="Provider API key to store in config.toml (chmod 0600).",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Optional OpenAI/Anthropic-compatible base URL.",
        ),
    ] = None,
    context_window: Annotated[
        int | None,
        typer.Option(
            "--context-window",
            help="Optional context window recorded on the model profile.",
        ),
    ] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            "--reasoning-effort",
            help="Optional reasoning-effort hint recorded on the model profile.",
        ),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "--cwd",
            help="Workspace to initialize. Defaults to the current directory.",
        ),
    ] = None,
    bot_name: Annotated[
        str,
        typer.Option("--name", help="Persona name for seeded IDENTITY.md."),
    ] = "Assistant",
    voice: Annotated[
        str,
        typer.Option("--voice", help="Optional extra voice/tone line for SOUL.md."),
    ] = "",
    quick: Annotated[
        bool,
        typer.Option(
            "--quick",
            help="Non-interactive setup. Fails if required model credentials are missing.",
        ),
    ] = False,
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="Show setup status without writing files.",
        ),
    ] = False,
    test_model: Annotated[
        bool,
        typer.Option(
            "--test",
            help="Run one real model request to verify credentials and model access.",
        ),
    ] = False,
    test_prompt: Annotated[
        str,
        typer.Option(
            "--test-prompt",
            help="Prompt used by --test.",
        ),
    ] = "Reply with exactly: agentm-ok",
    no_contrib: Annotated[
        bool,
        typer.Option(
            "--no-contrib",
            help="Skip installing demo scenarios into AGENTM_HOME.",
        ),
    ] = False,
    no_skills: Annotated[
        bool,
        typer.Option(
            "--no-skills",
            help="Skip installing bundled SKILL.md resources into AGENTM_HOME.",
        ),
    ] = False,
    no_persona: Annotated[
        bool,
        typer.Option(
            "--no-persona",
            help="Skip seeding SOUL.md / IDENTITY.md / USER.md in the workspace.",
        ),
    ] = False,
    force_model: Annotated[
        bool,
        typer.Option(
            "--force-model",
            help="Overwrite/create the model profile even when a default already exists.",
        ),
    ] = False,
) -> None:
    """Set up AgentM for first use with the fewest required choices."""
    from agentm.onboard import run_setup

    raise typer.Exit(
        code=run_setup(
            profile=profile,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            context_window=context_window,
            reasoning_effort=reasoning_effort,
            workspace=workspace,
            bot_name=bot_name,
            voice=voice,
            quick=quick,
            check=check,
            test_model=test_model,
            test_prompt=test_prompt,
            install_demo_scenarios=not no_contrib,
            install_skills=not no_skills,
            seed_persona_files=not no_persona,
            force_model=force_model,
        )
    )


@app.command(name="onboard")
def onboard_cmd() -> None:
    """Interactively bootstrap a fresh install."""
    from agentm.onboard import run_onboard

    run_onboard()


app.add_typer(_trace_app, name="trace")
app.add_typer(_workflow_app, name="workflow")
app.add_typer(_validate_app, name="validate")
app.add_typer(_daemon_app, name="daemon")
app.add_typer(_gateway_app, name="gateway")
app.add_typer(_contrib_app, name="contrib")
app.add_typer(_lint_app, name="lint")


def main() -> None:
    """Entry point for the ``agentm`` console script."""
    app()
