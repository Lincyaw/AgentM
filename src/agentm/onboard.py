"""Bootstrap helpers for a fresh AgentM install (``agentm setup`` / onboard).

Walks the user through five sections and writes their answers to the exact
files the runtime already reads:

1. Model / provider / key  -> ``$AGENTM_HOME/config.toml`` (read by
   ``agentm.core.lib.user_config.load_user_config``). The api_key is stored
   inline; the file is ``chmod 0600``.
2. Workspace path          -> created with its ``.agentm/`` subdir.
3. Persona                 -> ``<workspace>/SOUL.md`` + ``IDENTITY.md`` in the
   same markdown skeleton the ``persona`` contrib atom seeds.
4. Bundled skills          -> ``~/.agentm/skills/`` (self-awareness / self-debug
   skills discoverable by every scenario and deploy form).
5. Feishu credentials      -> ``<workspace>/.env`` with ``LARK_APP_ID`` /
   ``LARK_APP_SECRET`` / ``LARK_ALLOW_FROM`` (read by the Feishu client process
   via ``autoload_dotenv``; it does **not** read ``config.toml``).

The flow is idempotent: every existing value is detected and the user is asked
keep/overwrite per item, never silently clobbered.

This is presenter / CLI layer plus this lib module — not an atom (the §11
contract does not apply) and it never touches ``core.runtime.*``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import tomli_w
import typer

from agentm.ai import DEFAULT_PROVIDER_REGISTRY
from agentm.core.lib.user_config import agentm_home_dir

# --- persona skeletons: mirror the ``defaults`` block in
# contrib/scenarios/chatbot/manifest.yaml so an onboard-seeded workspace and a
# persona-atom-seeded workspace look the same. ---

_SOUL_TEMPLATE = """\
# Soul

- Warm but terse. Lead with the answer; skip the throat-clearing.
- Never sycophantic — no "great question". If you don't know, say so.
- Reply in the user's language.
{voice_block}"""

_IDENTITY_TEMPLATE = """\
# Identity

You are {name}, a conversational assistant reachable over a chat channel,
with real file and shell tools in this workspace.
"""

_USER_TEMPLATE = """\
# User

Nothing learned yet. Record durable facts about the user here as
you go (name, preferences, ongoing projects).
"""


def _config_path() -> Path:
    return agentm_home_dir() / "config.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _prompt_keep_or_overwrite(label: str, current: str) -> bool:
    """Return True when the user wants to overwrite *current* for *label*."""
    masked = current if len(current) <= 8 else current[:4] + "…" + current[-2:]
    return typer.confirm(
        f"{label} already set ({masked}). Overwrite?", default=False
    )


# --------------------------------------------------------------------------
# Section 1: model / provider / key -> config.toml
# --------------------------------------------------------------------------


def configure_model(
    *,
    profile: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    context_window: int | None = None,
    reasoning_effort: str | None = None,
) -> Path:
    """Merge a model profile into ``config.toml`` and set it as default.

    Reads any existing config, adds/updates ``[models.<profile>]`` and
    ``default_model`` in memory, writes the whole document back, and
    ``chmod 0600``s the file. The result round-trips through
    ``load_user_config()``.
    """
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_toml(path)

    models = data.get("models")
    if not isinstance(models, dict):
        models = {}
        data["models"] = models

    table: dict[str, Any] = {"provider": provider, "model": model}
    if base_url:
        table["base_url"] = base_url
    if api_key:
        table["api_key"] = api_key
    if context_window is not None:
        table["context_window"] = context_window
    if reasoning_effort:
        table["reasoning_effort"] = reasoning_effort

    models[profile] = table
    data["default_model"] = profile

    with open(path, "wb") as fh:
        tomli_w.dump(data, fh)
    path.chmod(0o600)
    return path


def _default_profile_name() -> str:
    data = _read_toml(_config_path())
    current = data.get("default_model")
    return current if isinstance(current, str) and current else "default"


def _env_base_url(provider: str) -> str | None:
    try:
        descriptor = DEFAULT_PROVIDER_REGISTRY.resolve(provider)
    except KeyError:
        return None
    if descriptor.base_url_env:
        value = os.environ.get(descriptor.base_url_env)
        if value:
            return value
    if descriptor.azure_endpoint_env:
        value = os.environ.get(descriptor.azure_endpoint_env)
        if value:
            return value
    return None


def _first_env_provider() -> str | None:
    for descriptor in DEFAULT_PROVIDER_REGISTRY.descriptors():
        if DEFAULT_PROVIDER_REGISTRY.get_env_api_key(descriptor.id):
            return descriptor.id
    return None


def _provider_default_model(provider: str) -> str:
    try:
        return DEFAULT_PROVIDER_REGISTRY.default_model(provider)
    except KeyError:
        return ""


def _has_existing_default_model() -> bool:
    data = _read_toml(_config_path())
    default_model = data.get("default_model")
    models = data.get("models")
    return (
        isinstance(default_model, str)
        and isinstance(models, dict)
        and isinstance(models.get(default_model), dict)
    )


@dataclass(frozen=True, slots=True)
class ModelSetup:
    profile: str
    provider: str
    model: str
    api_key: str
    base_url: str | None


def _infer_model_setup(
    *,
    profile: str | None,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> ModelSetup:
    resolved_provider = (
        provider
        or os.environ.get("AGENTM_PROVIDER")
        or _first_env_provider()
        or "openai"
    )
    resolved_model = (
        model
        or os.environ.get("AGENTM_MODEL")
        or _provider_default_model(resolved_provider)
    )
    return ModelSetup(
        profile=profile or _default_profile_name(),
        provider=resolved_provider,
        model=resolved_model,
        api_key=api_key
        if api_key is not None
        else (DEFAULT_PROVIDER_REGISTRY.get_env_api_key(resolved_provider) or ""),
        base_url=base_url if base_url is not None else _env_base_url(resolved_provider),
    )


def _redact_secret(value: str | None) -> str:
    if not value:
        return "missing"
    return "set"


# --------------------------------------------------------------------------
# Section 2: workspace
# --------------------------------------------------------------------------


def ensure_workspace(workspace: Path) -> Path:
    """Create *workspace* and its ``.agentm/`` subdir if missing."""
    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".agentm").mkdir(parents=True, exist_ok=True)
    return workspace


# --------------------------------------------------------------------------
# Section 3: persona
# --------------------------------------------------------------------------


def seed_persona(workspace: Path, *, name: str, voice: str) -> list[Path]:
    """Seed ``SOUL.md`` / ``IDENTITY.md`` / ``USER.md`` (absent files only).

    Returns the files actually written.
    """
    voice_block = f"- {voice.strip()}\n" if voice.strip() else ""
    contents = {
        "SOUL.md": _SOUL_TEMPLATE.format(voice_block=voice_block),
        "IDENTITY.md": _IDENTITY_TEMPLATE.format(name=name),
        "USER.md": _USER_TEMPLATE,
    }
    written: list[Path] = []
    for filename, body in contents.items():
        target = workspace / filename
        if target.exists():
            continue
        target.write_text(body, encoding="utf-8")
        written.append(target)
    return written


# --------------------------------------------------------------------------
# Section 4: Feishu credentials -> .env
# --------------------------------------------------------------------------


def _parse_env(path: Path) -> dict[str, str]:
    """Parse a ``KEY=value`` ``.env`` into an ordered dict (comments dropped)."""
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


# The three Feishu keys, spelled exactly as the client reads them. A real
# deployment was broken by an ``ARK_APP_ID`` typo; this constant is the single
# source of truth so the spelling cannot drift between writer and tests.
LARK_KEYS = ("LARK_APP_ID", "LARK_APP_SECRET", "LARK_ALLOW_FROM")


def configure_feishu(
    workspace: Path,
    *,
    app_id: str,
    app_secret: str,
    allow_from: str,
) -> Path:
    """Merge Feishu credentials into ``<workspace>/.env`` (preserving others)."""
    path = workspace / ".env"
    env = _parse_env(path)
    env["LARK_APP_ID"] = app_id
    env["LARK_APP_SECRET"] = app_secret
    env["LARK_ALLOW_FROM"] = allow_from

    body = "".join(f"{key}={value}\n" for key, value in env.items())
    path.write_text(body, encoding="utf-8")
    path.chmod(0o600)
    return path


# --------------------------------------------------------------------------
# Section: bundled skills -> ~/.agentm/skills
# --------------------------------------------------------------------------

# Curated repo skills copied into ~/.agentm/skills so skill_loader discovers
# them in every scenario and every deploy form. Each entry is a repo-relative
# skill directory; its basename is the skill name (skill_loader requires
# name == parent dir). Source of truth for what onboard installs — keep in
# sync with the actual dirs (a guard test asserts each exists with a SKILL.md).
BUNDLED_SKILLS = (
    ".claude/skills/deployment-awareness",
    ".claude/skills/self-debug",
    ".claude/skills/trace-analysis",
)


def _repo_root() -> Path | None:
    """Locate the source checkout that holds the bundled skills.

    onboard ships inside the ``agentm`` package; under an editable install
    (``uv sync``) ``__file__`` lives in ``<repo>/src/agentm/``, so walk up
    until a parent holds both ``pyproject.toml`` and ``.claude/skills``.
    Returns None for a pip-wheel install where the source tree isn't on disk.
    """
    for parent in Path(__file__).parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / ".claude" / "skills"
        ).is_dir():
            return parent
    return None


def _packaged_skills_root() -> Path | None:
    """Locate skills bundled inside the installed ``agentm`` package."""
    try:
        root = resources.files("agentm.skills")
    except (ModuleNotFoundError, TypeError):
        return None
    try:
        concrete = Path(os.fspath(root))  # type: ignore[call-overload]
    except TypeError:
        return None
    return concrete if concrete.is_dir() else None


def _bundled_skill_sources() -> list[Path]:
    root = _repo_root()
    if root is not None:
        return [
            src
            for rel in BUNDLED_SKILLS
            if (src := root / rel).is_dir() and (src / "SKILL.md").is_file()
        ]
    packaged = _packaged_skills_root()
    if packaged is None:
        return []
    return [
        src
        for src in sorted(packaged.iterdir())
        if src.is_dir() and (src / "SKILL.md").is_file()
    ]


def _skills_dir() -> Path:
    """The ``~/.agentm/skills`` dir skill_loader treats as the user default."""
    return agentm_home_dir() / "skills"


def install_bundled_skills(*, overwrite: bool = False) -> list[Path]:
    """Copy curated repo skills into ``~/.agentm/skills/<name>``.

    Idempotent: an existing ``<name>`` dir is kept intact unless *overwrite*.
    Returns the destination dirs actually written. Returns ``[]`` (the caller
    reports it) when the source checkout can't be located.
    """
    sources = _bundled_skill_sources()
    if not sources:
        return []
    dest_root = _skills_dir()
    dest_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for src in sources:
        dest = dest_root / src.name
        if dest.exists():
            if not overwrite:
                continue
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        written.append(dest)
    return written


# --------------------------------------------------------------------------
# Section: systemd services
# --------------------------------------------------------------------------


def install_systemd_services(workspace: Path) -> int:
    """Install the gateway + feishu systemd units for *workspace*.

    Reuses the single mechanism in ``agentm gateway --install-systemd`` by
    invoking it as a subprocess with ``--cwd <workspace>``: that bakes
    ``--cwd`` into the gateway ExecStart and points both units'
    ``EnvironmentFile`` at ``<workspace>/.env`` (where onboard wrote the
    Feishu + model credentials). Keeping one code path means onboard and the
    direct CLI invocation render identical units. Returns the subprocess exit
    code.
    """
    import subprocess

    agentm_bin = shutil.which("agentm")
    cmd = (
        [agentm_bin] if agentm_bin else ["uv", "run", "agentm"]
    ) + ["gateway", "--cwd", str(workspace), "--install-systemd"]
    typer.echo(f"  running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        typer.echo(
            f"  systemd setup exited with code {result.returncode}; "
            "you can re-run `agentm gateway --install-systemd` later."
        )
    return result.returncode


# --------------------------------------------------------------------------
# Non-interactive setup / diagnostics
# --------------------------------------------------------------------------


def _count_manifest_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        path.parent.name
        for path in root.glob("*/manifest.yaml")
        if path.is_file()
    )


def _list_installed_skills() -> list[str]:
    root = _skills_dir()
    if not root.is_dir():
        return []
    return sorted(
        path.parent.name
        for path in root.glob("*/SKILL.md")
        if path.is_file()
    )


def _print_status(workspace: Path) -> None:
    config_path = _config_path()
    data = _read_toml(config_path)
    default_model = data.get("default_model")
    models = data.get("models")
    typer.echo("AgentM setup status")
    typer.echo(f"  home:      {agentm_home_dir()}")
    typer.echo(f"  config:    {config_path} ({'ok' if config_path.is_file() else 'missing'})")
    if isinstance(default_model, str) and isinstance(models, dict):
        raw_profile = models.get(default_model)
        if isinstance(raw_profile, dict):
            provider = raw_profile.get("provider", "?")
            model = raw_profile.get("model", "?")
            key = _redact_secret(raw_profile.get("api_key"))
            typer.echo(
                f"  model:     {default_model} -> {provider}/{model} "
                f"(api_key={key})"
            )
        else:
            typer.echo(f"  model:     default_model={default_model!r} (profile missing)")
    else:
        typer.echo("  model:     missing")

    typer.echo(f"  workspace: {workspace}")
    persona = [name for name in ("SOUL.md", "IDENTITY.md", "USER.md") if (workspace / name).is_file()]
    typer.echo(f"  persona:   {', '.join(persona) if persona else 'missing'}")
    skills = _list_installed_skills()
    typer.echo(f"  skills:    {len(skills)} installed" + (f" ({', '.join(skills)})" if skills else ""))
    contrib = agentm_home_dir() / "contrib"
    scenarios = _count_manifest_dirs(contrib / "scenarios")
    extensions_dir = contrib / "extensions"
    typer.echo(
        "  contrib:   "
        f"{len(scenarios)} scenario(s)"
        + (f" ({', '.join(scenarios)})" if scenarios else "")
        + f"; extensions={'yes' if extensions_dir.exists() else 'no'}"
    )
    bins = ["agentm", "agentm-terminal", "agentm-feishu", "agentm-weixin"]
    found = [name for name in bins if shutil.which(name)]
    typer.echo(f"  binaries:  {', '.join(found) if found else 'agentm only / PATH not set'}")


async def _test_model_request(
    *,
    workspace: Path,
    model_name: str | None,
    prompt: str,
) -> int:
    """Run one real AgentM turn using the resolved config.toml provider."""
    from agentm.core.abi import EventBus, LoopConfig
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.lib.render import final_summary
    from agentm.core.lib.user_config import resolve_provider_model
    from agentm.core.runtime.resource_loader import DefaultResourceLoader
    from agentm.core.runtime.session import AgentSession
    from agentm.env import autoload_dotenv

    autoload_dotenv(workspace)
    provider, resolved_model, profile = resolve_provider_model(
        model_flag=model_name,
        registry=DEFAULT_PROVIDER_REGISTRY,
    )
    build_config = profile.to_build_config() if profile is not None else {"model": resolved_model}
    provider_spec = DEFAULT_PROVIDER_REGISTRY.build(provider, build_config)
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(workspace),
            provider=provider_spec,
            scenario="chatbot",
            resource_loader=DefaultResourceLoader(cwd=workspace),
            loop_config=LoopConfig(max_turns=1),
            bus=EventBus(),
            auto_commit=False,
        )
    )
    try:
        await session.prompt(prompt)
        await session.idle(timeout=30)
        report = final_summary(session.session_manager.get_messages())
        typer.echo("\n== Model test response ==")
        typer.echo(report.text.strip() or "(empty assistant response)")
        typer.echo(
            f"  usage: input={report.usage.input_tokens}, "
            f"output={report.usage.output_tokens}"
        )
    finally:
        await session.shutdown()
    return 0


def run_setup(
    *,
    profile: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    context_window: int | None = None,
    reasoning_effort: str | None = None,
    workspace: Path | None = None,
    bot_name: str = "Assistant",
    voice: str = "",
    quick: bool = False,
    check: bool = False,
    test_model: bool = False,
    test_prompt: str = "Reply with exactly: agentm-ok",
    sync_contrib_resources: bool = True,
    install_skills: bool = True,
    seed_persona_files: bool = True,
    force_model: bool = False,
) -> int:
    """Fast setup path for tool-installed AgentM.

    The default is intentionally low-friction: reuse an existing
    ``config.toml`` when present; otherwise infer provider/model/key from env
    and prompt only for missing credentials. ``quick`` makes the flow
    non-interactive for scripts.
    """

    workspace_path = (workspace or Path.cwd()).expanduser()
    if check:
        _print_status(workspace_path)
        if test_model:
            return asyncio.run(
                _test_model_request(
                    workspace=workspace_path,
                    model_name=model,
                    prompt=test_prompt,
                )
            )
        return 0

    typer.echo("AgentM setup")

    has_existing_model = _has_existing_default_model()
    explicit_model_input = any(
        value is not None
        for value in (profile, provider, api_key, base_url, context_window, reasoning_effort)
    ) or (model is not None and (force_model or not has_existing_model))
    config_path: Path | None = None
    if has_existing_model and not force_model and not explicit_model_input:
        typer.echo(f"  model:     kept existing {_config_path()}")
    else:
        inferred = _infer_model_setup(
            profile=profile,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        key = inferred.api_key
        if not key and not quick:
            key = typer.prompt(
                f"API key for {inferred.provider}",
                hide_input=True,
            )
        if not key:
            typer.echo(
                "  model:     missing API key. Re-run with --api-key, set the "
                "provider API key env var, or omit --quick to enter it.",
                err=True,
            )
            return 2
        config_path = configure_model(
            profile=inferred.profile,
            provider=inferred.provider,
            model=inferred.model,
            api_key=key,
            base_url=inferred.base_url,
            context_window=context_window,
            reasoning_effort=reasoning_effort,
        )
        typer.echo(
            f"  model:     wrote {config_path} "
            f"({inferred.profile} -> {inferred.provider}/{inferred.model})"
        )

    workspace_path = ensure_workspace(workspace_path)

    persona_paths: list[Path] = []
    if seed_persona_files:
        persona_paths = seed_persona(workspace_path, name=bot_name, voice=voice)
        if persona_paths:
            typer.echo(
                "  persona:   wrote "
                + ", ".join(path.name for path in persona_paths)
            )
        else:
            typer.echo("  persona:   already present")

    skills_written: list[Path] = []
    if install_skills:
        skills_written = install_bundled_skills(overwrite=False)
        if skills_written:
            typer.echo(
                "  skills:    installed "
                + ", ".join(path.name for path in skills_written)
            )
        else:
            typer.echo("  skills:    already present or unavailable")

    if sync_contrib_resources:
        from agentm.contrib_sync import SyncMode, sync_contrib

        records = sync_contrib(mode=SyncMode.copy, overwrite=False)
        summary = ", ".join(f"{r.kind}:{r.action}" for r in records)
        typer.echo(f"  contrib:   {summary}")

    typer.echo(f"  workspace: {workspace_path}")
    typer.echo("\nNext commands:")
    typer.echo(f'  agentm --cwd "{workspace_path}" -p "Say hi"')
    typer.echo(f'  agentm setup --workspace "{workspace_path}" --check')
    typer.echo(f'  agentm gateway --cwd "{workspace_path}" --bind unix:///tmp/agentm-gw.sock')
    typer.echo("  agentm trace messages --latest")

    if test_model:
        return asyncio.run(
            _test_model_request(
                workspace=workspace_path,
                model_name=model,
                prompt=test_prompt,
            )
        )
    _ = config_path
    return 0


# --------------------------------------------------------------------------
# Interactive driver
# --------------------------------------------------------------------------


def run_onboard() -> None:
    """Interactive question-and-answer flow. Called by the CLI command."""
    typer.echo("AgentM onboarding — bootstrap a fresh install.\n")

    # --- Section 1: model / provider / key ---
    typer.echo("== 1. Model / provider / API key ==")
    existing = _read_toml(_config_path())
    existing_models = existing.get("models")
    existing_models = existing_models if isinstance(existing_models, dict) else {}

    profile = typer.prompt("Profile name", default="doubao")
    do_model = True
    if profile in existing_models:
        do_model = typer.confirm(
            f"Profile [models.{profile}] already exists. Overwrite?",
            default=False,
        )

    config_path: Path | None = None
    if do_model:
        provider = typer.prompt("Provider (e.g. openai / anthropic)", default="openai")
        model = typer.prompt("Model id")
        base_url = typer.prompt("Base URL (optional)", default="", show_default=False)
        api_key = typer.prompt("API key", hide_input=True)
        cw_raw = typer.prompt(
            "Context window (optional int)", default="", show_default=False
        )
        effort = typer.prompt(
            "Reasoning effort (optional)", default="", show_default=False
        )
        context_window: int | None = None
        if cw_raw.strip():
            try:
                context_window = int(cw_raw.strip())
            except ValueError:
                typer.echo("  (not an integer; skipping context_window)")
        config_path = configure_model(
            profile=profile,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url.strip() or None,
            context_window=context_window,
            reasoning_effort=effort.strip() or None,
        )
        typer.echo(f"  wrote {config_path}")
    else:
        typer.echo("  kept existing profile.")

    # --- Section 2: workspace ---
    typer.echo("\n== 2. Workspace ==")
    ws_default = str(Path.cwd())
    ws_raw = typer.prompt("Workspace path", default=ws_default)
    workspace = ensure_workspace(Path(ws_raw))
    typer.echo(f"  workspace ready at {workspace}")

    # --- Section 3: persona ---
    typer.echo("\n== 3. Persona ==")
    persona_paths: list[Path] = []
    soul = workspace / "SOUL.md"
    identity = workspace / "IDENTITY.md"
    do_persona = True
    if soul.exists() or identity.exists():
        do_persona = typer.confirm(
            "SOUL.md / IDENTITY.md already exist. Overwrite/seed missing?",
            default=False,
        )
    if do_persona:
        bot_name = typer.prompt("Bot name", default="Assistant")
        voice = typer.prompt(
            "Voice / tone (short description)", default="", show_default=False
        )
        # Overwrite when the user opted in: remove so seed_persona rewrites.
        if soul.exists():
            soul.unlink()
        if identity.exists():
            identity.unlink()
        persona_paths = seed_persona(workspace, name=bot_name, voice=voice)
        for p in persona_paths:
            typer.echo(f"  wrote {p}")
    else:
        typer.echo("  kept existing persona files.")

    # --- Section 4: bundled skills ---
    typer.echo("\n== 4. Skills ==")
    skills_written: list[Path] = []
    if not _bundled_skill_sources():
        typer.echo(
            "  bundled skills not found — skipping skill copy. Drop "
            "SKILL.md dirs under "
            f"{_skills_dir()} manually if you want them."
        )
    elif typer.confirm(
        f"Copy {len(BUNDLED_SKILLS)} self-debug skills into ~/.agentm/skills "
        "so they're discoverable everywhere?",
        default=True,
    ):
        dest_root = _skills_dir()
        present = [r for r in BUNDLED_SKILLS if (dest_root / Path(r).name).exists()]
        overwrite = False
        if present:
            overwrite = typer.confirm(
                f"{len(present)} already present; overwrite them?", default=False
            )
        skills_written = install_bundled_skills(overwrite=overwrite)
        for p in skills_written:
            typer.echo(f"  copied {p.name} -> {p}")
        if not skills_written:
            typer.echo("  nothing new copied (all present, or none found).")
    else:
        typer.echo("  skipped.")

    # --- Section 5: Feishu (optional) ---
    typer.echo("\n== 5. Feishu / Lark bot (optional) ==")
    env_path: Path | None = None
    if typer.confirm("Set up a Feishu/Lark bot?", default=False):
        existing_env = _parse_env(workspace / ".env")
        do_feishu = True
        if "LARK_APP_ID" in existing_env:
            do_feishu = _prompt_keep_or_overwrite(
                "LARK_APP_ID", existing_env["LARK_APP_ID"]
            )
        if do_feishu:
            app_id = typer.prompt("LARK_APP_ID")
            app_secret = typer.prompt("LARK_APP_SECRET", hide_input=True)
            allow_from = typer.prompt("LARK_ALLOW_FROM", default="*")
            env_path = configure_feishu(
                workspace,
                app_id=app_id,
                app_secret=app_secret,
                allow_from=allow_from,
            )
            typer.echo(f"  wrote {env_path}")
            typer.echo("  reminder: ensure .env is gitignored.")
        else:
            typer.echo("  kept existing Feishu credentials.")
    else:
        typer.echo("  skipped.")

    # --- Section 6: systemd services (optional) ---
    typer.echo("\n== 6. systemd services (optional) ==")
    if typer.confirm(
        "Set up systemd services now (auto-restart + start on boot)?",
        default=False,
    ):
        install_systemd_services(workspace)
    else:
        typer.echo("  skipped.")

    # --- Final summary ---
    typer.echo("\n== Summary ==")
    if config_path is not None:
        typer.echo(f"  config:    {config_path}  (default_model = {profile!r})")
    if persona_paths:
        typer.echo(
            f"  persona:   {', '.join(p.name for p in persona_paths)} in {workspace}"
        )
    if skills_written:
        typer.echo(
            f"  skills:    {', '.join(p.name for p in skills_written)} "
            f"-> {_skills_dir()}"
        )
    if env_path is not None:
        typer.echo(f"  feishu:    {env_path}")
    typer.echo(f"  workspace: {workspace}")
    typer.echo(
        "\nNext: run `agentm gateway` from the workspace, or install the "
        "managed services with `agentm gateway --install-systemd` "
        "(see contrib/gateway-peers/deploy/README.md)."
    )
