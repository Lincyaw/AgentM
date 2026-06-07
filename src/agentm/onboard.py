"""Interactive bootstrap for a fresh AgentM install (``agentm onboard``).

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

import shutil
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
import typer

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
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / ".claude" / "skills"
        ).is_dir():
            return parent
    return None


def install_bundled_skills(*, overwrite: bool = False) -> list[Path]:
    """Copy curated repo skills into ``~/.agentm/skills/<name>``.

    Idempotent: an existing ``<name>`` dir is kept intact unless *overwrite*.
    Returns the destination dirs actually written. Returns ``[]`` (the caller
    reports it) when the source checkout can't be located.
    """
    root = _repo_root()
    if root is None:
        return []
    dest_root = agentm_home_dir() / "skills"
    dest_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rel in BUNDLED_SKILLS:
        src = root / rel
        if not (src / "SKILL.md").is_file():
            continue
        dest = dest_root / Path(rel).name
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
    import shutil
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
    if _repo_root() is None:
        typer.echo(
            "  source checkout not found (pip-wheel install?) — skipping "
            "bundled-skill copy. Drop SKILL.md dirs under "
            f"{agentm_home_dir() / 'skills'} manually if you want them."
        )
    elif typer.confirm(
        f"Copy {len(BUNDLED_SKILLS)} self-debug skills into ~/.agentm/skills "
        "so they're discoverable everywhere?",
        default=True,
    ):
        dest_root = agentm_home_dir() / "skills"
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
            f"-> {agentm_home_dir() / 'skills'}"
        )
    if env_path is not None:
        typer.echo(f"  feishu:    {env_path}")
    typer.echo(f"  workspace: {workspace}")
    typer.echo(
        "\nNext: run `agentm gateway` from the workspace, or install the "
        "managed services with `agentm gateway --install-systemd` "
        "(see contrib/gateway-peers/deploy/README.md)."
    )
