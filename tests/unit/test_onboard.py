"""Fail-stop tests for ``agentm onboard``.

The onboarding command's load-bearing contract is that what it writes is read
back correctly by the runtime: the model profile must round-trip through
``load_user_config()``, the Feishu keys must land in ``.env`` under the exact
``LARK_*`` names (an ``ARK_APP_ID`` typo once broke a real deployment), and the
persona files must carry the chosen bot name. These assert observable file
output, not the interactive prompting.
"""

from __future__ import annotations

from pathlib import Path

from agentm.core.lib import user_config
from agentm.core.lib.frontmatter import parse_frontmatter
from agentm.onboard import (
    BUNDLED_SKILLS,
    _repo_root,
    configure_feishu,
    configure_model,
    ensure_workspace,
    install_bundled_skills,
    seed_persona,
)


def test_model_profile_round_trips_through_load_user_config(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    user_config._reset_cache()

    path = configure_model(
        profile="doubao",
        provider="openai",
        model="doubao-seed-2-0-pro-260215",
        api_key="sk-secret-123",
        base_url="https://ark.example/api/v3",
        context_window=131072,
        reasoning_effort="high",
    )
    assert path == tmp_path / "config.toml"
    # Secret stored 0600.
    assert (path.stat().st_mode & 0o777) == 0o600

    user_config._reset_cache()
    cfg = user_config.load_user_config()
    assert cfg.default_model == "doubao"
    profile = cfg.models["doubao"]
    assert profile.provider == "openai"
    assert profile.model == "doubao-seed-2-0-pro-260215"
    assert profile.api_key == "sk-secret-123"
    assert profile.base_url == "https://ark.example/api/v3"
    assert profile.context_window == 131072
    assert profile.reasoning_effort == "high"


def test_configure_model_merges_without_dropping_existing_profiles(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    user_config._reset_cache()

    configure_model(
        profile="first", provider="openai", model="m-1", api_key="k1"
    )
    configure_model(
        profile="second", provider="anthropic", model="m-2", api_key="k2"
    )

    user_config._reset_cache()
    cfg = user_config.load_user_config()
    assert cfg.default_model == "second"
    assert set(cfg.models) == {"first", "second"}
    assert cfg.models["first"].model == "m-1"
    assert cfg.models["second"].provider == "anthropic"


def test_feishu_writes_exact_lark_keys_and_preserves_others(tmp_path: Path) -> None:
    workspace = ensure_workspace(tmp_path / "ws")
    (workspace / ".env").write_text("EXISTING_KEY=keepme\n", encoding="utf-8")

    path = configure_feishu(
        workspace,
        app_id="cli_app_id_xyz",
        app_secret="super-secret",
        allow_from="*",
    )
    assert path == workspace / ".env"
    assert (path.stat().st_mode & 0o777) == 0o600

    env = dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    assert env["LARK_APP_ID"] == "cli_app_id_xyz"
    assert env["LARK_APP_SECRET"] == "super-secret"
    assert env["LARK_ALLOW_FROM"] == "*"
    assert env["EXISTING_KEY"] == "keepme"
    # Guard against the ARK_APP_ID typo class.
    assert "ARK_APP_ID" not in env


def test_seed_persona_writes_files_with_bot_name(tmp_path: Path) -> None:
    workspace = ensure_workspace(tmp_path / "ws")
    written = seed_persona(workspace, name="Mira", voice="Playful and brisk")

    names = {p.name for p in written}
    assert {"SOUL.md", "IDENTITY.md", "USER.md"} <= names

    identity = (workspace / "IDENTITY.md").read_text(encoding="utf-8")
    soul = (workspace / "SOUL.md").read_text(encoding="utf-8")
    assert "Mira" in identity
    assert "Playful and brisk" in soul


def test_seed_persona_does_not_clobber_existing(tmp_path: Path) -> None:
    workspace = ensure_workspace(tmp_path / "ws")
    (workspace / "SOUL.md").write_text("# custom soul\n", encoding="utf-8")

    written = seed_persona(workspace, name="Mira", voice="")

    assert workspace / "SOUL.md" not in written
    assert (workspace / "SOUL.md").read_text(encoding="utf-8") == "# custom soul\n"


# --------------------------------------------------------------------------
# Bundled skills
# --------------------------------------------------------------------------


def _make_fake_repo(root: Path) -> None:
    """Create a minimal fake repo tree containing each BUNDLED_SKILLS entry."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
    for rel in BUNDLED_SKILLS:
        skill_dir = root / rel
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_name = Path(rel).name
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_name}\ndescription: test\n---\n# {skill_name}\n",
            encoding="utf-8",
        )


def test_install_bundled_skills_copies_all(tmp_path: Path, monkeypatch) -> None:
    fake_repo = tmp_path / "repo"
    fake_home = tmp_path / "home"
    _make_fake_repo(fake_repo)

    import agentm.onboard as onboard_mod

    monkeypatch.setattr(onboard_mod, "_repo_root", lambda: fake_repo)
    monkeypatch.setattr(onboard_mod, "agentm_home_dir", lambda: fake_home)

    written = install_bundled_skills()

    assert len(written) == len(BUNDLED_SKILLS)
    for rel in BUNDLED_SKILLS:
        dest = fake_home / "skills" / Path(rel).name / "SKILL.md"
        assert dest.is_file(), f"expected {dest} to exist"


def test_install_bundled_skills_idempotent(tmp_path: Path, monkeypatch) -> None:
    fake_repo = tmp_path / "repo"
    fake_home = tmp_path / "home"
    _make_fake_repo(fake_repo)

    import agentm.onboard as onboard_mod

    monkeypatch.setattr(onboard_mod, "_repo_root", lambda: fake_repo)
    monkeypatch.setattr(onboard_mod, "agentm_home_dir", lambda: fake_home)

    install_bundled_skills()
    second = install_bundled_skills()  # no overwrite
    assert second == [], "second call without overwrite must return []"


def test_install_bundled_skills_overwrite_refreshes(tmp_path: Path, monkeypatch) -> None:
    fake_repo = tmp_path / "repo"
    fake_home = tmp_path / "home"
    _make_fake_repo(fake_repo)

    import agentm.onboard as onboard_mod

    monkeypatch.setattr(onboard_mod, "_repo_root", lambda: fake_repo)
    monkeypatch.setattr(onboard_mod, "agentm_home_dir", lambda: fake_home)

    install_bundled_skills()

    # Mutate one installed file to verify overwrite replaces it.
    first_skill = fake_home / "skills" / Path(BUNDLED_SKILLS[0]).name / "SKILL.md"
    first_skill.write_text("# stale\n", encoding="utf-8")

    refreshed = install_bundled_skills(overwrite=True)
    assert len(refreshed) == len(BUNDLED_SKILLS)
    # Content should be back to the fake-repo original, not "stale".
    assert "stale" not in first_skill.read_text(encoding="utf-8")


def test_repo_root_finds_real_repo() -> None:
    # Positive assertion: running from an editable install the real checkout
    # is on disk and _repo_root() must find it.
    root = _repo_root()
    assert root is not None, "_repo_root() returned None in editable install"
    assert (root / ".claude" / "skills").is_dir()


def test_bundled_skills_guard(tmp_path: Path) -> None:
    """Fail-stop: every BUNDLED_SKILLS entry must exist with a valid SKILL.md
    and its frontmatter name must match the parent directory name."""
    root = _repo_root()
    assert root is not None, "real repo root must be locatable"
    for rel in BUNDLED_SKILLS:
        skill_md = root / rel / "SKILL.md"
        assert skill_md.is_file(), f"SKILL.md missing for bundled skill {rel!r}"
        meta, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        expected_name = Path(rel).name
        assert meta.get("name") == expected_name, (
            f"{skill_md}: frontmatter name={meta.get('name')!r}, "
            f"expected {expected_name!r}"
        )
