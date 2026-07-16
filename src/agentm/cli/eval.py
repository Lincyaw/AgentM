"""Lazy bridge to the optional ``agentm_eval`` command tree."""

from __future__ import annotations

import typer

_MISSING_EVAL_EXIT = 7


def register_eval_command(app: typer.Typer) -> None:
    """Register ``agentm eval`` without importing every benchmark at startup."""

    @app.command(
        name="eval",
        context_settings={
            "allow_extra_args": True,
            "ignore_unknown_options": True,
            "help_option_names": [],
        },
        help="Run and inspect optional benchmark evaluations.",
    )
    def eval_cmd(ctx: typer.Context) -> None:
        try:
            from agentm_eval.cli import app as eval_app
        except ImportError as exc:
            typer.echo(
                "agentm eval is unavailable: install optional eval support "
                "with `uv sync --extra eval` (workspace) or "
                f"`pip install 'agentm[eval]'` ({exc}).",
                err=True,
            )
            raise typer.Exit(_MISSING_EVAL_EXIT) from exc

        eval_app(
            args=list(ctx.args),
            prog_name="agentm eval",
        )


__all__ = ["register_eval_command"]
