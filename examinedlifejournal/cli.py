from __future__ import annotations

import importlib
import sys

import typer
from rich.console import Console

from .cli_init import init as init_cmd
from .cli_reconcile import reconcile as reconcile_cmd
from .cli_summaries import build_app as build_summaries_app
from .cli_journal import journal as journal_cmd
from .cli_merge import merge as merge_cmd

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

console = Console()


# Fail-fast dependency check for commands that require optional runtime libs
def _verify_runtime_deps_for_command(command_name: str) -> None:
    # Only enforce for commands that require interactive audio capture
    if command_name == "journal":
        required = [
            ("readchar", "Keyboard input for pause/quit controls"),
            ("sounddevice", "Microphone capture"),
            ("soundfile", "WAV read/write"),
            ("numpy", "Audio level meter / math"),
        ]
        missing: list[tuple[str, str]] = []
        for package, why in required:
            try:
                importlib.import_module(package)
            except Exception as exc:  # pragma: no cover - environment-specific
                missing.append((package, f"{exc.__class__.__name__}: {exc}"))

        if missing:
            console.print("[red]Missing required dependencies for 'journal':[/]")
            for name, detail in missing:
                why = next((w for p, w in required if p == name), "")
                console.print(f"- [bold]{name}[/]: {why} — {detail}")
            console.print()
            console.print(
                "[yellow]This often happens when running in the wrong virtualenv.[/]"
            )
            console.print(f"Python: {sys.executable}")
            console.print("Activate the recommended venv and install deps, then retry:")
            console.print(
                "  source /Users/greg/.venvs/experim__examinedlifejournal/bin/activate"
            )
            console.print("  uv sync --active")
            console.print()
            console.print("Or run without activating the venv using uv:")
            console.print("  uv run --active examinedlifejournal journal")
            raise typer.Exit(code=3)


# Run dependency verification before executing any subcommand
@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    # When help/version only, Typer may not set invoked_subcommand
    sub = ctx.invoked_subcommand or ""
    if sub:
        _verify_runtime_deps_for_command(sub)


# Sub-app for summaries utilities
summaries_app = build_summaries_app()
app.add_typer(summaries_app, name="summaries")

# Top-level commands
app.command()(journal_cmd)
app.command()(reconcile_cmd)
app.command()(init_cmd)
app.command()(merge_cmd)


@app.command()
def legacy_transcribe() -> None:
    """Temporary bridge to the legacy ffmpeg-based transcription CLI."""
    console.print(
        "[yellow]The legacy transcription workflow has moved to `legacy_transcribe_cli.py`."
    )
    console.print(
        "Run `python legacy_transcribe_cli.py --help` for the previous ffmpeg interface."
    )
