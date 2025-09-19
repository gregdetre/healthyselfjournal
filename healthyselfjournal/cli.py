from __future__ import annotations

import importlib
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import CONFIG
from .cli_init import init as init_cmd
from .cli_reconcile import reconcile as reconcile_cmd
from .cli_summarise import build_app as build_summaries_app
from .cli_journal_cli import build_app as build_journal_app
from .cli_session import build_app as build_session_app
from .cli_mic_check import mic_check as mic_check_cmd
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
        # Enforce only for `journal cli`; skip for other subcommands like `journal web`
        argv = sys.argv[1:]
        try:
            idx = argv.index("journal")
        except ValueError:
            return
        next_arg = argv[idx + 1] if idx + 1 < len(argv) else None
        if next_arg != "cli":
            return

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
            console.print("[red]Missing required dependencies for 'journal cli':[/]")
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
                "  source /Users/greg/.venvs/experim__healthyselfjournal/bin/activate"
            )
            console.print("  uv sync --active")
            console.print()
            console.print("Or run without activating the venv using uv:")
            console.print("  uv run --active healthyselfjournal journal cli")
            raise typer.Exit(code=3)


# Run dependency verification before executing any subcommand
@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    # When help/version only, Typer may not set invoked_subcommand
    sub = ctx.invoked_subcommand or ""
    if sub:
        _verify_runtime_deps_for_command(sub)


# Sub-apps
j_summaries_app = build_summaries_app()
journal_app = build_journal_app()
app.add_typer(j_summaries_app, name="summarise")
app.add_typer(journal_app, name="journal")

# Top-level commands
app.command()(reconcile_cmd)
app.command()(init_cmd)
app.command()(merge_cmd)

try:
    from .cli_desktop import desktop as desktop_command

    app.command()(desktop_command)
except Exception:
    # Desktop shell requires optional dependencies (pywebview).
    pass

# Session utilities group (moved out to cli_session.py)
app.add_typer(build_session_app(), name="session")


app.command()(mic_check_cmd)


@app.command()
def legacy_transcribe() -> None:
    """Temporary bridge to the legacy ffmpeg-based transcription CLI."""
    console.print(
        "[yellow]The legacy transcription workflow has moved to `legacy_transcribe_cli.py`."
    )
    console.print(
        "Run `python legacy_transcribe_cli.py --help` for the previous ffmpeg interface."
    )
