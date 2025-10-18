from __future__ import annotations

import importlib
import sys
from pathlib import Path

import typer
from typer.core import TyperGroup
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import CONFIG
from .cli_init import init as init_cmd
from .cli_init_app import build_app as build_init_app
from .cli_journal_cli import build_app as build_journal_app
from .cli_session import build_app as build_session_app
from .cli_insights import build_app as build_insights_app
from .cli_diagnose import build_app as build_diagnose_app
from .cli_fix import build_app as build_fix_app
from . import __version__
from .cli_reconcile import reconcile as reconcile_cmd


class _OrderedTopLevelGroup(TyperGroup):
    def list_commands(self, ctx):
        desired = [
            "version",
            "init",
            "diagnose",
            "journal",
            "fix",
            "sessions",
            "insight",
        ]
        names = list(self.commands.keys())
        ordered = [name for name in desired if name in names]
        remaining = [name for name in sorted(names) if name not in set(ordered)]
        return ordered + remaining


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=f"HealthySelfJournal {__version__}\n\nVoice-first journaling CLI.",
    cls=_OrderedTopLevelGroup,
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
            console.print("How to proceed:")
            console.print("- Recommended: run without activating a venv using uvx:")
            console.print("    uvx healthyselfjournal -- journal cli")
            console.print("")
            console.print("- Or use uv with an active venv (no user-specific paths):")
            console.print("    python -m venv .venv && source .venv/bin/activate")
            console.print("    uv sync")
            console.print("    uv run healthyselfjournal journal cli")
            raise typer.Exit(code=3)


# Run dependency verification before executing any subcommand
@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    # When help/version only, Typer may not set invoked_subcommand
    sub = ctx.invoked_subcommand or ""
    if sub:
        _verify_runtime_deps_for_command(sub)


# Sub-apps
journal_app = build_journal_app()
app.add_typer(journal_app, name="journal")
app.add_typer(build_diagnose_app(), name="diagnose")
app.add_typer(build_init_app(), name="init")

# Top-level commands


# Version command
@app.command("version")
def version() -> None:
    """Show installed package version."""
    typer.echo(__version__)


# Session utilities group (moved out to cli_session.py)
app.add_typer(build_session_app(), name="sessions")

app.add_typer(build_fix_app(), name="fix")

# Insight sub-app (v1): list and generate
app.add_typer(build_insights_app(), name="insight")


# mic-check is now part of the diagnose subcommands
