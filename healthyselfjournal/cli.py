from __future__ import annotations

import importlib
import sys
from pathlib import Path

import typer
from typer.core import TyperGroup
from rich.console import Console
from rich.markdown import Markdown
import webbrowser
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
    help=(
        f"HealthySelfJournal {__version__}\n\n"
        "Voice-first journaling CLI.\n\n"
        "Quickstart: run 'healthyselfjournal init' then 'healthyselfjournal journal' (defaults to CLI)."
    ),
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


@app.command("readme")
def readme(
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open README in your default browser instead of printing to the terminal.",
    )
) -> None:
    """Show the project README or open it in a browser."""

    # Try to locate a local README first (package root, then CWD)
    package_root = Path(__file__).resolve().parents[1]
    candidates = [package_root / "README.md", Path.cwd() / "README.md"]
    readme_path = next((p for p in candidates if p.exists()), None)

    # Fallback URL if no local README is present (wheel/sdist environments)
    fallback_url = "https://pypi.org/project/healthyselfjournal/"
    url: str

    if readme_path is not None:
        url = readme_path.resolve().as_uri()
    else:
        url = fallback_url

    if open_browser:
        try:
            webbrowser.open(url, new=2)
            console.print(f"[green]Opened:[/] {url}")
        except Exception:
            console.print(f"[yellow]Please open in your browser:[/] {url}")
        return

    console.print(f"[cyan]Open in browser:[/] {url}\n")
    if readme_path is not None:
        try:
            content = readme_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            console.print(f"[red]Failed to read README:[/] {exc}")
            console.print(f"[yellow]You can view it here:[/] {url}")
            return
        console.print(Markdown(content))
    else:
        console.print(
            "README.md not found locally. Use --open to view the online documentation."
        )
    console.print(f"\n[cyan]Open in browser:[/] {url}")
