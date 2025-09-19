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
from .cli_init_app import build_app as build_init_app
from .cli_journal_cli import build_app as build_journal_app
from .cli_session import build_app as build_session_app
from .cli_diagnose import build_app as build_diagnose_app

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
journal_app = build_journal_app()
app.add_typer(journal_app, name="journal")
app.add_typer(build_diagnose_app(), name="diagnose")
app.add_typer(build_init_app(), name="init")

# Top-level commands

# Session utilities group (moved out to cli_session.py)
app.add_typer(build_session_app(), name="session")

# New `fix` group consolidating reconciliation and summary maintenance
fix_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Repair utilities: STT backfill and summary regeneration.",
)


@fix_app.command("stt")
def fix_stt(
    sessions_dir: Path = typer.Option(
        CONFIG.recordings_dir,
        "--sessions-dir",
        help="Directory where session markdown/audio files are stored.",
    ),
    stt_backend: str = typer.Option(
        CONFIG.stt_backend,
        "--stt-backend",
        help=(
            "Transcription backend: cloud-openai, local-mlx, local-faster, "
            "local-whispercpp, or auto-private."
        ),
    ),
    stt_model: str = typer.Option(
        CONFIG.model_stt,
        "--stt-model",
        help="Model preset or identifier for the selected backend.",
    ),
    stt_compute: str = typer.Option(
        CONFIG.stt_compute or "auto",
        "--stt-compute",
        help="Optional compute precision override for local backends (e.g., int8_float16).",
    ),
    language: str = typer.Option(
        "en",
        "--language",
        help="Primary language for transcription.",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Maximum number of recordings to process (0 = no limit).",
    ),
    min_duration: float = typer.Option(
        0.1,
        "--min-duration",
        help="Minimum duration (seconds) to attempt transcription.",
    ),
    too_short_action: str = typer.Option(
        "skip",
        "--too-short",
        help="Action for recordings under thresholds: skip, mark, or delete.",
    ),
) -> None:
    """Alias of `reconcile` under the new `fix` group."""

    # Delegate to reconcile()
    reconcile_cmd(
        sessions_dir=sessions_dir,
        stt_backend=stt_backend,
        stt_model=stt_model,
        stt_compute=stt_compute,
        language=language,
        limit=limit,
        min_duration=min_duration,
        too_short_action=too_short_action,
    )


# Bring in summary maintenance under fix as well
from .cli_summarise import build_app as build_summaries_app  # type: ignore

_summaries_app = build_summaries_app()


@_summaries_app.callback()
def _summaries_banner() -> None:
    pass


# Mount summaries app under fix to avoid relying on Typer internals
fix_app.add_typer(_summaries_app, name="summaries")


app.add_typer(fix_app, name="fix")


# mic-check is now part of the diagnose subcommands
