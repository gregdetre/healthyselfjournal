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
from .transcription import resolve_backend_selection
from .mic_check import run_interactive_mic_check
from .cli_reconcile import reconcile as reconcile_cmd
from .cli_summaries import build_app as build_summaries_app
from .cli_journal import build_app as build_journal_app
from .cli_merge import merge as merge_cmd
from .storage import load_transcript

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

# Session utilities group (moved from `journal list` → `session list`)
session_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Session utilities (list, future: new, show, etc.).",
)


@session_app.command("list")
def list_sessions(
    sessions_dir: Path = typer.Option(
        CONFIG.recordings_dir,
        "--sessions-dir",
        help="Directory where session markdown/audio files are stored.",
    ),
    nchars: int | None = typer.Option(
        None,
        "--nchars",
        help="Limit summary snippet to N characters (None = full summary).",
    ),
) -> None:
    """List sessions by filename stem with a summary snippet from frontmatter."""

    markdown_files = sorted((p for p in sessions_dir.glob("*.md")))
    if not markdown_files:
        console.print("[yellow]No session markdown files found.[/]")
        return

    for path in markdown_files:
        try:
            doc = load_transcript(path)
            summary_raw = doc.frontmatter.data.get("summary")
            summary_text = summary_raw if isinstance(summary_raw, str) else ""
            normalized = " ".join(summary_text.split())
            if nchars is not None and nchars > 0:
                snippet = normalized[:nchars]
            else:
                snippet = normalized
            body = Text(snippet) if snippet else Text("(no summary)", style="dim")
            console.print(
                Panel.fit(
                    body,
                    title=path.stem,
                    border_style="cyan",
                )
            )
        except Exception as exc:  # pragma: no cover - defensive surface
            console.print(
                Panel.fit(
                    Text(f"error reading - {exc}", style="red"),
                    title=path.name,
                    border_style="red",
                )
            )


app.add_typer(session_app, name="session")


@app.command()
def mic_check(
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
    stt_formatting: str = typer.Option(
        CONFIG.stt_formatting,
        "--stt-formatting",
        help="Transcript formatting mode: sentences (default) or raw.",
    ),
    language: str = typer.Option(
        "en",
        "--language",
        help="Primary language for transcription.",
    ),
    seconds: float = typer.Option(
        3.0,
        "--seconds",
        min=0.5,
        max=30.0,
        help="Recording duration for mic check.",
    ),
    sample_rate: int = typer.Option(
        16_000,
        "--sample-rate",
        help="Sample rate for recording (Hz).",
    ),
) -> None:
    """Run an interactive microphone check and show the transcript."""

    # Resolve backend selection (auto-private supported)
    try:
        selection = resolve_backend_selection(stt_backend, stt_model, stt_compute)
    except Exception as exc:
        console.print(f"[red]STT configuration error:[/] {exc}")
        raise typer.Exit(code=2)

    # Require OpenAI key when cloud backend chosen
    if selection.backend_id == "cloud-openai":
        try:
            import os

            if not os.environ.get("OPENAI_API_KEY"):
                console.print("[red]OPENAI_API_KEY is required for cloud STT.[/]")
                raise typer.Exit(code=2)
        except typer.Exit:
            raise

    run_interactive_mic_check(
        selection,
        console=console,
        language=language,
        stt_formatting=stt_formatting,
        seconds=seconds,
        sample_rate=sample_rate,
    )


@app.command()
def legacy_transcribe() -> None:
    """Temporary bridge to the legacy ffmpeg-based transcription CLI."""
    console.print(
        "[yellow]The legacy transcription workflow has moved to `legacy_transcribe_cli.py`."
    )
    console.print(
        "Run `python legacy_transcribe_cli.py --help` for the previous ffmpeg interface."
    )
