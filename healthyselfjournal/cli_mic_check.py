from __future__ import annotations

import os

import typer
from rich.console import Console

from .config import CONFIG
from .transcription import resolve_backend_selection
from .mic_check import run_interactive_mic_check


console = Console()


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
        if not os.environ.get("OPENAI_API_KEY"):
            console.print("[red]OPENAI_API_KEY is required for cloud STT.[/]")
            raise typer.Exit(code=2)

    run_interactive_mic_check(
        selection,
        console=console,
        language=language,
        stt_formatting=stt_formatting,
        seconds=seconds,
        sample_rate=sample_rate,
    )


