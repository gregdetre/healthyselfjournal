from __future__ import annotations

"""Typer command that launches the FastHTML web server."""

from pathlib import Path

import typer
from rich.console import Console

from .config import CONFIG
from .web.app import WebAppConfig, run_app


console = Console()


def web(
    sessions_dir: Path = typer.Option(
        CONFIG.recordings_dir,
        "--sessions-dir",
        help="Directory where session markdown and audio artifacts are stored.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Interface to bind the development server to.",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        help="Port to serve the web interface on.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload/--no-reload",
        help="Enable FastHTML/uvicorn autoreload (development only).",
    ),
    voice_mode: bool = typer.Option(
        CONFIG.speak_llm,
        "--voice-mode/--no-voice-mode",
        help="Speak assistant questions in the browser using server-side TTS.",
    ),
    tts_model: str = typer.Option(
        CONFIG.tts_model,
        "--tts-model",
        help="TTS model identifier (server-side synthesis).",
    ),
    tts_voice: str = typer.Option(
        CONFIG.tts_voice,
        "--tts-voice",
        help="TTS voice name (server-side synthesis).",
    ),
    tts_format: str = typer.Option(
        CONFIG.tts_format,
        "--tts-format",
        help="TTS audio format returned to the browser (e.g., wav, mp3).",
    ),
) -> None:
    """Launch the FastHTML-powered web interface."""

    config = WebAppConfig(
        sessions_dir=sessions_dir,
        host=host,
        port=port,
        reload=reload,
        voice_enabled=voice_mode,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_format=tts_format,
    )
    console.print(
        f"[green]Starting Healthy Self Journal web server on {host}:{port}[/]"
    )
    console.print(f"Sessions directory: [cyan]{config.sessions_dir.expanduser()}[/]")

    try:
        run_app(config)
    except KeyboardInterrupt:  # pragma: no cover - direct CLI interrupt
        console.print("\n[cyan]Server stopped.[/]")
