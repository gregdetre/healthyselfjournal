from __future__ import annotations

"""Typer command for launching the PyWebView desktop shell."""

from pathlib import Path

import typer
from rich.console import Console

from .config import CONFIG

console = Console()


def desktop(
    sessions_dir: Path = typer.Option(
        CONFIG.recordings_dir,
        "--sessions-dir",
        help="Directory where session markdown/audio artifacts are stored.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume the most recent session instead of starting a new one.",
    ),
    port: int = typer.Option(
        0,
        "--port",
        help="Port for the embedded FastHTML server (0 = auto).",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address for the embedded FastHTML server.",
    ),
    voice_mode: bool = typer.Option(
        CONFIG.speak_llm,
        "--voice-mode/--no-voice-mode",
        help="Speak assistant questions in the desktop app using server-side TTS.",
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
        help="TTS audio format returned to the embedded web app (e.g., wav, mp3).",
    ),
    window_title: str = typer.Option(
        "Healthy Self Journal",
        "--title",
        help="Window title for the desktop shell.",
    ),
    window_width: int = typer.Option(
        1280,
        "--width",
        help="Initial window width in pixels.",
    ),
    window_height: int = typer.Option(
        860,
        "--height",
        help="Initial window height in pixels.",
    ),
    fullscreen: bool = typer.Option(
        False,
        "--fullscreen/--windowed",
        help="Launch the desktop shell in fullscreen mode.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Enable PyWebView debug mode (prints console logs to stdout).",
    ),
    devtools: bool = typer.Option(
        False,
        "--devtools/--no-devtools",
        help="Open the embedded browser devtools (for development only).",
    ),
    server_start_timeout: float = typer.Option(
        15.0,
        "--server-timeout",
        help="Seconds to wait for the embedded FastHTML server to start before failing.",
    ),
    confirm_close: bool = typer.Option(
        True,
        "--confirm-close/--no-confirm-close",
        help="Ask for confirmation before closing the window.",
    ),
) -> None:
    """Launch the Healthy Self Journal desktop experience."""

    from .desktop import DesktopConfig, run_desktop_app
    from .web.app import WebAppConfig

    web_cfg = WebAppConfig(
        sessions_dir=sessions_dir,
        resume=resume,
        host=host,
        port=port,
        reload=False,
        voice_enabled=voice_mode,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_format=tts_format,
    )
    desktop_cfg = DesktopConfig(
        web=web_cfg,
        window_title=window_title,
        window_width=window_width,
        window_height=window_height,
        fullscreen=fullscreen,
        debug=debug,
        open_devtools=devtools,
        server_start_timeout=server_start_timeout,
        confirm_close=confirm_close,
    )

    console.print("[green]Launching Healthy Self Journal desktop shell…[/]")
    console.print(f"Sessions directory: [cyan]{sessions_dir.expanduser()}[/]")
    console.print(f"Embedded server: [cyan]{web_cfg.host}:{web_cfg.port or 'auto'}[/]")

    try:
        run_desktop_app(desktop_cfg)
    except KeyboardInterrupt:  # pragma: no cover - direct user interrupt
        console.print("\n[yellow]Desktop session interrupted by user.[/]")
    except Exception as exc:
        console.print(f"[red]Failed to launch desktop app:[/] {exc}")
        raise typer.Exit(code=1) from exc
