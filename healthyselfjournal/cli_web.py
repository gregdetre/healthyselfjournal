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
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume the most recent session instead of starting a new one.",
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
    kill_existing: bool = typer.Option(
        False,
        "--kill-existing",
        help="If set, attempt to free the port by killing existing listeners before start.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Open the default browser to the server URL when ready.",
    ),
) -> None:
    """Launch the FastHTML-powered web interface."""

    config = WebAppConfig(
        sessions_dir=sessions_dir,
        resume=resume,
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

    # Optionally free the port before starting
    if kill_existing:
        try:
            from gjdutils.ports import free_port_if_in_use

            free_port_if_in_use(port, verbose=1)
            console.print(f"[yellow]Ensured port {port} is free before startup.[/]")
        except Exception:
            # Best-effort; continue to let server start or show usual bind error
            pass

    # Optionally open the user's browser once the server is ready to accept connections
    if open_browser:

        def _open_when_ready(host: str, port: int) -> None:
            import socket
            import time
            import webbrowser

            # Map wildcard hosts to a sensible local address for browsers
            browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
            # Bracket IPv6 literal for URLs
            display_host = (
                f"[{browser_host}]"
                if ":" in browser_host and not browser_host.startswith("[")
                else browser_host
            )
            url = f"http://{display_host}:{port}/"

            console.print(
                f"[cyan]Will open browser at {url} when the server is ready...[/]"
            )

            deadline = time.time() + 60.0
            # Poll for TCP accept on any resolved address
            while time.time() < deadline:
                try:
                    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
                except Exception:
                    infos = []
                connected = False
                for family, socktype, proto, _canon, sockaddr in infos:
                    s = None
                    try:
                        s = socket.socket(family, socktype, proto)
                        s.settimeout(0.5)
                        s.connect(sockaddr)
                        connected = True
                        break
                    except Exception:
                        pass
                    finally:
                        try:
                            s and s.close()
                        except Exception:
                            pass
                if connected:
                    # Give the server a brief moment before launching the browser
                    time.sleep(0.2)
                    try:
                        webbrowser.open(url, new=2)
                        console.print(f"[green]Opened browser:[/] {url}")
                    except Exception:
                        console.print(f"[yellow]Please open your browser at:[/] {url}")
                    return
                time.sleep(0.25)
            # Timed out waiting; print the URL for manual navigation
            console.print(
                f"[yellow]Server didn't become ready within 60s. Open:[/] {url}"
            )

        import threading

        t = threading.Thread(
            target=_open_when_ready,
            args=(host, port),
            name="hsj-open-browser",
            daemon=True,
        )
        t.start()

    try:
        run_app(config)
    except KeyboardInterrupt:  # pragma: no cover - direct CLI interrupt
        console.print("\n[cyan]Server stopped.[/]")
