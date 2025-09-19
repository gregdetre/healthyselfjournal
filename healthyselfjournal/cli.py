from __future__ import annotations

import importlib
import sys
from pathlib import Path

import typer
from rich.console import Console

from .config import CONFIG
from .cli_init import init as init_cmd
from .cli_reconcile import reconcile as reconcile_cmd
from .cli_summaries import build_app as build_summaries_app
from .cli_journal import build_app as build_journal_app
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
        # Skip heavy runtime deps for non-interactive subcommands like `journal list`
        argv = sys.argv[1:]
        if "journal" in argv:
            try:
                idx = argv.index("journal")
                if idx + 1 < len(argv) and argv[idx + 1] == "list":
                    return
            except Exception:
                pass
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
                "  source /Users/greg/.venvs/experim__healthyselfjournal/bin/activate"
            )
            console.print("  uv sync --active")
            console.print()
            console.print("Or run without activating the venv using uv:")
            console.print("  uv run --active healthyselfjournal journal")
            raise typer.Exit(code=3)


# Run dependency verification before executing any subcommand
@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    # When help/version only, Typer may not set invoked_subcommand
    sub = ctx.invoked_subcommand or ""
    if sub:
        _verify_runtime_deps_for_command(sub)


# Sub-apps
summaries_app = build_summaries_app()
journal_app = build_journal_app()
app.add_typer(summaries_app, name="summaries")
app.add_typer(journal_app, name="journal")

# Top-level commands
app.command()(reconcile_cmd)
app.command()(init_cmd)
app.command()(merge_cmd)


@app.command()
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
    """Launch the FastHTML-powered web interface (imports only when invoked)."""

    # Lazy import to avoid importing FastHTML at CLI startup
    from .web.app import WebAppConfig, run_app

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


@app.command()
def legacy_transcribe() -> None:
    """Temporary bridge to the legacy ffmpeg-based transcription CLI."""
    console.print(
        "[yellow]The legacy transcription workflow has moved to `legacy_transcribe_cli.py`."
    )
    console.print(
        "Run `python legacy_transcribe_cli.py --help` for the previous ffmpeg interface."
    )
