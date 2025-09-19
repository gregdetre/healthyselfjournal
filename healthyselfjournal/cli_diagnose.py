from __future__ import annotations

"""Diagnostics CLI: mic, local, cloud.

Subcommands:
- diagnose mic: Interactive microphone + STT check (existing mic check).
- diagnose local: End-to-end local diagnostics (STT selection, optional mic, local LLM probe, privacy-mode TTS check).
- diagnose cloud: Environment/key checks; optional live probes gated by flags.

This runs in a sandbox sessions directory by default to avoid touching real files.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import CONFIG
from .events import init_event_logger, log_event
from .mic_check import run_interactive_mic_check
from .transcription import (
    resolve_backend_selection,
    create_transcription_backend,
    BackendNotAvailableError,
)
from .llm import (
    get_active_llm_model_spec,
    get_model_provider,
    QuestionRequest,
    generate_followup_question,
)
from .model_manager import get_model_manager
from .tts import resolve_tts_options, synthesize_text, TTSError


console = Console()


def _bool_env(value: str | None) -> Optional[bool]:
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_init_events(sessions_dir: Optional[Path], sandbox: bool) -> Path:
    if sessions_dir is None:
        base = (
            Path(tempfile.mkdtemp(prefix="hsj_diag_"))
            if sandbox
            else CONFIG.recordings_dir
        )
    else:
        base = sessions_dir
        if sandbox:
            base = base / ".diagnostics"
    init_event_logger(base)
    return base


def _print_header(title: str) -> None:
    console.print(Panel.fit(title, border_style="cyan", title="Diagnostics"))


def _print_config_snapshot() -> None:
    table = Table(title="Effective configuration (snapshot)")
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="magenta")

    table.add_row("sessions_dir (default)", str(CONFIG.recordings_dir))
    table.add_row("llm_mode", str(CONFIG.llm_mode))
    table.add_row("llm_cloud_off", str(CONFIG.llm_cloud_off))
    table.add_row("llm_local_model", str(CONFIG.llm_local_model))
    table.add_row("tts_enabled", str(CONFIG.tts_enabled))
    table.add_row("stt_backend (default)", str(CONFIG.stt_backend))
    table.add_row("stt_model (default)", str(CONFIG.model_stt))
    table.add_row("stt_compute (default)", str(CONFIG.stt_compute))
    console.print(table)


def build_app() -> typer.Typer:
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        context_settings={"help_option_names": ["-h", "--help"]},
        help="Diagnostics utilities (mic, local, cloud)",
    )

    @app.command("mic")
    def mic(
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
        sessions_dir: Optional[Path] = typer.Option(
            None,
            "--sessions-dir",
            help="Directory where diagnostic logs will be written (defaults to sandbox temp dir).",
        ),
        sandbox: bool = typer.Option(
            True,
            "--sandbox/--no-sandbox",
            help="Write logs to a temp dir instead of your real sessions dir.",
        ),
    ) -> None:
        """Run an interactive microphone check and show the transcript."""

        base = _maybe_init_events(sessions_dir, sandbox)
        _print_header("Mic Check")
        console.print(Text(f"events.log: {base / 'events.log'}", style="dim"))

        try:
            selection = resolve_backend_selection(stt_backend, stt_model, stt_compute)
        except Exception as exc:
            console.print(f"[red]STT configuration error:[/] {exc}")
            raise typer.Exit(code=2)

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

    local_app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        help="Local diagnostics (no network): stt, llm, privacy/tts",
    )

    @local_app.command("stt")
    def local_stt(
        stt_backend: str = typer.Option(
            "auto-private",
            "--stt-backend",
            help="STT backend to test (default: auto-private).",
        ),
        stt_model: str = typer.Option(
            CONFIG.model_stt,
            "--stt-model",
            help="Model preset or identifier for the selected backend.",
        ),
        stt_compute: str = typer.Option(
            CONFIG.stt_compute or "auto",
            "--stt-compute",
            help="Optional compute precision override for local backends.",
        ),
        no_audio: bool = typer.Option(
            True,
            "--no-audio/--audio",
            help="Skip interactive mic capture (resolve and instantiate backend only).",
        ),
        language: str = typer.Option("en", "--language"),
        seconds: float = typer.Option(1.0, "--seconds"),
        sample_rate: int = typer.Option(16_000, "--sample-rate"),
        sessions_dir: Optional[Path] = typer.Option(
            None,
            "--sessions-dir",
            help="Diagnostics log directory (defaults to sandbox temp dir).",
        ),
        sandbox: bool = typer.Option(True, "--sandbox/--no-sandbox"),
    ) -> None:
        base = _maybe_init_events(sessions_dir, sandbox)
        _print_header("Local STT")
        _print_config_snapshot()
        console.print(Text(f"events.log: {base / 'events.log'}", style="dim"))

        try:
            selection = resolve_backend_selection(stt_backend, stt_model, stt_compute)
        except Exception as exc:
            console.print(f"[red]STT configuration error:[/] {exc}")
            raise typer.Exit(code=2)

        console.print(
            Panel.fit(
                Text(
                    f"Selected backend: {selection.backend_id}\nModel: {selection.model}\nCompute: {selection.compute or '-'}\nReason: {selection.reason or '-'}\nWarnings: {', '.join(selection.warnings) if selection.warnings else '-'}",
                    style="green",
                ),
                title="STT Selection",
                border_style="green",
            )
        )

        # Instantiate backend to ensure deps are present
        try:
            backend = create_transcription_backend(selection)
            console.print(
                Text(f"Backend instantiated: {backend.__class__.__name__}", style="dim")
            )
            # Show effective runtime characteristics (device/compute) after any fallbacks
            try:
                desc = getattr(backend, "describe", lambda: {})() or {}
            except Exception:
                desc = {}
            effective_compute = (
                desc.get("compute") or getattr(backend, "compute", None) or "-"
            )
            effective_device = getattr(backend, "_device", None) or "-"
            console.print(
                Panel.fit(
                    Text(
                        f"Device: {effective_device}\nCompute: {effective_compute}",
                        style="cyan",
                    ),
                    title="Backend Runtime",
                    border_style="cyan",
                )
            )
        except BackendNotAvailableError as exc:
            console.print(f"[red]Local STT backend not available:[/] {exc}")
            raise typer.Exit(code=2)

        if no_audio:
            return

        run_interactive_mic_check(
            selection,
            console=console,
            language=language,
            stt_formatting=CONFIG.stt_formatting,
            seconds=seconds,
            sample_rate=sample_rate,
        )

    @local_app.command("llm")
    def local_llm(
        prompt: str = typer.Option(
            "Say 'Local LLM OK' and ask one reflective follow-up.",
            "--prompt",
            help="Small fixed prompt to probe local LLM.",
        ),
        max_tokens: int = typer.Option(64, "--max-tokens"),
        sessions_dir: Optional[Path] = typer.Option(None, "--sessions-dir"),
        sandbox: bool = typer.Option(True, "--sandbox/--no-sandbox"),
        fail_on_missing_model: bool = typer.Option(
            False,
            "--fail-on-missing-model/--no-fail-on-missing-model",
            help="If true, error when the local gguf model file is missing instead of reporting guidance.",
        ),
    ) -> None:
        base = _maybe_init_events(sessions_dir, sandbox)
        _print_header("Local LLM")
        _print_config_snapshot()
        console.print(Text(f"events.log: {base / 'events.log'}", style="dim"))

        # Check for local model file presence first to avoid surprise downloads
        manager = get_model_manager()
        model_path = manager.llama_model_path(CONFIG.llm_local_model)
        if not model_path.exists():
            msg = (
                f"Local model not found: {model_path}\n"
                "- Place your gguf at this path, or set [llm].local_model in user_config.toml.\n"
                "- Optionally set [llm].local_model_url and local_model_sha256 to enable managed download.\n"
                "- Or run: healthyselfjournal init local-llm --url <gguf_url> [--sha256 <checksum>]"
            )
            if fail_on_missing_model:
                raise typer.Exit(code=2)
            console.print(
                Panel.fit(
                    Text(msg, style="yellow"),
                    title="Model Missing",
                    border_style="yellow",
                )
            )
            return

        spec = f"local:{CONFIG.llm_local_model}"
        req = QuestionRequest(
            model=spec,
            current_transcript=prompt,
            recent_summaries=[],
            opening_question="",
            question_bank=[],
            language="en",
            conversation_duration="0s",
            max_tokens=max_tokens,
        )

        started = time.time()
        try:
            resp = generate_followup_question(req)
        except Exception as exc:
            console.print(f"[red]Local LLM error:[/] {exc}")
            raise typer.Exit(code=2)
        elapsed = (time.time() - started) * 1000

        console.print(
            Panel.fit(
                Text(
                    f"Model: {resp.model}\nLatency: {elapsed:.0f} ms\nOutput: {resp.question}",
                    style="green",
                ),
                title="Local LLM Response",
                border_style="green",
            )
        )

    @local_app.command("privacy")
    def local_privacy(
        sessions_dir: Optional[Path] = typer.Option(None, "--sessions-dir"),
        sandbox: bool = typer.Option(True, "--sandbox/--no-sandbox"),
    ) -> None:
        base = _maybe_init_events(sessions_dir, sandbox)
        _print_header("Privacy & TTS")
        _print_config_snapshot()
        console.print(Text(f"events.log: {base / 'events.log'}", style="dim"))

        # TTS should be blocked when cloud_off is true and backend is cloud
        opts = resolve_tts_options({})
        try:
            synthesize_text("Test", opts)
            console.print(
                Panel.fit(
                    Text(
                        "Cloud TTS call was allowed – privacy mode is OFF or using local backend.",
                        style="yellow",
                    ),
                    title="TTS Privacy",
                    border_style="yellow",
                )
            )
        except TTSError as exc:
            console.print(
                Panel.fit(
                    Text(f"Blocked as expected: {exc}", style="green"),
                    title="TTS Privacy",
                    border_style="green",
                )
            )

    # Note: running `diagnose local` with no subcommand executes the callback above

    cloud_app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        help="Cloud diagnostics (keys, optional live probes)",
    )

    @cloud_app.command("llm")
    def cloud_llm(
        probe: bool = typer.Option(
            False, "--probe/--no-probe", help="Perform a minimal live Anthropic call"
        )
    ) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        provider = get_model_provider(get_active_llm_model_spec())
        if not api_key:
            console.print("[yellow]ANTHROPIC_API_KEY not set.[/]")
            return
        console.print(f"[green]ANTHROPIC_API_KEY present.[/]")
        if provider != "anthropic":
            console.print(
                Text(
                    f"Active provider is '{provider}'. Use [llm].mode='cloud' or set LLM_MODEL to Anthropic to probe.",
                    style="dim",
                )
            )
        if probe:
            from .llm import SummaryRequest, generate_summary

            try:
                resp = generate_summary(
                    SummaryRequest(
                        transcript_markdown="Probe",
                        recent_summaries=[],
                        model="anthropic:claude-sonnet-4:20250514",
                        max_tokens=8,
                    )
                )
                console.print(Text(f"Probe OK: {resp.model}", style="green"))
            except Exception as exc:
                console.print(f"[red]Probe failed:[/] {exc}")

    @cloud_app.command("stt")
    def cloud_stt() -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            console.print("[yellow]OPENAI_API_KEY not set.[/]")
            return
        console.print("[green]OPENAI_API_KEY present for cloud STT.[/]")

    @cloud_app.command("tts")
    def cloud_tts() -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            console.print("[yellow]OPENAI_API_KEY not set.[/]")
            return
        console.print("[green]OPENAI_API_KEY present for cloud TTS.[/]")

    app.add_typer(local_app, name="local")
    app.add_typer(cloud_app, name="cloud")

    return app
