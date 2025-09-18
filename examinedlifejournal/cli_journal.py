from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .cli_init import needs_init, run_init_wizard
from .config import CONFIG
from .events import get_event_log_path, init_event_logger, log_event
from .history import load_recent_summaries
from .llm import SummaryRequest, generate_summary
from .session import SessionConfig, SessionManager
from .storage import load_transcript, write_transcript
from .transcription import (
    BackendNotAvailableError,
    apply_transcript_formatting,
    create_transcription_backend,
    resolve_backend_selection,
)
from .tts import TTSOptions, speak_text


console = Console()


def journal(
    sessions_dir: Path = typer.Option(
        CONFIG.recordings_dir,
        "--sessions-dir",
        help="Directory where session markdown/audio files are stored.",
    ),
    llm_model: str = typer.Option(
        CONFIG.model_llm,
        "--llm-model",
        help="LLM model string: provider:model:version[:thinking] (e.g., anthropic:claude-sonnet-4:20250514:thinking)",
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
    stt_formatting: str = typer.Option(
        CONFIG.stt_formatting,
        "--stt-formatting",
        help="Transcript formatting mode: sentences (default) or raw.",
    ),
    opening_question: str = typer.Option(
        CONFIG.opening_question,
        "--opening-question",
        help="Initial question used to start each session.",
    ),
    language: str = typer.Option(
        "en",
        "--language",
        help="Primary language for transcription and LLM guidance.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume the most recent session in the sessions directory.",
    ),
    delete_wav_when_safe: bool = typer.Option(
        True,
        "--delete-wav-when-safe/--keep-wav",
        help="Delete WAV after MP3+STT exist (saves disk).",
    ),
    stream_llm: bool = typer.Option(
        True,
        "--stream-llm/--no-stream-llm",
        help="Stream the next question from the LLM for lower perceived latency.",
    ),
    voice_mode: bool = typer.Option(
        False,
        "--voice-mode/--no-voice-mode",
        help=(
            "Convenience switch: enable speech with default TTS settings (shimmer, gpt-4o-mini-tts, wav)."
        ),
    ),
    tts_model: str = typer.Option(
        CONFIG.tts_model,
        "--tts-model",
        help="TTS model identifier (default: gpt-4o-mini-tts).",
    ),
    tts_voice: str = typer.Option(
        CONFIG.tts_voice,
        "--tts-voice",
        help="TTS voice name (e.g., alloy).",
    ),
    tts_format: str = typer.Option(
        CONFIG.tts_format,
        "--tts-format",
        help="TTS audio format for playback (wav recommended).",
    ),
) -> None:
    """Run the interactive voice journaling session."""

    # Auto-run init wizard if critical prerequisites are missing and we are in a TTY.
    # This respects any values loaded from .env/.env.local in __init__ at import time.
    if needs_init(stt_backend):
        if sys.stdin.isatty():
            console.print(
                Panel.fit(
                    "It looks like you haven't finished setup yet. Launching the setup wizard…",
                    title="First-time Setup",
                    border_style="magenta",
                )
            )
            try:
                run_init_wizard()
            except typer.Abort:
                console.print("[red]Setup cancelled.[/]")
                raise typer.Exit(code=2)
            # Refresh effective options from env in case wizard updated them
            stt_backend = os.environ.get("STT_BACKEND", stt_backend)
            stt_model = os.environ.get("STT_MODEL", stt_model)
            stt_compute = os.environ.get("STT_COMPUTE", stt_compute)
        else:
            console.print(
                "[red]Configuration incomplete.[/] Run [cyan]examinedlifejournal init[/] to get set up."
            )
            raise typer.Exit(code=2)

    _require_env("ANTHROPIC_API_KEY")

    try:
        apply_transcript_formatting("sample", stt_formatting)
    except ValueError as exc:
        console.print(f"[red]Invalid --stt-formatting:[/] {exc}")
        raise typer.Exit(code=2)

    try:
        selection = resolve_backend_selection(stt_backend, stt_model, stt_compute)
    except (ValueError, BackendNotAvailableError) as exc:
        console.print(f"[red]STT configuration error:[/] {exc}")
        raise typer.Exit(code=2)

    CONFIG.model_stt = selection.model
    CONFIG.stt_backend = selection.backend_id
    CONFIG.stt_compute = selection.compute
    CONFIG.stt_formatting = stt_formatting

    if selection.reason:
        console.print(
            f"[cyan]auto-private[/] -> using [bold]{selection.backend_id}[/] ({selection.reason})"
        )
    if selection.warnings:
        for warning in selection.warnings:
            console.print(f"[yellow]STT warning:[/] {warning}")

    # Propagate config flags for TTS
    # Convenience: --voice-mode turns on TTS and sets sensible defaults
    if voice_mode:
        CONFIG.speak_llm = True
        tts_model = tts_model or "gpt-4o-mini-tts"
        tts_voice = tts_voice or "shimmer"
        tts_format = tts_format or "wav"

    # Allow explicit config override via env/CONFIG but not via removed flag
    CONFIG.tts_model = str(tts_model)
    CONFIG.tts_voice = str(tts_voice)
    CONFIG.tts_format = str(tts_format)

    # When speaking is enabled, disable LLM streaming for clearer UX
    if CONFIG.speak_llm and stream_llm:
        console.print(
            "[yellow]Speech enabled; disabling streaming display for clarity.[/]"
        )
        stream_llm = False

    # Require OpenAI key only if using cloud STT
    if selection.backend_id == "cloud-openai":
        _require_env("OPENAI_API_KEY")

    # Also require OpenAI key when TTS is enabled (OpenAI backend)
    if CONFIG.speak_llm:
        _require_env("OPENAI_API_KEY")

    # Initialize append-only metadata event logger
    init_event_logger(sessions_dir)
    log_event(
        "cli.start",
        {
            "sessions_dir": sessions_dir,
            "model_llm": llm_model,
            "stt_backend": selection.backend_id,
            "model_stt": selection.model,
            "stt_compute": selection.compute,
            "stt_requested_backend": stt_backend,
            "stt_requested_model": stt_model,
            "stt_requested_compute": stt_compute,
            "stt_formatting": stt_formatting,
            "language": language,
            "events_log": str(get_event_log_path() or ""),
            "app_version": __version__,
            "resume": resume,
            "stt_auto_reason": selection.reason,
            "stt_warnings": selection.warnings,
        },
    )

    # Propagate config flag
    CONFIG.delete_wav_when_safe = bool(delete_wav_when_safe)

    session_cfg = SessionConfig(
        base_dir=sessions_dir,
        llm_model=llm_model,
        stt_model=selection.model,
        stt_backend=selection.backend_id,
        stt_compute=selection.compute,
        opening_question=opening_question,
        language=language,
        stt_formatting=stt_formatting,
        stt_backend_requested=stt_backend,
        stt_model_requested=stt_model,
        stt_compute_requested=stt_compute,
        stt_auto_private_reason=selection.reason,
        stt_backend_selection=selection,
        stt_warnings=selection.warnings,
    )
    manager = SessionManager(session_cfg)

    # Determine whether to start new or resume recent session
    if resume:
        markdown_files = sorted((p for p in sessions_dir.glob("*.md")), reverse=True)
        if not markdown_files:
            console.print(
                Panel.fit(
                    "No prior sessions found. Starting a new session.",
                    title="Examined Life Journal",
                    border_style="magenta",
                )
            )
            state = manager.start()
            question = opening_question
        else:
            latest_md = markdown_files[0]
            state = manager.resume(latest_md)
            doc = load_transcript(state.markdown_path)
            if doc.body.strip():
                try:
                    next_q = manager.generate_next_question(doc.body)
                    question = next_q.question
                except Exception as exc:
                    console.print(f"[red]Question generation failed:[/] {exc}")
                    question = opening_question
            else:
                question = opening_question
            console.print(
                Panel.fit(
                    f"Resuming session {state.session_id}. Recording starts immediately.\n"
                    "Press any key to stop. Q saves then ends after this entry.\n\n"
                    "Tip: Say 'give me a question' to get a quick prompt from the question bank.",
                    title="Examined Life Journal",
                    border_style="magenta",
                )
            )
            # Surface pending transcription work, but don't auto-run.
            pending = _count_missing_stt(sessions_dir)
            if pending:
                console.print(
                    f"[yellow]{pending} recording(s) pending transcription.[/] "
                    f"Run [cyan]examinedlifejournal reconcile --sessions-dir '{sessions_dir}'[/] to backfill."
                )
    else:
        console.print(
            Panel.fit(
                "Voice journaling session starting. Recording starts immediately.\n"
                "Press any key to stop. Q saves then ends after this entry.\n\n"
                "Tip: Say 'give me a question' to get a quick prompt from the question bank.",
                title="Examined Life Journal",
                border_style="magenta",
            )
        )
        state = manager.start()
        question = opening_question
        pending = _count_missing_stt(sessions_dir)
        if pending:
            console.print(
                f"[yellow]{pending} recording(s) pending transcription.[/] "
                f"Run [cyan]examinedlifejournal reconcile --sessions-dir '{sessions_dir}'[/] to backfill."
            )

    try:
        while True:
            console.print(
                Panel.fit(
                    question,
                    title="AI",
                    border_style="cyan",
                )
            )
            console.print()

            # Speak the assistant's question before recording, if enabled
            if CONFIG.speak_llm:
                try:
                    console.print(
                        Text(
                            "(Press ENTER to skip the spoken question)",
                            style="dim",
                        )
                    )
                    speak_text(
                        question,
                        TTSOptions(
                            backend="openai",
                            model=CONFIG.tts_model,
                            voice=CONFIG.tts_voice,
                            audio_format=CONFIG.tts_format,  # type: ignore[arg-type]
                        ),
                    )
                except Exception as exc:  # pragma: no cover - runtime path
                    console.print(
                        f"[yellow]TTS failed; continuing without speech:[/] {exc}"
                    )

            try:
                exchange = manager.record_exchange(question, console)
            except Exception as exc:  # pragma: no cover - runtime error surface
                # Keep the session alive: audio is already saved on disk.
                console.print(
                    f"[red]Transcription failed:[/] {exc}\n"
                    "[yellow]Your audio was saved.[/] You can backfill later with: "
                    "[cyan]examinedlifejournal reconcile --sessions-dir '{sessions_dir}'[/]"
                )
                log_event(
                    "cli.error",
                    {
                        "where": "record_exchange",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "action": "continue_without_transcript",
                    },
                )
                # Re-ask the same question so the user can continue.
                # Skip transcript display and summary scheduling.
                continue

            if exchange is None:
                # Could be cancelled or discarded short answer; message accordingly
                if manager.state and manager.state.quit_requested:
                    console.print(
                        "[cyan]Quit requested. Ending session after summary update.[/]"
                    )
                    break
                # Provide clearer feedback depending on disposition flags
                if manager.state and getattr(manager.state, "last_cancelled", False):
                    console.print(
                        "[yellow]Cancelled. Take discarded. Re-asking the same question...[/]"
                    )
                    # Reset flag to avoid stale messaging on next loop
                    manager.state.last_cancelled = False
                elif manager.state and getattr(
                    manager.state, "last_discarded_short", False
                ):
                    console.print(
                        "[yellow]Very short/quiet; take discarded. Re-asking the same question...[/]"
                    )
                    manager.state.last_discarded_short = False
                else:
                    console.print(
                        "[yellow]No usable answer captured (cancelled or very short). Re-asking...[/]"
                    )
                continue

            console.print()
            console.print(
                Panel.fit(exchange.transcript, title="You", border_style="green")
            )

            try:
                # Background scheduling to reduce latency
                manager.schedule_summary_regeneration()
            except Exception as exc:
                console.print(f"[red]Summary scheduling failed:[/] {exc}")
                log_event(
                    "cli.error",
                    {
                        "where": "schedule_summary_regeneration",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )

            transcript_doc = load_transcript(state.markdown_path)

            if exchange.audio.quit_after:
                console.print(
                    "[cyan]Quit requested. Ending session after summary update.[/]"
                )
                break

            try:
                if stream_llm:
                    buffer: list[str] = []

                    def on_delta(chunk: str) -> None:
                        # Accumulate and re-render live panel
                        buffer.append(chunk)

                    # Render live as tokens stream in
                    with Live(
                        console=console, auto_refresh=True, transient=True
                    ) as live:
                        # Use a small spinner-like feedback until first token arrives
                        live.update(
                            Panel.fit(
                                Text("Thinking…", style="italic cyan"),
                                title="Next Question",
                                border_style="cyan",
                            )
                        )
                        next_question = manager.generate_next_question_streaming(
                            transcript_doc.body, on_delta
                        )
                        # Final render with the fully assembled text
                        streamed_text = "".join(buffer)
                        question_text = (
                            next_question.question
                            if next_question.question
                            else streamed_text
                        )
                        live.update(
                            Panel.fit(
                                question_text,
                                title="Next Question",
                                border_style="cyan",
                            )
                        )
                else:
                    next_question = manager.generate_next_question(transcript_doc.body)
            except Exception as exc:
                console.print(f"[red]Question generation failed:[/] {exc}")
                log_event(
                    "cli.error",
                    {
                        "where": "generate_next_question",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                break

            question = next_question.question

    except KeyboardInterrupt:
        console.print("[red]Session interrupted by user.[/]")
    finally:
        # Surface progress while waiting for the final summary flush
        console.print("[cyan]Finalizing summary before exit…[/]")
        manager.complete()
        console.print("[green]Summary updated.[/]")
        log_event(
            "cli.end",
            {
                "transcript_file": state.markdown_path.name,
                "session_id": state.session_id,
            },
        )
        console.print(
            Panel.fit(
                f"Session saved to {state.markdown_path.name}",
                title="Session Complete",
                border_style="magenta",
            )
        )
        # Remind user if there is pending STT work.
        pending = _count_missing_stt(sessions_dir)
        if pending:
            console.print(
                f"[yellow]{pending} recording(s) still pending transcription.[/] "
                f"Use [cyan]examinedlifejournal reconcile --sessions-dir '{sessions_dir}'[/] to process them."
            )


def _require_env(var_name: str) -> None:
    if not os.environ.get(var_name):
        console.print(f"[red]Environment variable {var_name} is required.[/]")
        raise typer.Exit(code=2)


def _count_missing_stt(audio_root: Path) -> int:
    """Return the number of .wav files without a sibling .stt.json under a root dir."""
    if not audio_root.exists():
        return 0
    missing = 0
    for wav in audio_root.rglob("*.wav"):
        stt = wav.with_suffix(".stt.json")
        if not stt.exists():
            missing += 1
    return missing
