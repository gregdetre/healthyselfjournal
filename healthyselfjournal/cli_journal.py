from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
import tempfile
import shutil

from . import __version__
from .cli_init import needs_init, run_init_wizard
from .config import CONFIG
from .events import get_event_log_path, init_event_logger, log_event
from .history import load_recent_summaries
from .llm import SummaryRequest, generate_summary, get_model_provider
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
    ctx: typer.Context,
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
    llm_questions_debug: bool = typer.Option(
        False,
        "--llm-questions-debug/--no-llm-questions-debug",
        help="Append a debug postscript to LLM questions about techniques used.",
    ),
    mic_check: bool = typer.Option(
        False,
        "--mic-check/--no-mic-check",
        help="Run a 3s mic check on startup (ENTER continue, ESC retry, q quit).",
    ),
) -> None:
    """Run the interactive voice journaling session."""

    # If a subcommand is invoked under the journal app, do nothing here
    if getattr(ctx, "invoked_subcommand", None):
        return

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
                "[red]Configuration incomplete.[/] Run [cyan]healthyselfjournal init[/] to get set up."
            )
            raise typer.Exit(code=2)

    selection, stream_llm, tts_model, tts_voice, tts_format = (
        prepare_runtime_and_backends(
            llm_model=llm_model,
            stt_backend=stt_backend,
            stt_model=stt_model,
            stt_compute=stt_compute,
            stt_formatting=stt_formatting,
            voice_mode=voice_mode,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_format=tts_format,
            language=language,
            mic_check=mic_check,
            stream_llm=stream_llm,
        )
    )

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
            "mic_check": mic_check,
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
        llm_questions_debug=llm_questions_debug,
    )
    manager = SessionManager(session_cfg)

    # Optional mic check before starting or resuming a session
    if mic_check:
        try:
            _run_mic_check(selection, language=language, stt_formatting=stt_formatting)
        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[yellow]Mic check failed; continuing:[/] {exc}")

    state, question = start_or_resume_session(
        manager,
        sessions_dir=sessions_dir,
        opening_question=opening_question,
        resume=resume,
    )

    try:
        run_journaling_loop(
            manager=manager,
            initial_question=question,
            stream_llm=stream_llm,
            sessions_dir=sessions_dir,
            state=state,
        )
    except KeyboardInterrupt:
        console.print("[red]Session interrupted by user.[/]")
    finally:
        finalize_or_cleanup(manager=manager, state=state, sessions_dir=sessions_dir)


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


def _run_mic_check(selection, *, language: str, stt_formatting: str) -> None:
    """Run a 3-second mic check loop until user accepts or quits.

    - Records to a temporary directory
    - Disables MP3 conversion and short-answer discard
    - Deletes all artifacts after display
    - ENTER continues; ESC retries; q quits the app
    """
    # Lazy imports to keep test deps light
    try:
        import readchar  # type: ignore
    except Exception:
        readchar = None  # type: ignore

    # Construct a transcription backend matching current selection
    backend = create_transcription_backend(selection)

    while True:
        console.print(
            Panel.fit(
                "Mic check: speak a few words. We'll record for 3 seconds and show the transcript.\n"
                "Press ENTER to continue, ESC to try again, or q to quit.",
                title="Mic Check",
                border_style="magenta",
            )
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix="elj_miccheck_"))
        try:
            # Reuse audio capture with fixed duration and no persistence side-effects
            from .audio import record_response

            capture = record_response(
                tmp_dir,
                base_filename="miccheck",
                console=console,
                sample_rate=16_000,
                ffmpeg_path=None,
                print_saved_message=False,
                convert_to_mp3=False,
                max_seconds=3.0,
                enforce_short_answer_guard=False,
            )

            if capture.cancelled:
                # If user pressed ESC during capture, just retry automatically
                continue

            # Transcribe and show formatted transcript without persisting
            transcription = backend.transcribe(capture.wav_path, language=language)
            try:
                formatted = apply_transcript_formatting(
                    transcription.text, stt_formatting
                )
            except Exception:
                formatted = transcription.text.strip()

            console.print()
            console.print(
                Panel.fit(
                    formatted or "(no speech detected)",
                    title="Mic Check Transcript",
                    border_style="green",
                )
            )
            console.print(
                Text(
                    f"Backend: {selection.backend_id}  Model: {selection.model}",
                    style="dim",
                )
            )
        finally:
            # Remove artifacts regardless of outcome
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

        # Await user decision
        console.print(
            Text(
                "Press ENTER to continue, ESC to try again, or q to quit…",
                style="cyan",
            )
        )
        # Use shared key normalization utility
        try:
            from .utils.keys import read_one_key_normalized
        except Exception:
            read_one_key_normalized = None  # type: ignore

        if readchar is None or read_one_key_normalized is None:
            # Fallback: on raw input, treat non-empty as retry, 'q' to quit
            try:
                response = input()
            except KeyboardInterrupt:
                raise typer.Exit(code=0)
            if response.strip().lower() == "q":
                console.print("[cyan]Quit requested. Exiting before session starts.[/]")
                raise typer.Exit(code=0)
            if response.strip():
                continue
            return
        else:
            try:
                key_name = read_one_key_normalized()
                if key_name == "ENTER":
                    return
                if key_name == "ESC":
                    continue
                if key_name == "Q":
                    console.print(
                        "[cyan]Quit requested. Exiting before session starts.[/]"
                    )
                    raise typer.Exit(code=0)
                # Any other key: accept and proceed
                return
            except KeyboardInterrupt:
                raise typer.Exit(code=0)


def prepare_runtime_and_backends(
    *,
    llm_model: str,
    stt_backend: str,
    stt_model: str,
    stt_compute: str,
    stt_formatting: str,
    voice_mode: bool,
    tts_model: str,
    tts_voice: str,
    tts_format: str,
    language: str,
    mic_check: bool,
    stream_llm: bool,
):
    """Resolve and validate runtime settings and STT/TTS backends.

    Returns (selection, stream_llm, tts_model, tts_voice, tts_format).
    """
    provider = get_model_provider(llm_model)
    if provider == "anthropic":
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
    if voice_mode:
        CONFIG.speak_llm = True
        tts_model = tts_model or "gpt-4o-mini-tts"
        tts_voice = tts_voice or "shimmer"
        tts_format = tts_format or "wav"

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

    return selection, stream_llm, tts_model, tts_voice, tts_format


def start_or_resume_session(
    manager: SessionManager,
    *,
    sessions_dir: Path,
    opening_question: str,
    resume: bool,
):
    """Start a new session or resume the most recent, returning (state, question)."""
    if resume:
        markdown_files = sorted((p for p in sessions_dir.glob("*.md")), reverse=True)
        if not markdown_files:
            console.print(
                Panel.fit(
                    "No prior sessions found. Starting a new session.",
                    title="Healthy Self Journal",
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
                    "Tip: Say 'give me a question' to get a quick prompt from the built-in examples.",
                    title="Healthy Self Journal",
                    border_style="magenta",
                )
            )
            # Surface pending transcription work, but don't auto-run.
            pending = _count_missing_stt(sessions_dir)
            if pending:
                console.print(
                    f"[yellow]{pending} recording(s) pending transcription.[/] "
                    f"Run [cyan]healthyselfjournal reconcile --sessions-dir '{sessions_dir}'[/] to backfill."
                )
    else:
        console.print(
            Panel.fit(
                "Voice journaling session starting. Recording starts immediately.\n"
                "Press any key to stop. Q saves then ends after this entry.\n\n"
                "Tip: Say 'give me a question' to get a quick prompt from the built-in examples.",
                title="Healthy Self Journal",
                border_style="magenta",
            )
        )
        state = manager.start()
        question = opening_question
        pending = _count_missing_stt(sessions_dir)
        if pending:
            console.print(
                f"[yellow]{pending} recording(s) pending transcription.[/] "
                f"Run [cyan]healthyselfjournal reconcile --sessions-dir '{sessions_dir}'[/] to backfill."
            )

    return state, question


def run_journaling_loop(
    *,
    manager: SessionManager,
    initial_question: str,
    stream_llm: bool,
    sessions_dir: Path,
    state,
) -> None:
    """Run the main capture → transcribe → ask loop until quit/cancel."""
    question = initial_question
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
                "[cyan]healthyselfjournal reconcile --sessions-dir '{sessions_dir}'[/]"
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
        console.print(Panel.fit(exchange.transcript, title="You", border_style="green"))

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
                    buffer.append(chunk)

                with Live(console=console, auto_refresh=True, transient=True) as live:
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


def finalize_or_cleanup(*, manager: SessionManager, state, sessions_dir: Path) -> None:
    """Finalize session and clean up empty artifacts as in original flow."""
    # If nothing was recorded and this wasn't a resume, don't keep an empty session file
    is_empty_session = False
    try:
        if manager.state is not None and not manager.state.resumed:
            has_exchanges = len(manager.state.exchanges) > 0
            has_audio_artifacts = (
                any(manager.state.audio_dir.glob("*.wav"))
                or any(manager.state.audio_dir.glob("*.mp3"))
                or any(manager.state.audio_dir.glob("*.stt.json"))
            )
            doc = load_transcript(state.markdown_path)
            body_empty = not bool(doc.body.strip())
            is_empty_session = (
                (not has_exchanges) and (not has_audio_artifacts) and body_empty
            )
    except Exception:
        is_empty_session = False

    if is_empty_session:
        try:
            if getattr(manager, "_summary_executor", None) is not None:
                manager._summary_executor.shutdown(wait=False)
        except Exception:
            pass

        try:
            state.markdown_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if state.audio_dir.exists() and not any(state.audio_dir.iterdir()):
                state.audio_dir.rmdir()
        except Exception:
            pass

        log_event(
            "cli.cancelled",
            {
                "session_id": state.session_id,
                "reason": "no_recordings",
            },
        )
        console.print("[yellow]Session cancelled; nothing saved.[/]")
    else:
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
        pending = _count_missing_stt(sessions_dir)
        if pending:
            console.print(
                f"[yellow]{pending} recording(s) still pending transcription.[/] "
                f"Use [cyan]healthyselfjournal reconcile --sessions-dir '{sessions_dir}'[/] to process them."
            )


def build_app() -> typer.Typer:
    """Build the Typer sub-app for `journal` with subcommands.

    Invoking `healthyselfjournal journal` with no subcommand runs the interactive
    journaling loop (default). `healthyselfjournal journal list` lists sessions.
    """

    app = typer.Typer(
        add_completion=False,
        no_args_is_help=False,
        context_settings={"help_option_names": ["-h", "--help"]},
        help="Voice journaling and related utilities.",
    )

    # Default behavior: when no subcommand is provided, run the journaling loop
    app.callback(invoke_without_command=True)(journal)

    @app.command("list")
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
        """List sessions by filename stem with the first 200 chars of the summary."""

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

    return app
