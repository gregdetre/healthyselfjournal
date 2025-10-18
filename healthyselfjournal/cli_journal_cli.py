from __future__ import annotations

import os
import platform
import time
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
import tempfile
import shutil
import httpx

from . import __version__
from .cli_init import needs_init, run_init_wizard
from .config import CONFIG
from .events import get_event_log_path, init_event_logger, log_event
from .history import load_recent_summaries
from .llm import SummaryRequest, generate_summary, get_model_provider
from .session import PendingTranscriptionError, SessionConfig, SessionManager
from .storage import load_transcript, write_transcript
from .transcription import (
    BackendNotAvailableError,
    apply_transcript_formatting,
    resolve_backend_selection,
    CLOUD_BACKEND,
    LOCAL_MLX_BACKEND,
    LOCAL_FASTER_BACKEND,
    LOCAL_WHISPERCPP_BACKEND,
    AUTO_PRIVATE_BACKEND,
)
from .mic_check import run_interactive_mic_check
from .tts import TTSOptions, speak_text
from .utils.pending import (
    count_pending_segments,
    count_pending_for_session,
    pending_segments_by_session,
    reconcile_command_for_dir,
)


console = Console()


# Default Ollama model used by the private mode alias
_OLLAMA_DEFAULT_MODEL = "ollama:qwen2.5:7b-instruct"


class _EnvScope:
    def __init__(self, updates: Dict[str, Optional[str]]) -> None:
        self._updates = updates
        self._original: Dict[str, Optional[str]] = {}

    def __enter__(self) -> None:
        for key, value in self._updates.items():
            self._original[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, original in self._original.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _is_interactive_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _normalise_path_option(value: Optional[Path], default: Path) -> Path:
    if value is None:
        return default
    return value


def _normalise_bool_option(value: Optional[bool], default: bool) -> bool:
    if value is None:
        return default
    return value


def _normalise_str_option(value: Optional[str], default: str) -> str:
    if value is None or value == "":
        return default
    return value


def _extract_ollama_model_id(llm_model: str) -> str:
    if ":" not in llm_model:
        return llm_model
    provider, rest = llm_model.split(":", 1)
    if provider.lower() != "ollama":
        return llm_model
    return rest


def _run_ollama_preflight_or_exit(llm_model: str) -> None:
    """Validate local Ollama availability for the given model.

    Checks:
    1) Binary present on PATH
    2) Daemon reachable at CONFIG.ollama_base_url (/api/tags)
    3) Model tag present; if missing, propose `ollama pull <model>`

    Offers interactive auto-fix in a TTY for safe operations:
    - Start desktop app on macOS (open -a Ollama), then retry
    - Pull missing model with `ollama pull <model>`
    Exits with code 2 when an unrecoverable issue remains.
    """
    provider = get_model_provider(llm_model)
    if provider != "ollama":
        return

    model_id = _extract_ollama_model_id(llm_model)
    base_url = CONFIG.ollama_base_url.rstrip("/")

    # 1) Binary present
    if not shutil.which("ollama"):
        lines = [
            "Ollama CLI not found on PATH.",
            "Next step:",
            "  - Install Ollama, then rerun this command.",
            "    macOS (Homebrew): brew install ollama",
            "    Or download: https://ollama.com/download",
        ]
        console.print(
            Panel.fit("\n".join(lines), title="Ollama Preflight", border_style="yellow")
        )
        raise typer.Exit(code=2)

    # 2) Daemon reachable
    tags_url = f"{base_url}/api/tags"
    try:
        resp = httpx.get(tags_url, timeout=CONFIG.ollama_timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        # Try to start the daemon interactively on macOS
        if _is_interactive_tty() and platform.system().lower() == "darwin":
            console.print(
                Panel.fit(
                    "Could not reach the Ollama daemon. Start it now?",
                    title="Ollama Preflight",
                    border_style="yellow",
                )
            )
            if typer.confirm("Start Ollama app now?", default=True):
                try:
                    subprocess.run(["open", "-a", "Ollama"], check=False)
                except Exception:
                    pass
                # Give it a moment, then retry once
                time.sleep(2.0)
                try:
                    resp = httpx.get(tags_url, timeout=CONFIG.ollama_timeout_seconds)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc2:
                    _show_ollama_connect_help(base_url, model_id, extra=str(exc2))
                    raise typer.Exit(code=2)
            else:
                _show_ollama_connect_help(base_url, model_id, extra=str(exc))
                raise typer.Exit(code=2)
        else:
            _show_ollama_connect_help(base_url, model_id, extra=str(exc))
            raise typer.Exit(code=2)

    # 3) Model available
    if not _ollama_tags_contains_model(data, model_id):
        suggestion = f"ollama pull {model_id}"
        if _is_interactive_tty():
            console.print(
                Panel.fit(
                    f"Required model not found: {model_id}\nPull now?",
                    title="Ollama Model",
                    border_style="yellow",
                )
            )
            if typer.confirm(f"Run '{suggestion}' now?", default=True):
                try:
                    subprocess.run(["ollama", "pull", model_id], check=True)
                except subprocess.CalledProcessError as exc:
                    console.print(f"[red]ollama pull failed:[/] {exc}")
                    raise typer.Exit(code=2)
            else:
                console.print(
                    Panel.fit(
                        f"Next: run\n  {suggestion}",
                        title="Ollama Model",
                        border_style="yellow",
                    )
                )
                raise typer.Exit(code=2)
        else:
            console.print(
                Panel.fit(
                    f"Missing model: {model_id}\nNext: run\n  {suggestion}",
                    title="Ollama Model",
                    border_style="yellow",
                )
            )
            raise typer.Exit(code=2)


def _ollama_tags_contains_model(tags_json: Any, model_id: str) -> bool:
    try:
        models = tags_json.get("models") or []
        for m in models:
            # Common keys: "model" or "name"
            val = (m.get("model") or m.get("name") or "").strip()
            # Accept exact match or base name match
            if (
                val == model_id
                or val.endswith(f":{model_id}")
                or model_id.endswith(f":{val}")
            ):
                return True
    except Exception:
        return False
    return False


def _show_ollama_connect_help(
    base_url: str, model_id: str, *, extra: str | None = None
) -> None:
    lines: list[str] = []
    if extra:
        lines.append(str(extra))
        lines.append("")
    lines.append(f"Could not reach Ollama at {base_url}.")
    lines.append("Next step:")
    if platform.system().lower() == "darwin":
        lines.append("  - Start the app: open -a Ollama")
        lines.append("  - Or with Homebrew: brew services start ollama")
    else:
        lines.append("  - Start the daemon: ollama serve")
    lines.append("  - If using a remote host/port, set OLLAMA_BASE_URL appropriately")
    lines.append("")
    lines.append("Then verify:")
    lines.append(f"  curl -s {base_url}/api/version && echo")
    lines.append("")
    lines.append("If the model is missing, pull it:")
    lines.append(f"  ollama pull {model_id}")
    console.print(
        Panel.fit("\n".join(lines), title="Ollama Preflight", border_style="yellow")
    )


def _run_cli_session(
    *,
    sessions_dir: Path,
    llm_model: str,
    stt_backend: str,
    stt_model: str,
    stt_compute: str,
    stt_formatting: str,
    opening_question: str,
    language: str,
    resume: bool,
    delete_wav_when_safe: bool,
    stream_llm: bool,
    voice_mode: bool,
    tts_model: str,
    tts_voice: str,
    tts_format: str,
    llm_questions_debug: bool,
    mic_check: bool,
    mode_alias: str | None,
) -> None:
    """Run the interactive voice journaling session."""

    # Auto-run init wizard if critical prerequisites are missing and we are in a TTY.
    # This respects any values loaded from .env/.env.local in __init__ at import time.
    if needs_init(stt_backend):
        if _is_interactive_tty():
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

    # Preflight for Ollama when using an Ollama model for question generation
    try:
        if get_model_provider(llm_model) == "ollama":
            _run_ollama_preflight_or_exit(llm_model)
    except typer.Exit:
        raise
    except Exception as exc:  # pragma: no cover - defensive surface
        console.print(f"[yellow]Ollama preflight warning:[/] {exc}")

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
    metadata = {
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
    }
    if mode_alias:
        metadata["cli.mode_alias"] = mode_alias
    log_event("cli.start", metadata)

    # Show effective runtime models/config once at startup
    try:
        _provider = get_model_provider(llm_model)
    except Exception:
        _provider = "unknown"
    _lines: list[str] = []
    _lines.append(
        f"LLM: {llm_model} (provider: {_provider}; mode={CONFIG.llm_mode}; cloud_off={CONFIG.llm_cloud_off})"
    )
    _lines.append(
        f"STT: {selection.backend_id} model={selection.model} compute={selection.compute or '-'}"
    )
    if selection.reason:
        _lines.append(f"STT auto-private: {selection.reason}")
    if selection.warnings:
        _lines.append("STT warnings: " + ", ".join(selection.warnings))
    if CONFIG.speak_llm and CONFIG.tts_enabled:
        _lines.append(
            f"TTS: openai:{CONFIG.tts_model} voice={CONFIG.tts_voice} format={CONFIG.tts_format}"
        )
    else:
        _lines.append("TTS: disabled")
    console.print(
        Panel.fit(
            Text("\n".join(_lines)),
            title="Runtime Models",
            border_style="cyan",
        )
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
            run_interactive_mic_check(
                selection,
                console=console,
                language=language,
                stt_formatting=stt_formatting,
                seconds=3.0,
            )
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
    """Return the number of recordings missing their transcription payloads."""

    return count_pending_segments(audio_root)


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

    # Propagate config flags for TTS, respecting privacy mode
    if voice_mode:
        CONFIG.speak_llm = True
        # If privacy mode is active, do not allow cloud TTS
        if CONFIG.llm_cloud_off:
            CONFIG.tts_enabled = False
            CONFIG.speak_llm = False
        else:
            CONFIG.tts_enabled = True
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
    if CONFIG.speak_llm and CONFIG.tts_enabled:
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
            state = manager.start()
            # Show absolute sessions dir and actual session file before panels
            abs_dir = str(sessions_dir.expanduser().resolve())
            console.print(
                Text(
                    f"Sessions directory: {abs_dir}\nSession file: {state.markdown_path.resolve()}",
                    style="dim",
                )
            )
            console.print(
                Panel.fit(
                    "No prior sessions found. Starting a new session.",
                    title="Healthyself Journal",
                    border_style="magenta",
                )
            )
            question = opening_question
        else:
            latest_md = markdown_files[0]
            state = manager.resume(latest_md)
            # Show absolute sessions dir and actual session file before panels
            abs_dir = str(sessions_dir.expanduser().resolve())
            console.print(
                Text(
                    f"Sessions directory: {abs_dir}\nSession file: {state.markdown_path.resolve()}",
                    style="dim",
                )
            )
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
                    title="Healthyself Journal",
                    border_style="magenta",
                )
            )
            # Surface pending transcription work, but don't auto-run.
            pending = count_pending_for_session(sessions_dir, state.session_id)
            if pending:
                cmd = reconcile_command_for_dir(sessions_dir)
                console.print(
                    f"[yellow]{pending} recording(s) pending transcription in this session.[/] "
                    f"Run [cyan]{cmd}[/] to backfill. "
                    f"[dim]Tip: add --min-duration 0.6 --too-short mark[/]"
                )
    else:
        state = manager.start()
        # Show absolute sessions dir and actual session file before panels
        abs_dir = str(sessions_dir.expanduser().resolve())
        console.print(
            Text(
                f"Sessions directory: {abs_dir}\nSession file: {state.markdown_path.resolve()}",
                style="dim",
            )
        )
        console.print(
            Panel.fit(
                "Voice journaling session starting. Recording starts immediately.\n"
                "Press any key to stop. Q saves then ends after this entry.\n\n"
                "Tip: Say 'give me a question' to get a quick prompt from the built-in examples.",
                title="Healthyself Journal",
                border_style="magenta",
            )
        )
        question = opening_question
        pending = count_pending_for_session(sessions_dir, state.session_id)
        if pending:
            cmd = reconcile_command_for_dir(sessions_dir)
            console.print(
                f"[yellow]{pending} recording(s) pending transcription in this session.[/] "
                f"Run [cyan]{cmd}[/] to backfill. "
                f"[dim]Tip: add --min-duration 0.6 --too-short mark[/]"
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
        except PendingTranscriptionError as exc:
            cmd = reconcile_command_for_dir(sessions_dir)
            console.print(
                f"[red]Transcription failed:[/] {exc.error}\n"
                "[yellow]Your audio was saved.[/] Backfill later with "
                f"[cyan]{cmd}[/]."
            )
            log_event(
                "cli.error",
                {
                    "where": "record_exchange",
                    "error_type": exc.error_type,
                    "error": str(exc.error),
                    "action": "continue_without_transcript",
                    "pending": True,
                    "segment": exc.segment_label,
                },
            )
            continue
        except Exception as exc:  # pragma: no cover - runtime error surface
            cmd = reconcile_command_for_dir(sessions_dir)
            console.print(
                f"[red]Transcription failed:[/] {exc}\n"
                "[yellow]Your audio was saved.[/] You can backfill later with: "
                f"[cyan]{cmd}[/]"
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
                f"Session saved to {state.markdown_path.resolve()}",
                title="Session Complete",
                border_style="magenta",
            )
        )
        pending = count_pending_for_session(sessions_dir, state.session_id)
        if pending:
            cmd = reconcile_command_for_dir(sessions_dir)
            console.print(
                f"[yellow]{pending} recording(s) still pending transcription in this session.[/] "
                f"Use [cyan]{cmd}[/] to process them. "
                f"[dim]Tip: add --min-duration 0.6 --too-short mark[/]"
            )


def journal(
    sessions_dir: Path = typer.Option(
        CONFIG.recordings_dir,
        "--sessions-dir",
        help="Directory where session markdown/audio files are stored.",
    ),
    llm_model: str = typer.Option(
        CONFIG.model_llm,
        "--llm-model",
        help="LLM model spec (e.g., anthropic:claude-sonnet-4:20250514 or ollama:qwen2.5:7b-instruct)",
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
        help="First question to show before recording begins.",
    ),
    language: str = typer.Option(
        "en", "--language", help="Primary language for transcription."
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--new",
        help="Resume the most recent session in the sessions directory.",
    ),
    delete_wav_when_safe: bool = typer.Option(
        True,
        "--delete-wav-when-safe/--keep-wav",
        help="Delete large WAV files once MP3 and STT JSON exist (default on).",
    ),
    stream_llm: bool = typer.Option(
        True,
        "--stream-llm/--no-stream-llm",
        help="Stream the next question as it is generated (off automatically when voice mode).",
    ),
    voice_mode: bool = typer.Option(
        False,
        "--voice-mode/--no-voice-mode",
        help="Speak the assistant's questions using TTS (cloud-only for now).",
    ),
    tts_model: str = typer.Option(
        CONFIG.tts_model, "--tts-model", help="TTS model (OpenAI)."
    ),
    tts_voice: str = typer.Option(
        CONFIG.tts_voice, "--tts-voice", help="TTS voice (OpenAI)."
    ),
    tts_format: str = typer.Option(
        CONFIG.tts_format, "--tts-format", help="TTS audio format."
    ),
    llm_questions_debug: bool = typer.Option(
        False,
        "--llm-questions-debug/--no-llm-questions-debug",
        help="Include hidden metadata in prompts to aid debugging.",
    ),
    mic_check: bool = typer.Option(
        False,
        "--mic-check/--no-mic-check",
        help="Run a brief mic check before starting the session.",
    ),
    mode_alias: Optional[str] = typer.Option(
        None,
        "--_mode-alias",
        hidden=True,
        help="Internal: set when invoked via an alias (private|cloud).",
    ),
):
    """Start the interactive CLI journaling session."""

    base_dir = _normalise_path_option(sessions_dir, CONFIG.recordings_dir)
    stt_backend_eff = _normalise_str_option(stt_backend, CONFIG.stt_backend)
    stt_model_eff = _normalise_str_option(stt_model, CONFIG.model_stt)
    stt_compute_eff = _normalise_str_option(stt_compute, CONFIG.stt_compute or "auto")
    stt_formatting_eff = _normalise_str_option(stt_formatting, CONFIG.stt_formatting)
    llm_model_eff = _normalise_str_option(llm_model, CONFIG.model_llm)
    opening_q_eff = _normalise_str_option(opening_question, CONFIG.opening_question)

    _run_cli_session(
        sessions_dir=base_dir,
        llm_model=llm_model_eff,
        stt_backend=stt_backend_eff,
        stt_model=stt_model_eff,
        stt_compute=stt_compute_eff,
        stt_formatting=stt_formatting_eff,
        opening_question=opening_q_eff,
        language=language,
        resume=bool(resume),
        delete_wav_when_safe=bool(delete_wav_when_safe),
        stream_llm=bool(stream_llm),
        voice_mode=bool(voice_mode),
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_format=tts_format,
        llm_questions_debug=bool(llm_questions_debug),
        mic_check=bool(mic_check),
        mode_alias=mode_alias,
    )


def build_app() -> typer.Typer:
    """Build the Typer sub-app for `journal` with explicit subcommands.

    New structure:
      - `healthyselfjournal journal cli` → run the interactive journaling loop
      - `healthyselfjournal journal web` → launch the local web interface (added elsewhere)
    """

    app = typer.Typer(
        add_completion=False,
        no_args_is_help=False,
        invoke_without_command=True,
        context_settings={"help_option_names": ["-h", "--help"]},
        help=(
            "Journaling interfaces.\n\n"
            "Tip: run 'healthyselfjournal journal' to start the CLI interface."
        ),
    )

    # When invoked as `healthyselfjournal journal` with no subcommand, run the CLI
    @app.callback()
    def _default(
        ctx: typer.Context,
        sessions_dir: Path = typer.Option(
            CONFIG.recordings_dir,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
        llm_model: str = typer.Option(
            CONFIG.model_llm,
            "--llm-model",
            help="LLM model spec (e.g., anthropic:claude-sonnet-4:20250514 or ollama:qwen2.5:7b-instruct)",
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
            help="First question to show before recording begins.",
        ),
        language: str = typer.Option(
            "en", "--language", help="Primary language for transcription."
        ),
        resume: bool = typer.Option(
            False,
            "--resume/--new",
            help="Resume the most recent session in the sessions directory.",
        ),
        delete_wav_when_safe: bool = typer.Option(
            True,
            "--delete-wav-when-safe/--keep-wav",
            help="Delete large WAV files once MP3 and STT JSON exist (default on).",
        ),
        stream_llm: bool = typer.Option(
            True,
            "--stream-llm/--no-stream-llm",
            help="Stream the next question as it is generated (off automatically when voice mode).",
        ),
        voice_mode: bool = typer.Option(
            False,
            "--voice-mode/--no-voice-mode",
            help="Speak the assistant's questions using TTS (cloud-only for now).",
        ),
        tts_model: str = typer.Option(
            CONFIG.tts_model, "--tts-model", help="TTS model (OpenAI)."
        ),
        tts_voice: str = typer.Option(
            CONFIG.tts_voice, "--tts-voice", help="TTS voice (OpenAI)."
        ),
        tts_format: str = typer.Option(
            CONFIG.tts_format, "--tts-format", help="TTS audio format."
        ),
        llm_questions_debug: bool = typer.Option(
            False,
            "--llm-questions-debug/--no-llm-questions-debug",
            help="Include hidden metadata in prompts to aid debugging.",
        ),
        mic_check: bool = typer.Option(
            False,
            "--mic-check/--no-mic-check",
            help="Run a brief mic check before starting the session.",
        ),
    ) -> None:
        # Only run the default journaling command when no subcommand is invoked
        # and not in help parsing mode. This prevents 'journal cli --help' from
        # starting a session.
        if getattr(ctx, "invoked_subcommand", None):
            return
        if getattr(ctx, "resilient_parsing", False):
            return
        return journal(
            sessions_dir=sessions_dir,
            llm_model=llm_model,
            stt_backend=stt_backend,
            stt_model=stt_model,
            stt_compute=stt_compute,
            stt_formatting=stt_formatting,
            opening_question=opening_question,
            language=language,
            resume=resume,
            delete_wav_when_safe=delete_wav_when_safe,
            stream_llm=stream_llm,
            voice_mode=voice_mode,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_format=tts_format,
            llm_questions_debug=llm_questions_debug,
            mic_check=mic_check,
        )

    # Explicit subcommands group for the interactive CLI loop and its aliases
    cli_app = typer.Typer(
        add_completion=False,
        no_args_is_help=False,
        invoke_without_command=True,
        context_settings={"help_option_names": ["-h", "--help"]},
        help=(
            "Interactive CLI journaling.\n\n"
            "Run 'healthyselfjournal journal cli' to start immediately, or use:\n"
            "  - 'healthyselfjournal journal cli private' for privacy-first defaults\n"
            "  - 'healthyselfjournal journal cli cloud' for cloud-first defaults"
        ),
    )

    @cli_app.callback()
    def _cli_default(
        ctx: typer.Context,
        sessions_dir: Path = typer.Option(
            CONFIG.recordings_dir,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
        llm_model: str = typer.Option(
            CONFIG.model_llm,
            "--llm-model",
            help="LLM model spec (e.g., anthropic:claude-sonnet-4:20250514 or ollama:qwen2.5:7b-instruct)",
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
            help="First question to show before recording begins.",
        ),
        language: str = typer.Option(
            "en", "--language", help="Primary language for transcription."
        ),
        resume: bool = typer.Option(
            False,
            "--resume/--new",
            help="Resume the most recent session in the sessions directory.",
        ),
        delete_wav_when_safe: bool = typer.Option(
            True,
            "--delete-wav-when-safe/--keep-wav",
            help="Delete large WAV files once MP3 and STT JSON exist (default on).",
        ),
        stream_llm: bool = typer.Option(
            True,
            "--stream-llm/--no-stream-llm",
            help="Stream the next question as it is generated (off automatically when voice mode).",
        ),
        voice_mode: bool = typer.Option(
            False,
            "--voice-mode/--no-voice-mode",
            help="Speak the assistant's questions using TTS (cloud-only for now).",
        ),
        tts_model: str = typer.Option(
            CONFIG.tts_model, "--tts-model", help="TTS model (OpenAI)."
        ),
        tts_voice: str = typer.Option(
            CONFIG.tts_voice, "--tts-voice", help="TTS voice (OpenAI)."
        ),
        tts_format: str = typer.Option(
            CONFIG.tts_format, "--tts-format", help="TTS audio format."
        ),
        llm_questions_debug: bool = typer.Option(
            False,
            "--llm-questions-debug/--no-llm-questions-debug",
            help="Include hidden metadata in prompts to aid debugging.",
        ),
        mic_check: bool = typer.Option(
            False,
            "--mic-check/--no-mic-check",
            help="Run a brief mic check before starting the session.",
        ),
    ) -> None:
        # Only run when no subcommand is invoked (so 'journal cli --help' works)
        if getattr(ctx, "invoked_subcommand", None):
            return
        if getattr(ctx, "resilient_parsing", False):
            return
        return journal(
            sessions_dir=sessions_dir,
            llm_model=llm_model,
            stt_backend=stt_backend,
            stt_model=stt_model,
            stt_compute=stt_compute,
            stt_formatting=stt_formatting,
            opening_question=opening_question,
            language=language,
            resume=resume,
            delete_wav_when_safe=delete_wav_when_safe,
            stream_llm=stream_llm,
            voice_mode=voice_mode,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_format=tts_format,
            llm_questions_debug=llm_questions_debug,
            mic_check=mic_check,
        )

    def _alias_effective_values(
        *,
        stt_backend: Optional[str],
        llm_model: Optional[str],
        private_mode: bool,
    ) -> tuple[Optional[str], Optional[str]]:
        """Return (stt_backend_eff, llm_model_eff) applying alias defaults with precedence.

        Precedence: explicit flags > env > alias defaults.
        """
        # STT backend defaulting with private-mode hardening: never fall back to cloud
        stt_backend_eff: Optional[str] = stt_backend
        env_stt = (os.environ.get("STT_BACKEND") or "").strip().lower()
        if private_mode:
            if stt_backend_eff in {None, ""}:
                if env_stt in {
                    LOCAL_MLX_BACKEND,
                    LOCAL_FASTER_BACKEND,
                    LOCAL_WHISPERCPP_BACKEND,
                    AUTO_PRIVATE_BACKEND,
                }:
                    stt_backend_eff = env_stt
                else:
                    # Ignore cloud env values; prefer auto-private
                    stt_backend_eff = AUTO_PRIVATE_BACKEND
        else:
            if stt_backend_eff in {None, ""} and env_stt in {None, ""}:
                stt_backend_eff = CLOUD_BACKEND

        # LLM model defaulting (only override when neither flag nor env set)
        llm_model_eff: Optional[str] = llm_model
        env_llm = os.environ.get("LLM_MODEL")
        if private_mode and (llm_model_eff in {None, ""}) and (env_llm in {None, ""}):
            llm_model_eff = _OLLAMA_DEFAULT_MODEL

        return stt_backend_eff, llm_model_eff

    @cli_app.command("private")
    def cli_private(
        sessions_dir: Optional[Path] = typer.Option(
            None,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
        llm_model: Optional[str] = typer.Option(
            None,
            "--llm-model",
            help="LLM model spec (defaults to ollama:qwen2.5:7b-instruct in private mode)",
        ),
        stt_backend: Optional[str] = typer.Option(
            None,
            "--stt-backend",
            help=(
                "Transcription backend: cloud-openai, local-mlx, local-faster, "
                "local-whispercpp, or auto-private (default in private mode)."
            ),
        ),
        stt_model: Optional[str] = typer.Option(
            None,
            "--stt-model",
            help="Model preset or identifier for the selected backend.",
        ),
        stt_compute: Optional[str] = typer.Option(
            None,
            "--stt-compute",
            help="Optional compute precision override for local backends (e.g., int8_float16).",
        ),
        stt_formatting: Optional[str] = typer.Option(
            None,
            "--stt-formatting",
            help="Transcript formatting mode: sentences or raw.",
        ),
        opening_question: Optional[str] = typer.Option(
            None, "--opening-question", help="First question before recording begins."
        ),
        language: Optional[str] = typer.Option(
            None, "--language", help="Primary language for transcription."
        ),
        resume: Optional[bool] = typer.Option(
            None,
            "--resume/--new",
            help="Resume the most recent session in the directory.",
        ),
        delete_wav_when_safe: Optional[bool] = typer.Option(
            None,
            "--delete-wav-when-safe/--keep-wav",
            help="Delete large WAV files once MP3 and STT JSON exist.",
        ),
        stream_llm: Optional[bool] = typer.Option(
            None,
            "--stream-llm/--no-stream-llm",
            help="Stream next question as it is generated (auto-off when voice mode).",
        ),
        voice_mode: Optional[bool] = typer.Option(
            None,
            "--voice-mode/--no-voice-mode",
            help="Speak the assistant's questions using TTS (cloud-only for now).",
        ),
        tts_model: Optional[str] = typer.Option(
            None, "--tts-model", help="TTS model (OpenAI)."
        ),
        tts_voice: Optional[str] = typer.Option(
            None, "--tts-voice", help="TTS voice (OpenAI)."
        ),
        tts_format: Optional[str] = typer.Option(
            None, "--tts-format", help="TTS audio format."
        ),
        llm_questions_debug: Optional[bool] = typer.Option(
            None,
            "--llm-questions-debug/--no-llm-questions-debug",
            help="Include hidden metadata in prompts to aid debugging.",
        ),
        mic_check: Optional[bool] = typer.Option(
            None, "--mic-check/--no-mic-check", help="Run a brief mic check first."
        ),
    ) -> None:
        # Apply alias defaults with precedence and enforce cloud_off within process
        stt_backend_eff, llm_model_eff = _alias_effective_values(
            stt_backend=stt_backend, llm_model=llm_model, private_mode=True
        )

        prev_cloud_off = CONFIG.llm_cloud_off
        with _EnvScope({"LLM_CLOUD_OFF": "1"}):
            CONFIG.llm_cloud_off = True
            try:
                return journal(
                    sessions_dir=sessions_dir or CONFIG.recordings_dir,
                    llm_model=llm_model_eff or CONFIG.model_llm,
                    stt_backend=stt_backend_eff or CONFIG.stt_backend,
                    stt_model=stt_model or CONFIG.model_stt,
                    stt_compute=stt_compute or (CONFIG.stt_compute or "auto"),
                    stt_formatting=stt_formatting or CONFIG.stt_formatting,
                    opening_question=opening_question or CONFIG.opening_question,
                    language=language or "en",
                    resume=bool(resume) if resume is not None else False,
                    delete_wav_when_safe=(
                        bool(delete_wav_when_safe)
                        if delete_wav_when_safe is not None
                        else True
                    ),
                    stream_llm=bool(stream_llm) if stream_llm is not None else True,
                    voice_mode=bool(voice_mode) if voice_mode is not None else False,
                    tts_model=tts_model or CONFIG.tts_model,
                    tts_voice=tts_voice or CONFIG.tts_voice,
                    tts_format=tts_format or CONFIG.tts_format,
                    llm_questions_debug=(
                        bool(llm_questions_debug)
                        if llm_questions_debug is not None
                        else False
                    ),
                    mic_check=bool(mic_check) if mic_check is not None else False,
                    mode_alias="private",
                )
            finally:
                CONFIG.llm_cloud_off = prev_cloud_off

    @cli_app.command("cloud")
    def cli_cloud(
        sessions_dir: Optional[Path] = typer.Option(
            None,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
        llm_model: Optional[str] = typer.Option(
            None,
            "--llm-model",
            help="LLM model spec (defaults to cloud model in config/env).",
        ),
        stt_backend: Optional[str] = typer.Option(
            None,
            "--stt-backend",
            help=(
                "Transcription backend: cloud-openai, local-mlx, local-faster, "
                "local-whispercpp, or auto-private."
            ),
        ),
        stt_model: Optional[str] = typer.Option(
            None,
            "--stt-model",
            help="Model preset or identifier for the selected backend.",
        ),
        stt_compute: Optional[str] = typer.Option(
            None,
            "--stt-compute",
            help="Optional compute precision override for local backends (e.g., int8_float16).",
        ),
        stt_formatting: Optional[str] = typer.Option(
            None,
            "--stt-formatting",
            help="Transcript formatting mode: sentences or raw.",
        ),
        opening_question: Optional[str] = typer.Option(
            None, "--opening-question", help="First question before recording begins."
        ),
        language: Optional[str] = typer.Option(
            None, "--language", help="Primary language for transcription."
        ),
        resume: Optional[bool] = typer.Option(
            None,
            "--resume/--new",
            help="Resume the most recent session in the directory.",
        ),
        delete_wav_when_safe: Optional[bool] = typer.Option(
            None,
            "--delete-wav-when-safe/--keep-wav",
            help="Delete large WAV files once MP3 and STT JSON exist.",
        ),
        stream_llm: Optional[bool] = typer.Option(
            None,
            "--stream-llm/--no-stream-llm",
            help="Stream next question as it is generated (auto-off when voice mode).",
        ),
        voice_mode: Optional[bool] = typer.Option(
            None,
            "--voice-mode/--no-voice-mode",
            help="Speak the assistant's questions using TTS (cloud-only for now).",
        ),
        tts_model: Optional[str] = typer.Option(
            None, "--tts-model", help="TTS model (OpenAI)."
        ),
        tts_voice: Optional[str] = typer.Option(
            None, "--tts-voice", help="TTS voice (OpenAI)."
        ),
        tts_format: Optional[str] = typer.Option(
            None, "--tts-format", help="TTS audio format."
        ),
        llm_questions_debug: Optional[bool] = typer.Option(
            None,
            "--llm-questions-debug/--no-llm-questions-debug",
            help="Include hidden metadata in prompts to aid debugging.",
        ),
        mic_check: Optional[bool] = typer.Option(
            None, "--mic-check/--no-mic-check", help="Run a brief mic check first."
        ),
    ) -> None:
        stt_backend_eff, llm_model_eff = _alias_effective_values(
            stt_backend=stt_backend, llm_model=llm_model, private_mode=False
        )

        prev_cloud_off = CONFIG.llm_cloud_off
        with _EnvScope({"LLM_CLOUD_OFF": "0"}):
            CONFIG.llm_cloud_off = False
            try:
                return journal(
                    sessions_dir=sessions_dir or CONFIG.recordings_dir,
                    llm_model=llm_model_eff or CONFIG.model_llm,
                    stt_backend=stt_backend_eff or CONFIG.stt_backend,
                    stt_model=stt_model or CONFIG.model_stt,
                    stt_compute=stt_compute or (CONFIG.stt_compute or "auto"),
                    stt_formatting=stt_formatting or CONFIG.stt_formatting,
                    opening_question=opening_question or CONFIG.opening_question,
                    language=language or "en",
                    resume=bool(resume) if resume is not None else False,
                    delete_wav_when_safe=(
                        bool(delete_wav_when_safe)
                        if delete_wav_when_safe is not None
                        else True
                    ),
                    stream_llm=bool(stream_llm) if stream_llm is not None else True,
                    voice_mode=bool(voice_mode) if voice_mode is not None else False,
                    tts_model=tts_model or CONFIG.tts_model,
                    tts_voice=tts_voice or CONFIG.tts_voice,
                    tts_format=tts_format or CONFIG.tts_format,
                    llm_questions_debug=(
                        bool(llm_questions_debug)
                        if llm_questions_debug is not None
                        else False
                    ),
                    mic_check=bool(mic_check) if mic_check is not None else False,
                    mode_alias="cloud",
                )
            finally:
                CONFIG.llm_cloud_off = prev_cloud_off

    app.add_typer(cli_app, name="cli")

    # Subcommand to launch the web interface
    import os as _os

    if _os.environ.get("HSJ_ENABLE_WEB", "0") in {"1", "true", "yes"}:
        try:
            from .cli_journal_web import (
                web as web_command,
            )  # Lazy-heavy imports are inside the function

            app.command("web")(web_command)
        except Exception:
            # If import fails at build time (e.g., optional deps), we still allow CLI usage.
            pass

    # Subcommand to launch the desktop app (PyWebView shell)
    try:
        from .cli_journal_desktop import (
            desktop as desktop_command,
        )  # Lazy-heavy imports are inside the function

        app.command("desktop")(desktop_command)
    except Exception:
        # Optional dependency (pywebview); if missing, omit the subcommand.
        pass

    return app
