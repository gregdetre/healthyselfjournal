from __future__ import annotations

from pathlib import Path
import contextlib
import wave

import typer
from rich.console import Console

from .config import CONFIG
from .events import log_event
from .transcription import (
    BackendNotAvailableError,
    apply_transcript_formatting,
    create_transcription_backend,
    format_transcript_sentences,
    resolve_backend_selection,
)
from .audio import analyze_wav_shortness
from .storage import clear_pending_flag, replace_pending_exchange
from .utils.audio_utils import maybe_delete_wav_when_safe
from .utils.pending import (
    count_pending_segments,
    iter_pending_segments,
    remove_error_sentinel,
)


console = Console()


def reconcile(
    sessions_dir: Path = typer.Option(
        CONFIG.recordings_dir,
        "--sessions-dir",
        help="Directory where session markdown/audio files are stored.",
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
    language: str = typer.Option(
        "en",
        "--language",
        help="Primary language for transcription.",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Maximum number of recordings to reconcile (0 = no limit).",
    ),
    min_duration: float = typer.Option(
        0.1,
        "--min-duration",
        help="Minimum duration (seconds) to attempt transcription.",
    ),
    too_short_action: str = typer.Option(
        "skip",
        "--too-short",
        help="Action for recordings under thresholds: skip, mark, or delete.",
    ),
) -> None:
    """Backfill missing transcriptions for saved audio recordings."""

    try:
        selection = resolve_backend_selection(stt_backend, stt_model, stt_compute)
    except (ValueError, BackendNotAvailableError) as exc:
        console.print(f"[red]STT configuration error:[/] {exc}")
        raise typer.Exit(code=2)

    # Require OpenAI key only if using cloud STT
    if selection.backend_id == "cloud-openai":
        if not _has_env("OPENAI_API_KEY"):
            console.print("[red]Environment variable OPENAI_API_KEY is required.[/]")
            raise typer.Exit(code=2)

    backend = create_transcription_backend(selection)
    console.print(
        f"Scanning '{sessions_dir}' for recordings needing transcription using "
        f"[bold]{selection.backend_id}[/] ({selection.model})."
    )

    pending_segments = list(iter_pending_segments(sessions_dir))
    initial_pending = len(pending_segments)

    log_event(
        "reconcile.started",
        {
            "sessions_dir": str(sessions_dir),
            "backend": selection.backend_id,
            "model": selection.model,
            "initial_pending": initial_pending,
            "limit": limit,
        },
    )

    if not pending_segments:
        console.print("[yellow]No recordings pending transcription.[/]")
        log_event(
            "reconcile.completed",
            {
                "sessions_dir": str(sessions_dir),
                "backend": selection.backend_id,
                "model": selection.model,
                "initial_pending": initial_pending,
                "transcribed": 0,
                "short_actioned": 0,
                "errors": 0,
                "remaining_pending": 0,
                "placeholders_replaced": 0,
            },
        )
        return

    processed = 0
    transcribed = 0
    short_actioned = 0
    errors = 0
    placeholders_replaced = 0

    action_normalized = (too_short_action or "skip").strip().lower()

    for segment in pending_segments:
        if limit and processed >= limit:
            break

        audio_path = segment.audio_path
        stt_json = segment.stt_path
        suffix = audio_path.suffix.lower()

        duration_s: float | None = None
        voiced_s: float | None = None
        is_short = False

        if suffix == ".wav":
            try:
                duration_s, voiced_s, is_short = analyze_wav_shortness(
                    audio_path,
                    duration_threshold_seconds=max(
                        min_duration, CONFIG.short_answer_duration_seconds
                    ),
                )
            except Exception:  # pragma: no cover - defensive
                duration_s, voiced_s, is_short = 0.0, 0.0, True

            if duration_s <= 0.0:
                with contextlib.suppress(Exception):
                    with wave.open(str(audio_path), "rb") as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate() or 1
                        duration_s = frames / float(rate)
                if duration_s is not None and duration_s <= 0.0:
                    is_short = True

        if suffix == ".wav" and (is_short or (duration_s is not None and duration_s < min_duration)):
            action_choice = action_normalized if action_normalized in {"skip", "mark", "delete"} else "skip"
            duration_text = f"{duration_s:.3f}s" if duration_s is not None else "unknown"
            voiced_text = f"{voiced_s:.3f}s" if voiced_s is not None else "unknown"

            if action_choice == "delete":
                with contextlib.suppress(Exception):
                    audio_path.unlink(missing_ok=True)
                remove_error_sentinel(audio_path)
                console.print(
                    f"[yellow]Deleted too-short recording:[/] {audio_path.name} ({duration_text}; voiced {voiced_text})"
                )
            elif action_choice == "mark":
                payload = {
                    "text": "",
                    "meta": {
                        "skipped_reason": "short_duration",
                        "duration_seconds": round((duration_s or 0.0), 3),
                        "voiced_seconds": round((voiced_s or 0.0), 3),
                        "backend": selection.backend_id,
                        "model": selection.model,
                    },
                }
                try:
                    _write_json_atomic(stt_json, payload)
                    remove_error_sentinel(audio_path)
                    console.print(
                        f"[yellow]Marked too-short recording (no STT):[/] {audio_path.name} ({duration_text}; voiced {voiced_text})"
                    )
                except Exception as exc:  # pragma: no cover - defensive surface
                    console.print(
                        f"[yellow]Failed to mark short recording:[/] {audio_path.name} ({exc})"
                    )
            else:
                console.print(
                    f"[yellow]Skipped too-short recording:[/] {audio_path.name} ({duration_text}; voiced {voiced_text})"
                )

            processed += 1
            short_actioned += 1
            continue

        try:
            result = backend.transcribe(audio_path, language=language)
        except Exception as exc:  # pragma: no cover - defensive surface
            errors += 1
            processed += 1
            log_event(
                "reconcile.error",
                {
                    "segment": audio_path.name,
                    "session_id": segment.session_id,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            console.print(f"[red]Failed to transcribe {audio_path.name}:[/] {exc}")
            continue

        _write_json_atomic(stt_json, result.raw_response)
        remove_error_sentinel(audio_path)

        try:
            formatted_text = apply_transcript_formatting(
                result.text,
                CONFIG.stt_formatting,
            )
        except ValueError:
            formatted_text = format_transcript_sentences(result.text)

        markdown_path = sessions_dir / f"{segment.session_id}.md"
        if markdown_path.exists():
            if replace_pending_exchange(
                markdown_path, segment.segment_label, formatted_text
            ):
                clear_pending_flag(markdown_path, segment.segment_label)
                placeholders_replaced += 1
                log_event(
                    "reconcile.placeholder_replaced",
                    {
                        "session_id": segment.session_id,
                        "segment": segment.segment_label,
                    },
                )

        if suffix == ".wav" and getattr(CONFIG, "delete_wav_when_safe", False):
            maybe_delete_wav_when_safe(audio_path)

        console.print(
            f"[green]Transcribed:[/] {segment.session_id}/{audio_path.name}"
        )
        transcribed += 1
        processed += 1

    remaining = count_pending_segments(sessions_dir)
    console.print(
        f"Completed. Transcribed {transcribed}; short-action {short_actioned}; "
        f"errors {errors}; placeholders updated {placeholders_replaced}; "
        f"remaining pending {remaining}."
    )
    log_event(
        "reconcile.completed",
        {
            "sessions_dir": str(sessions_dir),
            "backend": selection.backend_id,
            "model": selection.model,
            "initial_pending": initial_pending,
            "transcribed": transcribed,
            "short_actioned": short_actioned,
            "errors": errors,
            "remaining_pending": remaining,
            "placeholders_replaced": placeholders_replaced,
        },
    )


def _write_json_atomic(output_path: Path, payload: dict) -> None:
    tmp_path = output_path.with_name(output_path.name + ".partial")
    import json

    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(output_path)


def _has_env(var_name: str) -> bool:
    import os

    return bool(os.environ.get(var_name))
