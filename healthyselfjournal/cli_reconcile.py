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
    resolve_backend_selection,
    create_transcription_backend,
)
from .audio import analyze_wav_shortness


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
    """Backfill missing transcriptions for saved WAV files."""

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

    processed = 0
    skipped = 0
    errors = 0

    wav_files = sorted(sessions_dir.rglob("*.wav"))
    if not wav_files:
        console.print("[yellow]No WAV files found.[/]")
        return

    for wav in wav_files:
        stt_json = wav.with_suffix(".stt.json")
        if stt_json.exists():
            skipped += 1
            continue
        # Offline preflight: compute duration and voiced seconds; treat <min_duration or short-guard as too short
        try:
            duration_s, voiced_s, is_short = analyze_wav_shortness(
                wav,
                duration_threshold_seconds=max(
                    min_duration, CONFIG.short_answer_duration_seconds
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            duration_s, voiced_s, is_short = 0.0, 0.0, True
        # Also guard against corrupt/zero-length WAVs via stdlib as a fallback
        if duration_s <= 0.0:
            with contextlib.suppress(Exception):
                with wave.open(str(wav), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate() or 1
                    duration_s = frames / float(rate)
            if duration_s <= 0.0:
                is_short = True

        if duration_s < min_duration or is_short:
            action = (too_short_action or "skip").strip().lower()
            if action not in {"skip", "mark", "delete"}:
                action = "skip"
            if action == "delete":
                with contextlib.suppress(Exception):
                    wav.unlink(missing_ok=True)
                console.print(
                    f"[yellow]Deleted too-short recording:[/] {wav.name} ({duration_s:.3f}s; voiced {voiced_s:.3f}s)"
                )
                processed += 1
                if limit and processed >= limit:
                    break
                continue
            if action == "mark":
                try:
                    payload = {
                        "text": "",
                        "meta": {
                            "skipped_reason": "short_duration",
                            "duration_seconds": round(duration_s, 3),
                            "voiced_seconds": round(voiced_s, 3),
                            "backend": selection.backend_id,
                            "model": selection.model,
                        },
                    }
                    _write_json_atomic(stt_json, payload)
                    console.print(
                        f"[yellow]Marked too-short recording (no STT):[/] {wav.name} ({duration_s:.3f}s; voiced {voiced_s:.3f}s)"
                    )
                except Exception:
                    console.print(
                        f"[yellow]Skipped too-short recording (failed to mark):[/] {wav.name} ({duration_s:.3f}s)"
                    )
                processed += 1
                if limit and processed >= limit:
                    break
                continue
            # Default: skip
            console.print(
                f"[yellow]Skipped too-short recording:[/] {wav.name} ({duration_s:.3f}s; voiced {voiced_s:.3f}s)"
            )
            processed += 1
            if limit and processed >= limit:
                break
            continue
        try:
            result = backend.transcribe(wav, language=language)
            _write_json_atomic(stt_json, result.raw_response)
            # Optional cleanup: delete WAV when safe (MP3 + STT present)
            try:
                if getattr(CONFIG, "delete_wav_when_safe", False):
                    mp3_path = wav.with_suffix(".mp3")
                    if mp3_path.exists() and wav.exists():
                        wav.unlink(missing_ok=True)
                        log_event(
                            "audio.wav.deleted",
                            {
                                "wav": wav.name,
                                "reason": "safe_delete_after_mp3_and_stt",
                            },
                        )
            except Exception:
                pass

            console.print(f"[green]Transcribed:[/] {wav.name}")
            processed += 1
            if limit and processed >= limit:
                break
        except Exception as exc:  # pragma: no cover - defensive surface
            errors += 1
            log_event("reconcile.error", {"wav": wav.name, "error": str(exc)})
            console.print(f"[red]Failed to transcribe {wav.name}:[/] {exc}")

    console.print(
        f"Completed. Processed {processed}; skipped {skipped} existing; errors {errors}."
    )


def _write_json_atomic(output_path: Path, payload: dict) -> None:
    tmp_path = output_path.with_name(output_path.name + ".partial")
    import json

    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(output_path)


def _has_env(var_name: str) -> bool:
    import os

    return bool(os.environ.get(var_name))
