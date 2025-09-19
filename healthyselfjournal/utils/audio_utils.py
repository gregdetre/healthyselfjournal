from __future__ import annotations

"""Audio-related utilities shared across modules."""

from pathlib import Path
from typing import Iterable, Optional

from ..config import CONFIG

SUPPORTED_UPLOAD_MIME_TYPES: tuple[str, ...] = (
    "audio/webm",
    "audio/webm;codecs=opus",
    "audio/ogg",
    "audio/ogg;codecs=opus",
    "application/ogg",
)


def normalize_mime(mime: str | None) -> str | None:
    """Return a normalized (lowercased) MIME type without parameters."""

    if not mime:
        return None
    base = mime.split(";", 1)[0].strip().lower()
    return base or None


def is_supported_media_type(mime: str | None) -> bool:
    """True if the provided MIME type is in the upload allowlist."""

    normalized = normalize_mime(mime)
    if normalized is None:
        return False
    return normalized in SUPPORTED_UPLOAD_MIME_TYPES


def extension_for_media_type(mime: str | None, filename: str | None = None) -> str:
    """Infer a conservative extension for the supplied MIME/filename."""

    normalized = normalize_mime(mime)
    if normalized in {"audio/webm", "audio/webm;codecs=opus"}:
        return ".webm"
    if normalized in {"audio/ogg", "audio/ogg;codecs=opus", "application/ogg"}:
        return ".ogg"
    if normalized in {"audio/mpeg", "audio/mp3"}:
        return ".mp3"
    if normalized in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return ".wav"
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[-1].lower()
    return ".bin"


def segment_exists(audio_dir: Path, basename: str, *, suffixes: Iterable[str] | None = None) -> bool:
    """Return True if any file matching the basename already exists."""

    if suffixes:
        return any((audio_dir / f"{basename}{suffix}").exists() for suffix in suffixes)
    return any(candidate.exists() for candidate in audio_dir.glob(f"{basename}.*"))


def should_discard_short_answer(
    duration_seconds: float,
    voiced_seconds: float,
    cfg: object = CONFIG,
) -> bool:
    """Return True if the clip should be discarded as a short/quiet answer."""

    try:
        threshold_duration = float(getattr(cfg, "short_answer_duration_seconds"))
    except Exception:
        threshold_duration = CONFIG.short_answer_duration_seconds
    try:
        threshold_voiced = float(getattr(cfg, "short_answer_voiced_seconds"))
    except Exception:
        threshold_voiced = CONFIG.short_answer_voiced_seconds

    return (
        duration_seconds <= threshold_duration
        and voiced_seconds <= threshold_voiced
    )

from ..events import log_event


def maybe_delete_wav_when_safe(wav_path: Path) -> Optional[bool]:
    """Delete WAV if sibling MP3 and STT JSON exist and file still present.

    Returns True if deleted, False if not deleted, or None if an error occurred.
    Emits the existing "audio.wav.deleted" event on success.
    """
    try:
        mp3_path = wav_path.with_suffix(".mp3")
        stt_json = wav_path.with_suffix(".stt.json")
        if mp3_path.exists() and stt_json.exists() and wav_path.exists():
            wav_path.unlink(missing_ok=True)
            try:
                log_event(
                    "audio.wav.deleted",
                    {
                        "wav": wav_path.name,
                        "reason": "safe_delete_after_mp3_and_stt",
                    },
                )
            except Exception:
                pass
            return True
        return False
    except Exception:
        # Defensive: never raise from cleanup path
        return None
