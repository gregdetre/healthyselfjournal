from __future__ import annotations

"""Utilities for tracking pending speech-to-text backfill work."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

SUPPORTED_AUDIO_SUFFIXES: tuple[str, ...] = (".wav", ".webm", ".ogg")
SENTINEL_SUFFIX = ".stt.error.txt"
PLACEHOLDER_COMMENT = "<!-- hsj:pending segment=\"{segment}\" -->"


@dataclass(slots=True)
class PendingSegment:
    """Represents an audio clip awaiting transcription."""

    session_id: str
    audio_path: Path
    stt_path: Path
    sentinel_path: Path
    has_error: bool

    @property
    def segment_label(self) -> str:
        return self.audio_path.name


def iter_pending_segments(sessions_dir: Path) -> Iterator[PendingSegment]:
    """Yield pending audio clips that are missing their `.stt.json` payloads."""

    if not sessions_dir.exists():
        return
    seen = set()
    for path in sorted(sessions_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_SUFFIXES:
            continue
        stt_path = path.with_suffix(".stt.json")
        if stt_path.exists():
            continue
        # Avoid double-processing the same stem when a second extension slips through
        key = (path.parent, path.stem)
        if key in seen:
            continue
        seen.add(key)
        sentinel_path = path.with_suffix(SENTINEL_SUFFIX)
        yield PendingSegment(
            session_id=path.parent.name,
            audio_path=path,
            stt_path=stt_path,
            sentinel_path=sentinel_path,
            has_error=sentinel_path.exists(),
        )


def count_pending_segments(sessions_dir: Path) -> int:
    """Return a simple count of pending segments under the sessions directory."""

    return sum(1 for _ in iter_pending_segments(sessions_dir))


def count_pending_for_session(sessions_dir: Path, session_id: str) -> int:
    """Return the number of pending clips for a specific session id."""

    return sum(
        1 for segment in iter_pending_segments(sessions_dir)
        if segment.session_id == session_id
    )


def pending_segments_by_session(sessions_dir: Path) -> dict[str, list[PendingSegment]]:
    """Group pending segments by parent session id."""

    grouped: dict[str, list[PendingSegment]] = {}
    for segment in iter_pending_segments(sessions_dir):
        grouped.setdefault(segment.session_id, []).append(segment)
    return grouped


def reconcile_command_for_dir(sessions_dir: Path) -> str:
    """Return the canonical reconcile command string for display to users."""

    return (
        "uv run --active healthyselfjournal reconcile --sessions-dir "
        f"'{sessions_dir}'"
    )


def write_error_sentinel(audio_path: Path, error: Exception | str) -> Path:
    """Persist a small text file describing the failure for later inspection."""

    message = str(error)
    error_type = error.__class__.__name__ if isinstance(error, Exception) else "Error"
    timestamp = datetime.now().isoformat(timespec="seconds")
    sentinel_path = audio_path.with_suffix(SENTINEL_SUFFIX)
    sentinel_path.write_text(
        (
            f"timestamp: {timestamp}\n"
            f"error_type: {error_type}\n"
            f"message: {message}\n"
        ),
        encoding="utf-8",
    )
    return sentinel_path


def remove_error_sentinel(audio_path: Path) -> None:
    """Delete the failure sentinel if present."""

    sentinel_path = audio_path.with_suffix(SENTINEL_SUFFIX)
    try:
        sentinel_path.unlink(missing_ok=True)
    except Exception:
        pass


def placeholder_comment(segment_label: str) -> str:
    """Return the HTML comment marker that denotes a pending transcript block."""

    return PLACEHOLDER_COMMENT.format(segment=segment_label)


def placeholder_block(segment_label: str) -> str:
    """Return the text inserted into markdown when an STT failure occurs."""

    return (
        f"(transcription pending – segment {segment_label})\n"
        f"{placeholder_comment(segment_label)}"
    )
