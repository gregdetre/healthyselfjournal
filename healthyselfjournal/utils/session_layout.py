from __future__ import annotations

"""Helpers for consistent session segment naming across interfaces."""

from pathlib import Path

from .audio_utils import segment_exists


def next_cli_segment_name(
    session_id: str,
    audio_dir: Path,
    *,
    start_index: int | None = None,
) -> tuple[int, str]:
    """Return the next available CLI segment index and basename."""

    index = max(start_index or 1, 1)
    while True:
        basename = f"{session_id}_{index:02d}"
        if not segment_exists(audio_dir, basename):
            return index, basename
        index += 1


def next_web_segment_name(
    audio_dir: Path,
    *,
    start_index: int | None = None,
) -> tuple[int, str]:
    """Return the next available web segment index and basename."""

    index = max(start_index or 1, 1)
    while True:
        basename = f"browser-{index:03d}"
        if not segment_exists(audio_dir, basename):
            return index, basename
        index += 1


def build_segment_path(audio_dir: Path, basename: str, extension: str) -> Path:
    """Return the full path for a segment basename/extension pair."""

    return audio_dir / f"{basename}{extension}"
