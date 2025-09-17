from __future__ import annotations

"""Audio capture utilities for the journaling app.

This module isolates the `sounddevice` recording logic, including the
real-time RMS meter and keyboard control handling described in the V1 plan.
"""

from dataclasses import dataclass
import itertools
import logging
import math
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import readchar
import sounddevice as sd
import soundfile as sf
from rich.live import Live
from rich.text import Text

from .config import CONFIG

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from rich.console import Console


_LOGGER = logging.getLogger(__name__)
from .events import log_event


@dataclass
class AudioCaptureResult:
    """Capture metadata returned after each recording segment."""

    wav_path: Path
    mp3_path: Optional[Path]
    duration_seconds: float
    voiced_seconds: float
    cancelled: bool
    quit_after: bool
    discarded_short_answer: bool = False


def record_response(
    output_dir: Path,
    base_filename: str,
    console: "Console",
    *,
    sample_rate: int = 16_000,
    meter_refresh_hz: float = 20.0,
    ffmpeg_path: str | None = None,
) -> AudioCaptureResult:
    """Record audio until a keypress while updating a visual meter."""

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = _next_available_path(output_dir / f"{base_filename}.wav")
    mp3_path: Optional[Path] = None

    log_event(
        "audio.record.start",
        {
            "wav": wav_path.name,
            "sample_rate": sample_rate,
        },
    )

    console.print(
        Text(
            "Recording started. Press any key to stop (ESC cancels, Q quits after this response).",
            style="bold green",
        )
    )

    frames_queue: queue.Queue[np.ndarray] = queue.Queue()
    level_queue: queue.Queue[float] = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    cancel_flag = threading.Event()
    quit_flag = threading.Event()
    interrupt_flag = threading.Event()

    voiced_seconds = 0.0
    frame_duration_sec = 0.0

    def _audio_callback(
        indata, frames, time_info, status
    ):  # pragma: no cover - exercised at runtime
        if status:
            _LOGGER.warning("Audio status: %s", status)
        frames_queue.put_nowait(indata.copy())
        rms = float(np.sqrt(np.mean(np.square(indata), dtype=np.float64)))
        try:
            level_queue.put_nowait(rms)
        except queue.Full:
            pass

    def _wait_for_stop():  # pragma: no cover - blocking on user input
        try:
            key = readchar.readkey()
        except KeyboardInterrupt:
            stop_event.set()
            interrupt_flag.set()
            return

        if key == readchar.key.CTRL_C:
            stop_event.set()
            interrupt_flag.set()
            return
        if key == readchar.key.ESC:
            cancel_flag.set()
        elif key.lower() == "q":
            quit_flag.set()
        stop_event.set()

    listener_thread = threading.Thread(target=_wait_for_stop, daemon=True)
    listener_thread.start()

    frames_written = 0
    status_text = Text("Initializing input…", style="italic yellow")

    try:
        with sf.SoundFile(
            wav_path,
            mode="x",
            samplerate=sample_rate,
            channels=1,
            subtype="PCM_16",
        ) as wav_file:

            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                callback=_audio_callback,
            ):
                status_text = Text("Recording", style="bold green")
                start_time = time.monotonic()
                last_render = 0.0

                with Live(console=console, auto_refresh=False) as live:
                    while not (stop_event.is_set() and frames_queue.empty()):
                        try:
                            chunk = frames_queue.get(timeout=0.05)
                        except queue.Empty:
                            chunk = None

                        if chunk is not None:
                            wav_file.write(chunk)
                            frames_written += len(chunk)
                            # Update voiced time using simple RMS threshold in dBFS
                            # This is a lightweight proxy for VAD
                            frame_duration_sec = len(chunk) / sample_rate
                            # Convert RMS to dBFS-like scale: db = 20*log10(rms)
                            # Use configured threshold (e.g., -40 dBFS) to count as voiced
                            rms_value = float(
                                np.sqrt(np.mean(np.square(chunk), dtype=np.float64))
                            )
                            if _rms_above_threshold(
                                rms_value, CONFIG.voice_rms_dbfs_threshold
                            ):
                                voiced_seconds += frame_duration_sec

                        if time.monotonic() - last_render >= 1.0 / meter_refresh_hz:
                            level = _drain_latest_level(level_queue)
                            message = _render_meter(level, status_text)
                            live.update(message, refresh=True)
                            last_render = time.monotonic()

                        if stop_event.is_set() and frames_queue.empty():
                            break

                duration_sec = max(frames_written / sample_rate, 0.0)

    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.exception("Error during audio capture: %s", exc)
        log_event(
            "audio.record.error",
            {
                "wav": wav_path.name,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        raise

    finally:
        stop_event.set()
        listener_thread.join(timeout=0.1)

    if interrupt_flag.is_set():
        raise KeyboardInterrupt

    duration_sec = frames_written / sample_rate

    if cancel_flag.is_set():
        wav_path.unlink(missing_ok=True)
        console.print(Text("Recording cancelled.", style="yellow"))
        log_event(
            "audio.record.cancelled",
            {
                "wav": wav_path.name,
            },
        )
        return AudioCaptureResult(
            wav_path=wav_path,
            mp3_path=None,
            duration_seconds=0.0,
            voiced_seconds=0.0,
            cancelled=True,
            quit_after=False,
            discarded_short_answer=False,
        )

    _LOGGER.debug("Captured %.2f seconds to %s", duration_sec, wav_path)

    # Short-answer auto-discard gating: skip saving/transcribing if likely accidental
    discarded_short = False
    if (
        duration_sec <= CONFIG.short_answer_duration_seconds
        and voiced_seconds <= CONFIG.short_answer_voiced_seconds
    ):
        # Treat as noise/accidental: delete wav and do not convert to mp3
        wav_path.unlink(missing_ok=True)
        discarded_short = True
        console.print(Text("Very short answer detected; discarded.", style="yellow"))
        log_event(
            "audio.record.discarded_short",
            {
                "wav": wav_path.name,
                "duration_seconds": round(duration_sec, 2),
                "voiced_seconds": round(voiced_seconds, 2),
                "threshold_duration": CONFIG.short_answer_duration_seconds,
                "threshold_voiced": CONFIG.short_answer_voiced_seconds,
            },
        )
        return AudioCaptureResult(
            wav_path=wav_path,
            mp3_path=None,
            duration_seconds=duration_sec,
            voiced_seconds=voiced_seconds,
            cancelled=False,
            quit_after=quit_flag.is_set(),
            discarded_short_answer=True,
        )

    mp3_path = _maybe_start_mp3_conversion(
        wav_path,
        ffmpeg_path=ffmpeg_path,
    )

    if mp3_path:
        console.print(
            Text(
                f"Saved WAV → {wav_path.name} ({_format_duration(duration_sec)}); MP3 conversion queued.",
                style="green",
            )
        )
    else:
        console.print(
            Text(
                f"Saved WAV → {wav_path.name} ({_format_duration(duration_sec)})",
                style="green",
            )
        )

    return AudioCaptureResult(
        wav_path=wav_path,
        mp3_path=mp3_path,
        duration_seconds=duration_sec,
        voiced_seconds=voiced_seconds,
        cancelled=False,
        quit_after=quit_flag.is_set(),
        discarded_short_answer=False,
    )


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in itertools.count(1):
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate


def _drain_latest_level(level_queue: queue.Queue[float]) -> float:
    level = 0.0
    while True:
        try:
            level = level_queue.get_nowait()
        except queue.Empty:
            break
    return level


def _render_meter(level_rms: float, status_text: Text) -> Text:
    normalized = _normalize_rms(level_rms)
    blocks = 16
    filled = int(round(normalized * blocks))
    filled = max(0, min(filled, blocks))
    bar = "█" * filled + "░" * (blocks - filled)
    text = Text()
    text.append(status_text)
    text.append("  [")
    text.append(bar, style="cyan")
    text.append("] Press any key to stop (ESC cancels, Q quits)")
    return text


def _normalize_rms(rms: float) -> float:
    if rms <= 0:
        return 0.0
    dbfs = 20.0 * math.log10(rms + 1e-10)
    scaled = (dbfs + 60.0) / 60.0
    return max(0.0, min(scaled, 1.0))


def _rms_above_threshold(rms: float, threshold_dbfs: float) -> bool:
    """Return True if the given RMS is above the dBFS threshold.

    - If rms <= 0, treat as silence.
    - threshold_dbfs is negative (e.g., -40.0). We compute db = 20*log10(rms+eps)
      and compare.
    """
    if rms <= 0:
        return False
    db = 20.0 * math.log10(rms + 1e-10)
    return db >= threshold_dbfs


def _format_duration(seconds: float) -> str:
    total_seconds = int(round(max(0.0, seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _maybe_start_mp3_conversion(
    wav_path: Path, ffmpeg_path: str | None = None
) -> Optional[Path]:
    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg:
        _LOGGER.info("ffmpeg not found; skipping MP3 conversion")
        log_event(
            "audio.mp3.skip",
            {
                "wav": wav_path.name,
                "reason": "ffmpeg_not_found",
            },
        )
        return None

    mp3_path = wav_path.with_suffix(".mp3")

    def _convert():  # pragma: no cover - background worker
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(wav_path),
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    str(mp3_path),
                ],
                check=True,
            )
            log_event(
                "audio.mp3.converted",
                {
                    "wav": wav_path.name,
                    "mp3": mp3_path.name,
                },
            )
        except subprocess.CalledProcessError as exc:
            _LOGGER.warning("MP3 conversion failed: %s", exc)
            log_event(
                "audio.mp3.error",
                {
                    "wav": wav_path.name,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
        except FileNotFoundError:
            _LOGGER.warning("ffmpeg vanished during conversion; skipping MP3 output")
            log_event(
                "audio.mp3.error",
                {
                    "wav": wav_path.name,
                    "error_type": "FileNotFoundError",
                    "error": "ffmpeg vanished",
                },
            )

    threading.Thread(target=_convert, daemon=True).start()
    return mp3_path
