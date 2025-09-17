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
import signal
import threading
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
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
    print_saved_message: bool = True,
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
            "Recording started. SPACE pauses/resumes; any key stops (Q quits after this response).",
            style="bold green",
        )
    )

    frames_queue: queue.Queue[np.ndarray] = queue.Queue()
    level_queue: queue.Queue[float] = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    cancel_flag = threading.Event()
    quit_flag = threading.Event()
    interrupt_flag = threading.Event()
    paused_event = threading.Event()

    # Install a temporary SIGINT handler so Ctrl-C always stops recording,
    # even if readchar doesn't surface it as a key while paused.
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):  # pragma: no cover - signal path
        stop_event.set()
        interrupt_flag.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    voiced_seconds = 0.0
    frame_duration_sec = 0.0

    def _audio_callback(
        indata, frames, time_info, status
    ):  # pragma: no cover - exercised at runtime
        if status:
            _LOGGER.warning("Audio status: %s", status)
        # Always compute level for the meter
        rms = float(np.sqrt(np.mean(np.square(indata), dtype=np.float64)))
        try:
            level_queue.put_nowait(rms)
        except queue.Full:
            pass
        # Do not enqueue frames after a stop/cancel has been requested,
        # and only enqueue when not paused
        if not stop_event.is_set() and not paused_event.is_set():
            frames_queue.put_nowait(indata.copy())

    def _wait_for_stop():  # pragma: no cover - blocking on user input
        # Lazy import to avoid hard dependency during unit tests
        try:
            import readchar  # type: ignore
        except Exception as _exc:  # noqa: N816 - local name matches import
            _LOGGER.warning("readchar not available: %s", _exc)
            # Fallback: stop on any input via input() (no pause/cancel support)
            try:
                input()
            except Exception:
                pass
            stop_event.set()
            return
        try:
            while True:
                key = readchar.readkey()
                # Normalize to string in case backend returns bytes
                if isinstance(key, (bytes, bytearray)):
                    try:
                        key = key.decode("utf-8", "ignore")
                    except Exception:
                        key = str(key)
                # Handle Ctrl-C robustly across readchar versions and terminals
                if key == "\x03" or key == getattr(readchar.key, "CTRL_C", None):
                    stop_event.set()
                    interrupt_flag.set()
                    return
                # Treat ESC and any ESC-prefixed sequence (e.g., ANSI) as cancel
                if key == getattr(readchar.key, "ESC", "\x1b") or (
                    isinstance(key, str) and key.startswith("\x1b")
                ):
                    cancel_flag.set()
                    stop_event.set()
                    return
                if key.lower() == "q":
                    quit_flag.set()
                    stop_event.set()
                    return
                # SPACE toggles pause/resume without stopping
                if key == getattr(readchar.key, "SPACE", " ") or key == " ":
                    if paused_event.is_set():
                        paused_event.clear()
                    else:
                        paused_event.set()
                    continue
                # Any other key stops recording
                stop_event.set()
                return
        except KeyboardInterrupt:
            stop_event.set()
            interrupt_flag.set()
            return

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
                            # Drain the latest level regardless to keep queues fresh
                            level = _drain_latest_level(level_queue)
                            is_paused = paused_event.is_set()
                            # Update status text based on pause state
                            status_text = (
                                Text("Paused", style="bold yellow")
                                if is_paused
                                else Text("Recording", style="bold green")
                            )
                            message = _render_meter(
                                level, status_text, paused=is_paused
                            )
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
        # Restore previous SIGINT handler
        try:
            signal.signal(signal.SIGINT, previous_sigint_handler)
        except Exception:
            pass
        stop_event.set()
        listener_thread.join(timeout=0.1)

    if interrupt_flag.is_set():
        raise KeyboardInterrupt

    duration_sec = frames_written / sample_rate

    if cancel_flag.is_set():
        wav_path.unlink(missing_ok=True)
        console.print(
            Text(
                "Cancelled. Take discarded. Press any key to stop; Q ends after next take.",
                style="yellow",
            )
        )
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

    # Lightweight post-processing: trim leading/trailing silence and attenuate peaks
    try:
        new_duration = _postprocess_wav_simple(wav_path, sample_rate)
        if new_duration is not None:
            duration_sec = new_duration
    except Exception:
        # Fail-safe: never block the flow due to post-processing issues
        _LOGGER.debug("Post-processing skipped due to error", exc_info=True)

    mp3_path = _maybe_start_mp3_conversion(
        wav_path,
        ffmpeg_path=ffmpeg_path,
    )

    if print_saved_message:
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


def _render_meter(level_rms: float, status_text: Text, paused: bool = False) -> Text:
    text = Text()
    text.append(status_text)
    if not paused:
        normalized = _normalize_rms(level_rms)
        blocks = 16
        filled = int(round(normalized * blocks))
        filled = max(0, min(filled, blocks))
        bar = "█" * filled + "░" * (blocks - filled)
        text.append("  [")
        text.append(bar, style="cyan")
        text.append("]")
    text.append(" SPACE pause/resume; any key stops (Q quits)")
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
            # Optional cleanup: delete WAV once MP3 is present and STT JSON exists
            try:
                from .config import CONFIG as _CFG

                if getattr(_CFG, "delete_wav_when_safe", False):
                    stt_json = wav_path.with_suffix(".stt.json")
                    if stt_json.exists() and wav_path.exists():
                        try:
                            wav_path.unlink(missing_ok=True)
                            log_event(
                                "audio.wav.deleted",
                                {
                                    "wav": wav_path.name,
                                    "reason": "safe_delete_after_mp3_and_stt",
                                },
                            )
                        except Exception:
                            pass
            except Exception:
                pass
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


def _postprocess_wav_simple(wav_path: Path, input_sample_rate: int) -> Optional[float]:
    """Trim leading/trailing silence and attenuate peaks to avoid clipping.

    Returns the new duration in seconds if changes were written; otherwise None.

    This intentionally avoids heavy dependencies and complex DSP. It uses a
    simple absolute-amplitude threshold derived from the configured dBFS
    threshold and adds small pre/post padding to avoid cutting transients.
    """
    try:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.debug("Failed to read WAV for post-processing: %s", exc)
        log_event(
            "audio.wav.postprocess.error",
            {
                "wav": wav_path.name,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "stage": "read",
            },
        )
        return None

    # Downmix defensively if multi-channel (should be mono already)
    if getattr(audio, "ndim", 1) == 2:
        try:
            audio = audio.mean(axis=1)
        except Exception:
            # If shape unexpected, bail out safely
            return None

    if audio.size == 0:
        return None

    # Compute simple amplitude threshold from dBFS setting
    threshold_dbfs = CONFIG.voice_rms_dbfs_threshold
    amplitude_threshold = float(10.0 ** (threshold_dbfs / 20.0))
    amplitude_threshold = max(1e-5, min(amplitude_threshold, 0.5))

    abs_audio = np.abs(audio)
    above = np.nonzero(abs_audio > amplitude_threshold)[0]

    trimmed = audio
    trimmed_any = False
    if above.size > 0:
        pad_samples = int(0.05 * sr)
        start = max(int(above[0]) - pad_samples, 0)
        end = min(int(above[-1]) + pad_samples + 1, audio.shape[0])
        if end - start > 0 and (start > 0 or end < audio.shape[0]):
            candidate = audio[start:end]
            # Avoid pathological over-trimming: require at least 0.2s remain or skip
            if candidate.shape[0] >= int(0.2 * sr) or audio.shape[0] < int(0.25 * sr):
                trimmed = candidate
                trimmed_any = True

    max_abs = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
    attenuated = False
    # Only attenuate if dangerously close to full-scale
    desired_peak = 0.98
    if max_abs > desired_peak and max_abs > 0:
        scale = desired_peak / max_abs
        trimmed = (trimmed * scale).astype(np.float32, copy=False)
        attenuated = True

    # If nothing changed materially, skip rewrite
    if not trimmed_any and not attenuated:
        return None

    try:
        sf.write(wav_path, trimmed.astype(np.float32, copy=False), sr, subtype="PCM_16")
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.debug("Failed to write post-processed WAV: %s", exc)
        log_event(
            "audio.wav.postprocess.error",
            {
                "wav": wav_path.name,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "stage": "write",
            },
        )
        return None

    try:
        details: dict[str, object] = {
            "wav": wav_path.name,
            "sr": sr,
            "trimmed": trimmed_any,
            "attenuated": attenuated,
            "duration_seconds": round(len(trimmed) / float(sr), 3),
        }
        log_event("audio.wav.postprocess", details)
    except Exception:
        pass

    return len(trimmed) / float(sr)
