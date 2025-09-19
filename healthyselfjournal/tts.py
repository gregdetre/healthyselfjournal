from __future__ import annotations

"""Text-to-speech utilities for speaking assistant questions.

Default backend uses OpenAI TTS and plays back via a local audio player
(`afplay` on macOS; falls back to `ffplay` if available).
"""

from dataclasses import dataclass
import os
import sys
import select
import shutil
import subprocess
import tempfile
from typing import Literal, Any

from .events import log_event
from .config import CONFIG


AudioFormat = Literal["wav", "mp3", "flac", "ogg", "opus", "aac", "pcm"]


@dataclass(slots=True)
class TTSOptions:
    backend: str = "openai"
    model: str = "gpt-4o-mini-tts"
    voice: str = "shimmer"
    audio_format: AudioFormat = "wav"


class TTSError(RuntimeError):
    pass


def resolve_tts_options(overrides: dict[str, Any] | None = None) -> TTSOptions:
    """Merge configuration defaults with runtime overrides."""

    opts = TTSOptions(
        backend="openai",
        model=CONFIG.tts_model,
        voice=CONFIG.tts_voice,
        audio_format=CONFIG.tts_format,  # type: ignore[arg-type]
    )
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if hasattr(opts, key):
                setattr(opts, key, value)
    return opts


def speak_text(text: str, opts: TTSOptions) -> None:
    """Synthesize and play the given text using the configured backend.

    Blocks until playback completes. Raises TTSError on failure.
    """
    if not text.strip():
        return

    try:
        if opts.backend == "openai":
            audio_bytes = synthesize_text(text, opts)
            _play_audio_bytes(audio_bytes, file_ext=opts.audio_format)
        else:
            raise TTSError(f"Unsupported TTS backend: {opts.backend}")
    except Exception as exc:  # pragma: no cover - defensive surface
        log_event(
            "tts.error",
            {
                "backend": opts.backend,
                "model": getattr(opts, "model", None),
                "voice": getattr(opts, "voice", None),
                "format": getattr(opts, "audio_format", None),
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        raise


def synthesize_text(text: str, opts: TTSOptions) -> bytes:
    """Return synthesized speech bytes using the configured backend.

    Does not perform playback, suitable for testing.
    """
    if not text.strip():
        return b""
    if opts.backend == "openai":
        return _synthesize_openai(
            text, model=opts.model, voice=opts.voice, audio_format=opts.audio_format
        )
    raise TTSError(f"Unsupported TTS backend: {opts.backend}")


def _synthesize_openai(
    text: str, *, model: str, voice: str, audio_format: AudioFormat
) -> bytes:
    """Return synthesized speech bytes from OpenAI TTS."""
    # Lazy import to avoid import cost when feature disabled
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise TTSError("OPENAI_API_KEY not set for OpenAI TTS")

    client = OpenAI()

    log_event(
        "tts.request",
        {
            "backend": "openai",
            "model": model,
            "voice": voice,
            "format": audio_format,
            "bytes": 0,
        },
    )

    # Prefer non-streaming API; fall back to streaming if necessary
    try:
        # SDKs differ on parameter name: prefer "response_format" then fall back
        base_kwargs: dict[str, Any] = {"model": model, "voice": voice, "input": text}
        response: Any | None = None
        last_exc: Exception | None = None
        for fmt_key in ("response_format", "format", "output_format"):
            try:
                current_kwargs = dict(base_kwargs)
                current_kwargs[fmt_key] = audio_format
                # type: ignore[arg-type]
                response = client.audio.speech.create(
                    **current_kwargs
                )  # pyright: ignore[reportArgumentType]
                break
            except TypeError as e:
                last_exc = e
                continue
        if response is None:
            if last_exc:
                raise last_exc
            raise TTSError("Failed to call OpenAI TTS create()")

        data: bytes | bytearray | None = None
        to_bytes = getattr(response, "to_bytes", None)
        if callable(to_bytes):
            try:
                data_bytes = to_bytes()
                if isinstance(data_bytes, (bytes, bytearray)):
                    data = data_bytes
            except Exception:
                data = None
        if data is None:
            maybe_content: Any = getattr(response, "content", None)
            if isinstance(maybe_content, (bytes, bytearray)):
                data = maybe_content
        if data is None:
            read = getattr(response, "read", None)
            if callable(read):
                read_bytes = read()
                if isinstance(read_bytes, (bytes, bytearray)):
                    data = read_bytes
        if not isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data) if data is not None else None
            except Exception:
                data = None
        if not data:
            raise TTSError("Empty TTS response from OpenAI")
        out = bytes(data)
        log_event(
            "tts.response",
            {
                "backend": "openai",
                "model": model,
                "voice": voice,
                "format": audio_format,
                "bytes": len(out),
            },
        )
        return out
    except Exception as exc_primary:  # pragma: no cover - fallback path
        # Attempt streaming response fallback for wider client support
        try:
            # type: ignore[attr-defined]
            stream_factory = (
                client.audio.speech.with_streaming_response.create
            )  # pyright: ignore[reportAttributeAccessIssue]
            # Try parameter name variants for streaming as well
            stream_resp_ctx = None
            last_exc2: Exception | None = None
            for fmt_key in ("response_format", "format", "output_format"):
                try:
                    kwargs2: dict[str, Any] = {
                        "model": model,
                        "voice": voice,
                        "input": text,
                    }
                    kwargs2[fmt_key] = audio_format
                    # type: ignore[arg-type]
                    stream_resp_ctx = stream_factory(
                        **kwargs2
                    )  # pyright: ignore[reportArgumentType]
                    break
                except TypeError as e:
                    last_exc2 = e
                    continue
            if stream_resp_ctx is None:
                if last_exc2:
                    raise last_exc2
                raise TTSError("Failed to call OpenAI TTS streaming create()")

            with stream_resp_ctx as stream_resp:
                with tempfile.NamedTemporaryFile(
                    suffix=f".{audio_format}", delete=False
                ) as tmpf:
                    tmp_path = tmpf.name
                try:
                    stream_resp.stream_to_file(tmp_path)
                    with open(tmp_path, "rb") as f:
                        data2 = f.read()
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            if not data2:
                raise TTSError("Empty TTS streaming response from OpenAI")
            log_event(
                "tts.response",
                {
                    "backend": "openai",
                    "model": model,
                    "voice": voice,
                    "format": audio_format,
                    "bytes": len(data2),
                    "mode": "streaming",
                },
            )
            return data2
        except Exception:
            raise exc_primary


def _play_audio_bytes(data: bytes, *, file_ext: str = "wav") -> None:
    """Write bytes to a temp file and play via a local player synchronously."""
    if not data:
        return

    with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        tmp_path = tmp.name

    # Try afplay first (macOS), then ffplay (ffmpeg)
    try:

        def _play_with_enter_skip(cmd: list[str]) -> None:
            """Run player command and allow skipping on ENTER when in a TTY.

            - Spawns the player as a subprocess and waits for completion.
            - If stdin is a TTY, polls for an ENTER (\n or \r); on detection, terminates playback early.
            - Silences player stdout/stderr to keep the UI clean.
            """
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                raise TTSError(f"Failed to start audio player: {e}") from e

            player_name = os.path.basename(cmd[0]) if cmd else "player"

            if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                # Poll stdin for ENTER while the player is running
                try:
                    while proc.poll() is None:
                        try:
                            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                        except Exception:
                            # If select() isn't available, fall back to blocking wait
                            proc.wait()
                            break
                        if rlist:
                            try:
                                ch = sys.stdin.read(1)
                            except Exception:
                                ch = ""
                            if ch in ("\n", "\r"):
                                try:
                                    proc.terminate()
                                except Exception:
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                                log_event(
                                    "tts.skip",
                                    {
                                        "reason": "enter",
                                        "player": player_name,
                                        "format": file_ext,
                                    },
                                )
                                break
                    # Ensure the process has ended
                    try:
                        proc.wait(timeout=1)
                    except Exception:
                        pass
                except KeyboardInterrupt:
                    # Propagate standard interruption semantics
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    raise
            else:
                # Non-interactive: just wait for playback to finish
                proc.wait()

        afplay = shutil.which("afplay")
        if afplay:
            _play_with_enter_skip([afplay, tmp_path])
            return

        ffplay = shutil.which("ffplay")
        if ffplay:
            _play_with_enter_skip(
                [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", tmp_path]
            )
            return

        raise TTSError("No audio player found (tried afplay, ffplay)")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - system dep
        raise TTSError(f"Audio player failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
