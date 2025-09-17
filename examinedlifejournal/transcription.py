from __future__ import annotations

"""Speech-to-text transcription helpers."""

from dataclasses import dataclass
import logging
import random
import time
from pathlib import Path
from typing import Any
from functools import lru_cache

from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError

_LOGGER = logging.getLogger(__name__)
from .events import log_event


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    return OpenAI()


@dataclass
class TranscriptionResult:
    """Holds the core fields persisted after transcription."""

    text: str
    raw_response: dict[str, Any]
    model: str


def transcribe_wav(
    wav_path: Path,
    model: str,
    language: str | None = "en",
    max_retries: int = 3,
    backoff_base_seconds: float = 1.5,
) -> TranscriptionResult:
    """Call the Whisper API for the given WAV file with retry/backoff."""

    client = _get_openai_client()
    last_error: Exception | None = None

    log_event(
        "stt.start",
        {
            "wav": wav_path.name,
            "model": model,
            "language": language,
            "max_retries": max_retries,
        },
    )

    for attempt in range(1, max_retries + 1):
        try:
            with wav_path.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    file=audio_file,
                    model=model,
                    language=language,
                    response_format="json",
                )

            # The OpenAI SDK returns a pydantic model; convert to a serialisable dict
            raw = response.model_dump()
            text = raw.get("text") or ""
            _LOGGER.info(
                "Transcription succeeded on attempt %s (len=%s chars)",
                attempt,
                len(text),
            )
            log_event(
                "stt.success",
                {
                    "wav": wav_path.name,
                    "model": model,
                    "attempt": attempt,
                    "text_len": len(text),
                },
            )
            return TranscriptionResult(text=text.strip(), raw_response=raw, model=model)

        except (APIStatusError, APIConnectionError, OpenAIError) as exc:
            last_error = exc
            _LOGGER.warning(
                "Transcription attempt %s/%s failed: %s", attempt, max_retries, exc
            )
            log_event(
                "stt.retry",
                {
                    "wav": wav_path.name,
                    "model": model,
                    "attempt": attempt,
                    "error_type": exc.__class__.__name__,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive catch
            last_error = exc
            _LOGGER.exception("Unexpected transcription failure: %s", exc)
            log_event(
                "stt.error",
                {
                    "wav": wav_path.name,
                    "model": model,
                    "attempt": attempt,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )

        if attempt < max_retries:
            sleep_for = backoff_base_seconds * (2 ** (attempt - 1))
            jitter = random.uniform(0, sleep_for * 0.3)
            total_sleep = sleep_for + jitter
            _LOGGER.debug("Retrying transcription in %.2f seconds", total_sleep)
            time.sleep(total_sleep)

    assert last_error is not None
    log_event(
        "stt.failed",
        {
            "wav": wav_path.name,
            "model": model,
            "attempts": max_retries,
            "error_type": last_error.__class__.__name__,
            "error": str(last_error),
        },
    )
    raise last_error
