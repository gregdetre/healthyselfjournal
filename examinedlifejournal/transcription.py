from __future__ import annotations

"""Speech-to-text transcription helpers with switchable backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
import importlib.util
import logging
import os
import platform
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError

from .events import log_event

_LOGGER = logging.getLogger(__name__)

CLOUD_BACKEND = "cloud-openai"
LOCAL_MLX_BACKEND = "local-mlx"
LOCAL_FASTER_BACKEND = "local-faster"
LOCAL_WHISPERCPP_BACKEND = "local-whispercpp"
AUTO_PRIVATE_BACKEND = "auto-private"

SUPPORTED_BACKENDS = {
    CLOUD_BACKEND,
    LOCAL_MLX_BACKEND,
    LOCAL_FASTER_BACKEND,
    LOCAL_WHISPERCPP_BACKEND,
    AUTO_PRIVATE_BACKEND,
}

MODEL_PRESETS: dict[str, dict[str, str]] = {
    CLOUD_BACKEND: {
        "default": "gpt-4o-transcribe",
        "accuracy": "gpt-4o-transcribe",
        "fast": "gpt-4o-mini-transcribe",
    },
    LOCAL_MLX_BACKEND: {
        "default": "large-v2",
        "accuracy": "large-v2",
        "fast": "small",
    },
    LOCAL_FASTER_BACKEND: {
        "default": "large-v2",
        "accuracy": "large-v2",
        "fast": "small",
    },
    LOCAL_WHISPERCPP_BACKEND: {
        "default": "large-v2",
        "accuracy": "large-v2",
        "fast": "base",
    },
}

DEFAULT_COMPUTE: dict[str, str] = {
    LOCAL_FASTER_BACKEND: "int8_float16",
}

FORMAT_SENTENCES = "sentences"
FORMAT_RAW = "raw"
_FORMATTING_ALIASES = {
    "sentences": FORMAT_SENTENCES,
    "plain": FORMAT_RAW,
    "raw": FORMAT_RAW,
    "none": FORMAT_RAW,
}


class BackendNotAvailableError(RuntimeError):
    """Raised when a backend cannot be constructed due to missing dependencies."""


@dataclass(slots=True)
class BackendSelection:
    backend_id: str
    model: str
    compute: str | None = None
    requested_backend: str | None = None
    requested_model: str | None = None
    requested_compute: str | None = None
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptionResult:
    """Holds the core fields persisted after transcription."""

    text: str
    raw_response: dict[str, Any]
    model: str
    backend: str


class TranscriptionBackend(ABC):
    """Abstract base class for concrete transcription backends."""

    backend_id: str

    def __init__(self, model: str, compute: str | None = None) -> None:
        self.model = model
        self.compute = compute

    @abstractmethod
    def transcribe(
        self, wav_path: Path, *, language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe the given WAV file and return structured output."""

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_id,
            "model": self.model,
            "compute": self.compute,
        }


class OpenAITranscriptionBackend(TranscriptionBackend):
    """Cloud transcription via OpenAI Audio Transcriptions API."""

    backend_id = CLOUD_BACKEND

    def __init__(
        self,
        model: str,
        *,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.5,
    ) -> None:
        super().__init__(model=model, compute=None)
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds

    def transcribe(
        self, wav_path: Path, *, language: str | None = "en"
    ) -> TranscriptionResult:
        client = _get_openai_client()
        last_error: Exception | None = None

        log_event(
            "stt.start",
            {
                "backend": self.backend_id,
                "wav": wav_path.name,
                "model": self.model,
                "language": language,
                "max_retries": self.max_retries,
            },
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                with wav_path.open("rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        file=audio_file,
                        model=self.model,
                        language=language,
                        response_format="json",
                    )

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
                        "backend": self.backend_id,
                        "wav": wav_path.name,
                        "model": self.model,
                        "attempt": attempt,
                        "text_len": len(text),
                    },
                )
                return TranscriptionResult(
                    text=text.strip(),
                    raw_response=raw,
                    model=self.model,
                    backend=self.backend_id,
                )

            except (APIStatusError, APIConnectionError, OpenAIError) as exc:
                last_error = exc
                _LOGGER.warning(
                    "Transcription attempt %s/%s failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                log_event(
                    "stt.retry",
                    {
                        "backend": self.backend_id,
                        "wav": wav_path.name,
                        "model": self.model,
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
                        "backend": self.backend_id,
                        "wav": wav_path.name,
                        "model": self.model,
                        "attempt": attempt,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )

            if attempt < self.max_retries:
                sleep_for = self.backoff_base_seconds * (2 ** (attempt - 1))
                jitter = random.uniform(0, sleep_for * 0.3)
                total_sleep = sleep_for + jitter
                _LOGGER.debug("Retrying transcription in %.2f seconds", total_sleep)
                time.sleep(total_sleep)

        assert last_error is not None
        log_event(
            "stt.failed",
            {
                "backend": self.backend_id,
                "wav": wav_path.name,
                "model": self.model,
                "attempts": self.max_retries,
                "error_type": last_error.__class__.__name__,
                "error": str(last_error),
            },
        )
        raise last_error


class FasterWhisperBackend(TranscriptionBackend):
    """Local transcription via faster-whisper (CTranslate2)."""

    backend_id = LOCAL_FASTER_BACKEND

    def __init__(
        self,
        model: str,
        compute: str | None = None,
        *,
        device: str | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        compute_type = compute or DEFAULT_COMPUTE.get(self.backend_id) or "int8_float16"
        super().__init__(model=model, compute=compute_type)
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise BackendNotAvailableError(
                "faster-whisper not installed. Install with `uv add faster-whisper`."
            ) from exc

        self._model = WhisperModel(
            model,
            device=device or "auto",
            compute_type=compute_type,
            cpu_threads=cpu_threads or 0,
        )

    def transcribe(
        self, wav_path: Path, *, language: str | None = "en"
    ) -> TranscriptionResult:
        log_event(
            "stt.start",
            {
                "backend": self.backend_id,
                "wav": wav_path.name,
                "model": self.model,
                "language": language,
                "compute": self.compute,
            },
        )

        try:
            segments_iter, info = self._model.transcribe(
                str(wav_path), language=language
            )
            segments_list: list[dict[str, Any]] = []
            text_parts: list[str] = []
            for segment in segments_iter:
                seg_text = (segment.text or "").strip()
                if seg_text:
                    text_parts.append(seg_text)
                segments_list.append(
                    {
                        "start": float(getattr(segment, "start", 0.0)),
                        "end": float(getattr(segment, "end", 0.0)),
                        "text": seg_text,
                    }
                )

            text = " ".join(text_parts).strip()
            raw = {
                "segments": segments_list,
                "duration": getattr(info, "duration", None),
                "language": getattr(info, "language", None),
                "temperature": getattr(info, "temperature", None),
            }
            log_event(
                "stt.success",
                {
                    "backend": self.backend_id,
                    "wav": wav_path.name,
                    "model": self.model,
                    "text_len": len(text),
                    "compute": self.compute,
                },
            )
            return TranscriptionResult(
                text=text,
                raw_response=raw,
                model=self.model,
                backend=self.backend_id,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            log_event(
                "stt.failed",
                {
                    "backend": self.backend_id,
                    "wav": wav_path.name,
                    "model": self.model,
                    "compute": self.compute,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise


class MLXWhisperBackend(TranscriptionBackend):
    """Local transcription via Apple MLX `mlx_whisper` CLI."""

    backend_id = LOCAL_MLX_BACKEND

    def __init__(self, model: str) -> None:
        super().__init__(model=model, compute=None)
        if not _has_mlx_whisper():
            raise BackendNotAvailableError(
                "mlx-whisper CLI not found. Install with `uv add mlx-whisper`."
            )

    def transcribe(
        self, wav_path: Path, *, language: str | None = "en"
    ) -> TranscriptionResult:
        model_id = _resolve_mlx_model_id(self.model)
        cmd = [
            "mlx_whisper",
            "--model",
            model_id,
            "--task",
            "transcribe",
            str(wav_path),
        ]
        if language:
            cmd.extend(["--language", language])

        env = {**os.environ, "HF_HUB_ENABLE_TELEMETRY": "0"}
        log_event(
            "stt.start",
            {
                "backend": self.backend_id,
                "wav": wav_path.name,
                "model": self.model,
                "language": language,
            },
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
        except FileNotFoundError as exc:  # pragma: no cover - defensive guard
            raise BackendNotAvailableError(
                "mlx-whisper CLI not found on PATH after initial availability check."
            ) from exc
        except subprocess.CalledProcessError as exc:
            log_event(
                "stt.failed",
                {
                    "backend": self.backend_id,
                    "wav": wav_path.name,
                    "model": self.model,
                    "error": exc.stderr or exc.stdout,
                },
            )
            raise RuntimeError(
                f"mlx_whisper command failed (exit code {exc.returncode})."
            ) from exc

        stdout = proc.stdout or ""
        text, segments = _parse_mlx_stdout(stdout)
        raw = {"stdout": stdout, "segments": segments}
        log_event(
            "stt.success",
            {
                "backend": self.backend_id,
                "wav": wav_path.name,
                "model": self.model,
                "text_len": len(text),
            },
        )
        return TranscriptionResult(
            text=text,
            raw_response=raw,
            model=self.model,
            backend=self.backend_id,
        )


class WhisperCppBackend(TranscriptionBackend):
    """Local transcription via whisper.cpp Python bindings."""

    backend_id = LOCAL_WHISPERCPP_BACKEND

    def __init__(self, model: str) -> None:
        super().__init__(model=model, compute=None)
        try:
            from whispercpp import Whisper  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise BackendNotAvailableError(
                "whispercpp not installed. Install with `uv add whispercpp`."
            ) from exc

        model_path = Path(model).expanduser()
        if not model_path.exists():
            raise BackendNotAvailableError(
                "whispercpp backend expects `--stt-model` to point to a GGUF/GGML file."
            )

        self._model_path = model_path
        self._whisper_cls = Whisper
        if hasattr(Whisper, "from_pretrained"):
            self._whisper = Whisper.from_pretrained(str(model_path))
        else:  # pragma: no cover - fallback constructor
            self._whisper = Whisper(str(model_path))

    def transcribe(
        self, wav_path: Path, *, language: str | None = "en"
    ) -> TranscriptionResult:
        log_event(
            "stt.start",
            {
                "backend": self.backend_id,
                "wav": wav_path.name,
                "model": self.model,
                "language": language,
            },
        )

        try:
            # whispercpp APIs vary slightly; try kwargs first then positional fallback.
            try:
                segments_iter = self._whisper.transcribe(
                    audio=str(wav_path), language=language
                )
            except TypeError:  # pragma: no cover - API variant
                segments_iter = self._whisper.transcribe(str(wav_path))
        except Exception as exc:  # pragma: no cover - defensive guard
            log_event(
                "stt.failed",
                {
                    "backend": self.backend_id,
                    "wav": wav_path.name,
                    "model": self.model,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise

        segments: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for segment in segments_iter:
            seg_text, start, end = _extract_whispercpp_segment(segment)
            if seg_text:
                text_parts.append(seg_text)
            segments.append({"start": start, "end": end, "text": seg_text})

        text = " ".join(text_parts).strip()
        raw = {"segments": segments, "model_path": str(self._model_path)}
        log_event(
            "stt.success",
            {
                "backend": self.backend_id,
                "wav": wav_path.name,
                "model": self.model,
                "text_len": len(text),
            },
        )
        return TranscriptionResult(
            text=text,
            raw_response=raw,
            model=self.model,
            backend=self.backend_id,
        )


@dataclass(slots=True)
class _AutoSelection:
    backend_id: str
    reason: str
    compute: str | None = None
    warnings: list[str] = field(default_factory=list)


def resolve_backend_selection(
    backend: str | None,
    model: str | None,
    compute: str | None = None,
) -> BackendSelection:
    """Resolve requested backend/model/compute into a concrete selection."""

    backend_normalised = (backend or CLOUD_BACKEND).strip().lower()
    model_requested = (model or "default").strip()
    compute_requested = (compute or "auto").strip()

    if backend_normalised not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported STT backend '{backend}'. Expected one of: {sorted(SUPPORTED_BACKENDS)}."
        )

    warnings: list[str] = []
    reason: str | None = None

    if backend_normalised == AUTO_PRIVATE_BACKEND:
        auto = _select_auto_private()
        backend_normalised = auto.backend_id
        reason = auto.reason
        warnings.extend(auto.warnings)
        if compute_requested in {"auto", "", None}:
            compute_requested = auto.compute or compute_requested

    resolved_model = _resolve_model_for_backend(backend_normalised, model_requested)
    resolved_compute, compute_warnings = _resolve_compute_for_backend(
        backend_normalised, compute_requested
    )
    warnings.extend(compute_warnings)

    return BackendSelection(
        backend_id=backend_normalised,
        model=resolved_model,
        compute=resolved_compute,
        requested_backend=backend,
        requested_model=model,
        requested_compute=compute,
        reason=reason,
        warnings=warnings,
    )


def create_transcription_backend(
    selection: BackendSelection,
    *,
    max_retries: int = 3,
    backoff_base_seconds: float = 1.5,
    device: str | None = None,
    cpu_threads: int | None = None,
) -> TranscriptionBackend:
    """Instantiate a concrete backend from a resolved selection."""

    backend_id = selection.backend_id
    model = selection.model
    compute = selection.compute

    if backend_id == CLOUD_BACKEND:
        return OpenAITranscriptionBackend(
            model,
            max_retries=max_retries,
            backoff_base_seconds=backoff_base_seconds,
        )
    if backend_id == LOCAL_FASTER_BACKEND:
        return FasterWhisperBackend(
            model,
            compute=compute,
            device=device,
            cpu_threads=cpu_threads,
        )
    if backend_id == LOCAL_MLX_BACKEND:
        return MLXWhisperBackend(model)
    if backend_id == LOCAL_WHISPERCPP_BACKEND:
        return WhisperCppBackend(model)

    raise ValueError(f"Unsupported backend '{backend_id}'")


def transcribe_wav(
    wav_path: Path,
    model: str,
    language: str | None = "en",
    max_retries: int = 3,
    backoff_base_seconds: float = 1.5,
) -> TranscriptionResult:
    """Backwards-compatible helper that proxies to the OpenAI backend."""

    backend = OpenAITranscriptionBackend(
        model,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
    )
    return backend.transcribe(wav_path, language=language)


def format_transcript_sentences(text: str) -> str:
    """Return a lightly formatted transcript with sentence-per-line splits."""

    if not text:
        return ""

    normalized = " ".join(text.strip().split())
    ellipsis_token = "[[ELLIPSIS]]"
    normalized = normalized.replace("...", ellipsis_token)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)

    sentences: list[str] = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        s = s.replace(ellipsis_token, "...")
        sentences.append(s)

    return "\n".join(sentences)


def apply_transcript_formatting(text: str, mode: str) -> str:
    """Apply the requested transcript formatting mode."""

    if not text:
        return ""

    key = _FORMATTING_ALIASES.get(mode.lower(), mode.lower())
    if key == FORMAT_SENTENCES:
        return format_transcript_sentences(text)
    if key == FORMAT_RAW:
        return text.strip()
    raise ValueError(f"Unsupported formatting mode '{mode}'.")


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    return OpenAI()


def _resolve_model_for_backend(backend: str, requested: str) -> str:
    presets = MODEL_PRESETS.get(backend, {})
    if requested in presets:
        return presets[requested]
    return requested


def _resolve_compute_for_backend(
    backend: str, compute: str | None
) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    compute_clean = (compute or "").strip()

    if backend == LOCAL_FASTER_BACKEND:
        if not compute_clean or compute_clean == "auto":
            return DEFAULT_COMPUTE.get(LOCAL_FASTER_BACKEND), warnings
        return compute_clean, warnings

    if compute_clean and compute_clean != "auto":
        warnings.append(
            f"Compute option '{compute_clean}' is ignored for backend '{backend}'."
        )
    return None, warnings


def _select_auto_private() -> _AutoSelection:
    system = platform.system().lower()
    machine = platform.machine().lower()
    warnings: list[str] = []

    if system == "darwin" and machine in {"arm64", "aarch64"}:
        if _has_mlx_whisper():
            return _AutoSelection(
                backend_id=LOCAL_MLX_BACKEND,
                reason="Apple Silicon detected; mlx-whisper available.",
            )
        warnings.append("Apple Silicon detected but mlx-whisper not found on PATH.")

    if _has_faster_whisper():
        return _AutoSelection(
            backend_id=LOCAL_FASTER_BACKEND,
            reason="faster-whisper package available.",
            compute=DEFAULT_COMPUTE.get(LOCAL_FASTER_BACKEND),
            warnings=warnings,
        )

    if _has_whispercpp():
        return _AutoSelection(
            backend_id=LOCAL_WHISPERCPP_BACKEND,
            reason="whispercpp Python bindings available.",
            warnings=warnings,
        )

    raise BackendNotAvailableError(
        "auto-private could not find a local backend. Install `mlx-whisper`, `faster-whisper`, or `whispercpp` first."
    )


def _has_mlx_whisper() -> bool:
    return (
        bool(shutil.which("mlx_whisper"))
        or importlib.util.find_spec("mlx_whisper") is not None
    )


def _has_faster_whisper() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def _has_whispercpp() -> bool:
    return importlib.util.find_spec("whispercpp") is not None


def _resolve_mlx_model_id(model: str) -> str:
    return model if "/" in model else f"mlx-community/whisper-{model}"


def _parse_mlx_stdout(stdout: str) -> tuple[str, list[dict[str, Any]]]:
    segments: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Downloading"):
            continue

        match = re.match(
            r"\[(?P<start>[0-9:.]+)\s*(?:\u2192|->)\s*(?P<end>[0-9:.]+)\]\s*(?P<text>.*)",
            line,
        )
        if match:
            start = _timecode_to_seconds(match.group("start"))
            end = _timecode_to_seconds(match.group("end"))
            seg_text = match.group("text").strip()
            if seg_text:
                text_parts.append(seg_text)
            segments.append({"start": start, "end": end, "text": seg_text})
            continue

        text_parts.append(line)

    text = " ".join(text_parts).strip()
    return text, segments


def _timecode_to_seconds(value: str) -> float:
    try:
        parts = [float(part) for part in value.split(":")]
    except ValueError:
        return 0.0
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts[-3], parts[-2], parts[-1]
    return hours * 3600 + minutes * 60 + seconds


def _extract_whispercpp_segment(segment: Any) -> tuple[str, float | None, float | None]:
    """Normalise whispercpp segment output across API variants."""

    text = ""
    start: float | None = None
    end: float | None = None

    if hasattr(segment, "text"):
        text = getattr(segment, "text", "") or ""
        start = _maybe_float(getattr(segment, "start", getattr(segment, "t0", None)))
        end = _maybe_float(getattr(segment, "end", getattr(segment, "t1", None)))
    elif isinstance(segment, dict):
        text = str(segment.get("text", ""))
        start = _maybe_float(segment.get("start") or segment.get("t0"))
        end = _maybe_float(segment.get("end") or segment.get("t1"))
    elif isinstance(segment, (list, tuple)) and len(segment) >= 3:
        start = _maybe_float(segment[0])
        end = _maybe_float(segment[1])
        text = str(segment[2] or "")
    else:  # pragma: no cover - defensive fallback
        text = str(segment)

    return text.strip(), start, end


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
