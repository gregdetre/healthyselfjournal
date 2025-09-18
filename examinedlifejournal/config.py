from __future__ import annotations

"""Application-wide configuration defaults."""

from dataclasses import dataclass
import os
from pathlib import Path

_SESSIONS_DIR_ENV = os.environ.get("SESSIONS_DIR") or os.environ.get("RECORDINGS_DIR")
DEFAULT_RECORDINGS_DIR = (
    Path(_SESSIONS_DIR_ENV).expanduser().resolve()
    if _SESSIONS_DIR_ENV
    else Path.cwd() / "sessions"
)
DEFAULT_MAX_RECENT_SUMMARIES = 50
DEFAULT_MAX_HISTORY_TOKENS = 5_000
DEFAULT_SESSION_BREAK_MINUTES = 20
# Default LLM model string. Supports provider:model:version[:thinking]
# If LLM_MODEL is set in env, prefer that; otherwise default to Opus 4.1.
DEFAULT_MODEL_LLM = os.environ.get(
    # thinking mode is broken at the moment
    "LLM_MODEL",
    "anthropic:claude-opus-4-1:20250805:thinking",
    # "LLM_MODEL",
    # "anthropic:claude-opus-4-1:20250805",
)
# Speech-to-text selection defaults; allow env overrides for persistence via .env/.env.local
DEFAULT_STT_BACKEND = os.environ.get("STT_BACKEND", "cloud-openai")
# Model presets are resolved per-backend; "default" maps to provider-specific defaults.
DEFAULT_MODEL_STT = os.environ.get("STT_MODEL", "default")
DEFAULT_STT_COMPUTE = os.environ.get("STT_COMPUTE", "auto")
DEFAULT_STT_FORMATTING = os.environ.get("STT_FORMATTING", "sentences")
DEFAULT_PROMPT_BUDGET_TOKENS = 8_000
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_BASE_MS = 1_500
DEFAULT_SHORT_ANSWER_DURATION_SECONDS = 1.2
DEFAULT_SHORT_ANSWER_VOICED_SECONDS = 0.6
DEFAULT_VOICE_RMS_DBFS_THRESHOLD = -40.0
DEFAULT_TEMPERATURE_QUESTION = 0.5
DEFAULT_TEMPERATURE_SUMMARY = 0.4
DEFAULT_LLM_TOP_P = None
DEFAULT_LLM_TOP_K = None
DEFAULT_MAX_TOKENS_QUESTION = 1200
DEFAULT_MAX_TOKENS_SUMMARY = 1200

# Text-to-speech defaults (OpenAI backend)
DEFAULT_SPEAK_LLM = os.environ.get("SPEAK_LLM", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TTS_MODEL = os.environ.get("TTS_MODEL", "gpt-4o-mini-tts")
DEFAULT_TTS_VOICE = os.environ.get("TTS_VOICE", "shimmer")
DEFAULT_TTS_FORMAT = os.environ.get("TTS_FORMAT", "wav")


@dataclass(slots=True)
class AppConfig:
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR
    model_llm: str = DEFAULT_MODEL_LLM
    model_stt: str = DEFAULT_MODEL_STT
    stt_backend: str = DEFAULT_STT_BACKEND
    stt_compute: str | None = DEFAULT_STT_COMPUTE
    stt_formatting: str = DEFAULT_STT_FORMATTING
    max_recent_summaries: int = DEFAULT_MAX_RECENT_SUMMARIES
    max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS
    prompt_budget_tokens: int = DEFAULT_PROMPT_BUDGET_TOKENS
    retry_max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS
    retry_backoff_base_ms: int = DEFAULT_RETRY_BACKOFF_BASE_MS
    session_break_minutes: int = DEFAULT_SESSION_BREAK_MINUTES
    ffmpeg_path: str | None = None
    opening_question: str = (
        "What feels most present for you right now, and what would you like to explore?"
    )
    # Short-answer auto-discard gating
    short_answer_duration_seconds: float = DEFAULT_SHORT_ANSWER_DURATION_SECONDS
    short_answer_voiced_seconds: float = DEFAULT_SHORT_ANSWER_VOICED_SECONDS
    voice_rms_dbfs_threshold: float = DEFAULT_VOICE_RMS_DBFS_THRESHOLD
    # Optional: delete large WAV files once MP3 and STT JSON exist
    delete_wav_when_safe: bool = True
    # LLM generation controls
    llm_temperature_question: float = DEFAULT_TEMPERATURE_QUESTION
    llm_temperature_summary: float = DEFAULT_TEMPERATURE_SUMMARY
    llm_top_p: float | None = DEFAULT_LLM_TOP_P
    llm_top_k: int | None = DEFAULT_LLM_TOP_K
    llm_max_tokens_question: int = DEFAULT_MAX_TOKENS_QUESTION
    llm_max_tokens_summary: int = DEFAULT_MAX_TOKENS_SUMMARY
    # Optional TTS of LLM questions
    speak_llm: bool = DEFAULT_SPEAK_LLM
    tts_model: str = DEFAULT_TTS_MODEL
    tts_voice: str = DEFAULT_TTS_VOICE
    tts_format: str = DEFAULT_TTS_FORMAT


CONFIG = AppConfig()
