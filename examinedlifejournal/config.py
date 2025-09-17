from __future__ import annotations

"""Application-wide configuration defaults."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_RECORDINGS_DIR = Path.cwd() / "sessions"
DEFAULT_MAX_RECENT_SUMMARIES = 50
DEFAULT_MAX_HISTORY_TOKENS = 5_000
DEFAULT_SESSION_BREAK_MINUTES = 20
DEFAULT_MODEL_LLM = "anthropic:claude-sonnet-4-20250514"
DEFAULT_MODEL_STT = "gpt-4o-mini-transcribe"
DEFAULT_PROMPT_BUDGET_TOKENS = 8_000
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_BASE_MS = 1_500
DEFAULT_SHORT_ANSWER_DURATION_SECONDS = 1.2
DEFAULT_SHORT_ANSWER_VOICED_SECONDS = 0.6
DEFAULT_VOICE_RMS_DBFS_THRESHOLD = -40.0


@dataclass(slots=True)
class AppConfig:
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR
    model_llm: str = DEFAULT_MODEL_LLM
    model_stt: str = DEFAULT_MODEL_STT
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


CONFIG = AppConfig()
