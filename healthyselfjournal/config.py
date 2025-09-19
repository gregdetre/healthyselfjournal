from __future__ import annotations

"""Application-wide configuration defaults."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Tuple

try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # pragma: no cover - fallback
    _toml = None  # type: ignore


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


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
    "LLM_MODEL",
    "anthropic:claude-sonnet-4:20250514:thinking",
    # "LLM_MODEL",
    # "anthropic:claude-opus-4-1:20250805:thinking",
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
DEFAULT_QUIT_DISCARD_DURATION_SECONDS = _env_float("QUIT_DISCARD_DURATION_SECONDS", 5.0)
DEFAULT_TEMPERATURE_QUESTION = 0.5
DEFAULT_TEMPERATURE_SUMMARY = 0.4
DEFAULT_LLM_TOP_P = None
DEFAULT_LLM_TOP_K = None
DEFAULT_MAX_TOKENS_QUESTION = 1200
DEFAULT_MAX_TOKENS_SUMMARY = 1200
DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_TIMEOUT_SECONDS = _env_float("OLLAMA_TIMEOUT_SECONDS", 30.0)
DEFAULT_OLLAMA_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 8192)

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
    # If Q is pressed and total duration is under this threshold, discard the take
    quit_discard_duration_seconds: float = DEFAULT_QUIT_DISCARD_DURATION_SECONDS
    # Optional: delete large WAV files once MP3 and STT JSON exist
    delete_wav_when_safe: bool = True
    # LLM generation controls
    llm_temperature_question: float = DEFAULT_TEMPERATURE_QUESTION
    llm_temperature_summary: float = DEFAULT_TEMPERATURE_SUMMARY
    llm_top_p: float | None = DEFAULT_LLM_TOP_P
    llm_top_k: int | None = DEFAULT_LLM_TOP_K
    llm_max_tokens_question: int = DEFAULT_MAX_TOKENS_QUESTION
    llm_max_tokens_summary: int = DEFAULT_MAX_TOKENS_SUMMARY
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    ollama_num_ctx: int = DEFAULT_OLLAMA_NUM_CTX
    # Optional TTS of LLM questions
    speak_llm: bool = DEFAULT_SPEAK_LLM
    tts_model: str = DEFAULT_TTS_MODEL
    tts_voice: str = DEFAULT_TTS_VOICE
    tts_format: str = DEFAULT_TTS_FORMAT
    # Optional user-specific vocabulary loaded from user_config.toml
    user_vocabulary_terms: List[str] = field(default_factory=list)
    user_config_loaded_from: Path | None = None


CONFIG = AppConfig()


def _find_user_config_path() -> Path | None:
    """Return the first existing user_config.toml path based on precedence.

    Precedence:
    1) HSJ_USER_CONFIG env var (absolute or relative to CWD)
    2) Project root (package parent)/user_config.toml
    3) Current working directory user_config.toml
    4) XDG config (~/.config/healthyselfjournal/user_config.toml)
    """

    # 1) Explicit path via env var
    explicit = os.environ.get("HSJ_USER_CONFIG")
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            return p

    # 2) Project root (parent of package directory)
    project_root = Path(__file__).resolve().parents[1]
    candidate = project_root / "user_config.toml"
    if candidate.exists():
        return candidate

    # 3) Current working directory
    cwd_candidate = Path.cwd() / "user_config.toml"
    if cwd_candidate.exists():
        return cwd_candidate

    # 4) XDG config
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    xdg_path = Path(xdg) / "healthyselfjournal" / "user_config.toml"
    if xdg_path.exists():
        return xdg_path

    return None


def _load_user_config() -> Tuple[List[str], Path | None]:
    """Parse user_config.toml if present and return (terms, path)."""

    path = _find_user_config_path()
    if path is None:
        return [], None
    if _toml is None:  # pragma: no cover - missing toml reader
        return [], None

    try:
        raw_text = path.read_text(encoding="utf-8")
        data = _toml.loads(raw_text)  # type: ignore[attr-defined]
    except Exception:
        # On parse error, ignore silently (treat as absent)
        return [], None

    terms: List[str] = []

    try:
        vocab = data.get("vocabulary") or {}
        raw_terms = vocab.get("terms") or []
        if isinstance(raw_terms, list):
            terms = [str(x) for x in raw_terms if isinstance(x, (str, int, float))]
    except Exception:
        terms = []

    # Also allow top-level keys as a convenience
    try:
        top_terms = data.get("vocabulary_terms")
        if isinstance(top_terms, list):
            terms.extend(
                [str(x) for x in top_terms if isinstance(x, (str, int, float))]
            )
    except Exception:
        pass

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped_terms: List[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            deduped_terms.append(t)

    return deduped_terms, path


# Load user config once at import and attach to CONFIG
try:
    _terms, _path = _load_user_config()
    if _terms:
        CONFIG.user_vocabulary_terms = _terms
        CONFIG.user_config_loaded_from = _path
except Exception:
    # Non-fatal; user_config is optional
    pass
