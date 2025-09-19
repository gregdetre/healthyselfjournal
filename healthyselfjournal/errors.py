from __future__ import annotations

"""Shared error code constants for CLI and web surfaces."""

UNKNOWN_SESSION = "unknown_session"
INACTIVE_SESSION = "inactive_session"
MISSING_AUDIO = "missing_audio"
INVALID_PAYLOAD = "invalid_payload"
EMPTY_AUDIO = "empty_audio"
AUDIO_FORMAT_UNSUPPORTED = "audio_format_unsupported"
UPLOAD_TOO_LARGE = "upload_too_large"
PROCESSING_FAILED = "processing_failed"
QUESTION_FAILED = "question_failed"
VOICE_DISABLED = "voice_disabled"
MISSING_TEXT = "missing_text"
TTS_FAILED = "tts_failed"
TTS_ERROR = "tts_error"
SHORT_ANSWER_DISCARDED = "short_answer_discarded"
REVEAL_FAILED = "reveal_failed"
UNSUPPORTED_PLATFORM = "unsupported_platform"

# The constants above define the canonical set of JSON error identifiers
# exposed by the app. Adding new codes should happen in this module so that
# both CLI and web surfaces stay in sync.
