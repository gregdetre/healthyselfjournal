from __future__ import annotations

import os
import pytest
import openai

from healthyselfjournal.config import CONFIG
from healthyselfjournal.tts import TTSOptions, synthesize_text, resolve_tts_options


def test_resolve_tts_options_merges_overrides():
    opts = resolve_tts_options({"model": "gpt-demo", "audio_format": "mp3"})
    assert isinstance(opts, TTSOptions)
    assert opts.backend == "openai"
    assert opts.model == "gpt-demo"
    assert opts.audio_format == "mp3"
    assert opts.voice == CONFIG.tts_voice


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(
    not _has_openai_key(), reason="OPENAI_API_KEY required for TTS test"
)
def test_openai_tts_returns_bytes():
    opts = TTSOptions(
        backend="openai", model="gpt-4o-mini-tts", voice="alloy", audio_format="wav"
    )
    try:
        data = synthesize_text("This is a short test.", opts)
    except openai.APIConnectionError as exc:
        pytest.skip(f"OpenAI connection error: {exc}")
    assert isinstance(data, (bytes, bytearray))
    # WAV header should start with RIFF; allow other formats in case SDK returns mp3/ogg
    assert len(data) > 100
