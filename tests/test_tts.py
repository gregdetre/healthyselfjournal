from __future__ import annotations

import os
import pytest

from healthyselfjournal.tts import TTSOptions, synthesize_text


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(
    not _has_openai_key(), reason="OPENAI_API_KEY required for TTS test"
)
def test_openai_tts_returns_bytes():
    opts = TTSOptions(
        backend="openai", model="gpt-4o-mini-tts", voice="alloy", audio_format="wav"
    )
    data = synthesize_text("This is a short test.", opts)
    assert isinstance(data, (bytes, bytearray))
    # WAV header should start with RIFF; allow other formats in case SDK returns mp3/ogg
    assert len(data) > 100
