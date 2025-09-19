from __future__ import annotations

from pathlib import Path

from healthyselfjournal.llm import QuestionResponse
from healthyselfjournal.transcription import TranscriptionResult


class StubTranscriptionBackend:
    """Simple transcription backend that returns deterministic text."""

    backend_id = "stub"

    def __init__(self, transcript: str = "stub transcript") -> None:
        self._transcript = transcript

    def transcribe(
        self, wav_path: Path, *, language: str | None = None
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text=self._transcript,
            raw_response={"text": self._transcript, "path": str(wav_path)},
            model="stub-model",
            backend="stub-backend",
        )


def stub_followup_question(text: str = "Stub follow-up?", model: str = "stub-llm") -> QuestionResponse:
    return QuestionResponse(question=text, model=model)


def stub_tts(_: str, __) -> bytes:
    """Return deterministic bytes for TTS tests."""

    return b"FAKEAUDIOBYTES"
