from __future__ import annotations

from pathlib import Path

from examinedlifejournal.audio import AudioCaptureResult
from examinedlifejournal.history import HistoricalSummary
from examinedlifejournal.question_bank import QUESTION_BANK
from examinedlifejournal.session import Exchange, SessionConfig, SessionManager
from examinedlifejournal.storage import (
    Frontmatter,
    TranscriptDocument,
    load_transcript,
    write_transcript,
)
from examinedlifejournal.transcription import TranscriptionResult
from examinedlifejournal.config import CONFIG


def test_short_answer_discard_sets_quit_and_skips_transcription(tmp_path, monkeypatch):
    base_dir = tmp_path / "session"
    config = SessionConfig(
        base_dir=base_dir,
        llm_model="anthropic:test",
        stt_model="whisper-test",
        opening_question="What is present?",
        app_version="test-app",
    )
    manager = SessionManager(config)
    state = manager.start()

    # Monkeypatch record_response to simulate a very short, low-voiced capture with quit requested
    fake_capture = AudioCaptureResult(
        wav_path=base_dir / state.session_id / f"{state.session_id}_01.wav",
        mp3_path=None,
        duration_seconds=CONFIG.short_answer_duration_seconds,
        voiced_seconds=0.0,
        cancelled=False,
        quit_after=True,
        discarded_short_answer=True,
    )

    monkeypatch.setattr(
        "examinedlifejournal.session.record_response", lambda *a, **k: fake_capture
    )

    # Monkeypatch transcribe_wav to raise if called (it should not be called)
    def _fail(*args, **kwargs):  # pragma: no cover - explicit guard
        raise AssertionError(
            "transcribe_wav should not be called for discarded short answer"
        )

    monkeypatch.setattr("examinedlifejournal.session.transcribe_wav", _fail)

    exchange = manager.record_exchange("Test Q", None)  # console is unused in this path

    assert exchange is None
    assert manager.state is not None and manager.state.quit_requested is True


def _create_transcript(path: Path, summary: str) -> None:
    frontmatter = Frontmatter(data={"summary": summary})
    write_transcript(path, TranscriptDocument(frontmatter=frontmatter, body=""))


def test_session_start_carries_recent_history(tmp_path):
    base_dir = tmp_path / "sessions"
    base_dir.mkdir()

    _create_transcript(base_dir / "250101_0900.md", "Explored morning routine.")
    _create_transcript(base_dir / "250101_1000.md", "Focused on progress at work.")
    _create_transcript(base_dir / "250101_1100.md", "Reflected on gratitude.")

    config = SessionConfig(
        base_dir=base_dir,
        llm_model="anthropic:test-v1",
        stt_model="whisper-test",
        opening_question="How are you arriving today?",
        max_history_tokens=1000,
        recent_summaries_limit=2,
        app_version="test-app",
    )
    manager = SessionManager(config)
    state = manager.start()

    created_path = base_dir / f"{state.session_id}.md"
    doc = load_transcript(created_path)

    assert doc.frontmatter.data["recent_summary_refs"] == [
        "250101_1000.md",
        "250101_1100.md",
    ]
    assert [item.summary for item in state.recent_history] == [
        "Focused on progress at work.",
        "Reflected on gratitude.",
    ]
    assert doc.frontmatter.data["model_llm"] == "anthropic:test-v1"
    assert doc.frontmatter.data["model_stt"] == "whisper-test"
    assert doc.frontmatter.data["transcript_file"] == created_path.name


def test_session_complete_updates_frontmatter(tmp_path):
    base_dir = tmp_path / "session"
    config = SessionConfig(
        base_dir=base_dir,
        llm_model="anthropic:test",
        stt_model="whisper-test",
        opening_question="What is present?",
        max_history_tokens=800,
        recent_summaries_limit=3,
        app_version="test-app",
    )
    manager = SessionManager(config)
    state = manager.start()

    audio_dir = base_dir / state.session_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav1 = audio_dir / f"{state.session_id}_01.wav"
    wav2 = audio_dir / f"{state.session_id}_02.wav"
    mp3_path = audio_dir / f"{state.session_id}_02.mp3"
    wav1.write_bytes(b"fake-wav-1")
    wav2.write_bytes(b"fake-wav-2")
    mp3_path.write_bytes(b"fake-mp3")

    state.recent_history = [
        HistoricalSummary(
            filename="250101_1000.md", summary="Context carries forward."
        ),
    ]

    state.exchanges.extend(
        [
            Exchange(
                question="How did the day begin?",
                transcript="It felt slow but intentional.",
                audio=AudioCaptureResult(
                    wav_path=wav1,
                    mp3_path=None,
                    duration_seconds=1.5,
                    voiced_seconds=1.2,
                    cancelled=False,
                    quit_after=False,
                ),
                transcription=TranscriptionResult(
                    text="It felt slow but intentional.",
                    raw_response={},
                    model="whisper-test",
                ),
            ),
            Exchange(
                question="What supported you later on?",
                transcript="A walk cleared my head.",
                audio=AudioCaptureResult(
                    wav_path=wav2,
                    mp3_path=mp3_path,
                    duration_seconds=2.25,
                    voiced_seconds=1.8,
                    cancelled=False,
                    quit_after=False,
                ),
                transcription=TranscriptionResult(
                    text="A walk cleared my head.",
                    raw_response={},
                    model="whisper-test",
                ),
            ),
        ]
    )

    manager.complete()

    doc = load_transcript(state.markdown_path)
    assert doc.frontmatter.data["duration_seconds"] == 3.75
    assert doc.frontmatter.data["recent_summary_refs"] == ["250101_1000.md"]
    assert doc.frontmatter.data["audio_file"] == [
        {"wav": wav1.name, "mp3": None, "duration_seconds": 1.5},
        {"wav": wav2.name, "mp3": mp3_path.name, "duration_seconds": 2.25},
    ]


def test_generate_next_question_uses_question_bank(tmp_path, monkeypatch):
    base_dir = tmp_path / "session"
    config = SessionConfig(
        base_dir=base_dir,
        llm_model="anthropic:test",
        stt_model="whisper-test",
        opening_question="What would you like to explore?",
        max_history_tokens=800,
        recent_summaries_limit=3,
        app_version="test-app",
    )
    manager = SessionManager(config)
    state = manager.start()

    audio_dir = base_dir / state.session_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav = audio_dir / f"{state.session_id}_01.wav"
    wav.write_bytes(b"fake-wav")

    state.exchanges.append(
        Exchange(
            question="Opening question",
            transcript="I talked about my day.",
            audio=AudioCaptureResult(
                wav_path=wav,
                mp3_path=None,
                duration_seconds=1.0,
                voiced_seconds=0.8,
                cancelled=False,
                quit_after=False,
            ),
            transcription=TranscriptionResult(
                text="I talked about my day.",
                raw_response={},
                model="whisper-test",
            ),
        )
    )

    monkeypatch.setattr("random.choice", lambda seq: seq[0])

    response = manager.generate_next_question("Could you give me a question?")

    assert response.question == QUESTION_BANK[0]
    assert response.model == "question-bank"
    assert state.exchanges[-1].followup_question == response
