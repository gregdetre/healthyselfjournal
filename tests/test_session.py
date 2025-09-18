from __future__ import annotations

from pathlib import Path

from examinedlifejournal.audio import AudioCaptureResult
from examinedlifejournal.history import HistoricalSummary
from examinedlifejournal.session import Exchange, SessionConfig, SessionManager
from examinedlifejournal.storage import (
    Frontmatter,
    TranscriptDocument,
    load_transcript,
    write_transcript,
)
from examinedlifejournal.transcription import TranscriptionResult
from examinedlifejournal.config import CONFIG
from examinedlifejournal import llm as llm_module


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

    # Backend should never be requested for discarded short answers
    def _fail_backend(self):  # pragma: no cover - explicit guard
        raise AssertionError(
            "transcription backend should not be requested for discarded short answer"
        )

    monkeypatch.setattr(SessionManager, "_get_transcription_backend", _fail_backend)

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
    assert doc.frontmatter.data["stt_backend"] == manager.config.stt_backend
    assert doc.frontmatter.data["stt_formatting"] == manager.config.stt_formatting
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
                    backend="cloud-openai",
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
                    backend="cloud-openai",
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


def test_generate_next_question_handles_give_me_a_question_via_llm(
    tmp_path, monkeypatch
):
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
                backend="cloud-openai",
            ),
        )
    )

    # Mock the LLM path to ensure we still get a question mark and proper model echo
    def fake_generate_followup_question(req):
        # When user asks for variety, prompt instructs model to pick from embedded bank
        return llm_module.QuestionResponse(
            question="What felt most alive for you today?", model=req.model
        )

    # Patch the symbol used inside session module to avoid real API call
    monkeypatch.setattr(
        "examinedlifejournal.session.generate_followup_question",
        fake_generate_followup_question,
    )

    response = manager.generate_next_question("Could you give me a question?")

    assert response.question.endswith("?")
    assert response.model == config.llm_model
    assert state.exchanges[-1].followup_question == response


def test_generate_next_question_streams_with_callback(tmp_path, monkeypatch):
    base_dir = tmp_path / "session"
    config = SessionConfig(
        base_dir=base_dir,
        llm_model="anthropic:test",
        stt_model="whisper-test",
        opening_question="What would you like to explore?",
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
            transcript="I talked about streaming.",
            audio=AudioCaptureResult(
                wav_path=wav,
                mp3_path=None,
                duration_seconds=1.0,
                voiced_seconds=0.8,
                cancelled=False,
                quit_after=False,
            ),
            transcription=TranscriptionResult(
                text="I talked about streaming.",
                raw_response={},
                model="whisper-test",
                backend="cloud-openai",
            ),
        )
    )

    # Monkeypatch the underlying streaming call to emit two chunks and return a final question
    from examinedlifejournal import llm as llm_module

    def fake_stream_followup_question(request, on_delta):
        on_delta("What ")
        on_delta("now?")
        return llm_module.QuestionResponse(question="What now?", model=request.model)

    monkeypatch.setattr(
        llm_module, "stream_followup_question", fake_stream_followup_question
    )

    chunks: list[str] = []

    def on_delta(chunk: str) -> None:
        chunks.append(chunk)

    response = manager.generate_next_question_streaming("Body text", on_delta)

    assert response.question.endswith("?")
    assert "".join(chunks) == "What now?"
    assert state.exchanges[-1].followup_question == response


def test_anthropic_thinking_includes_budget_and_temp_non_stream(monkeypatch):
    captured: dict[str, dict] = {}

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                captured["non_stream"] = kwargs

                class R:
                    content = [type("T", (), {"type": "text", "text": "What now?"})]

                return R()

    monkeypatch.setattr(llm_module, "_ANTHROPIC_CLIENT", FakeClient())

    req = llm_module.QuestionRequest(
        model="anthropic:claude-sonnet-4:20250514:thinking",
        current_transcript="Hello",
        recent_summaries=[],
        opening_question="O",
        question_bank=[],
        language="en",
        conversation_duration="short",
        max_tokens=64,
    )
    llm_module.generate_followup_question(req)

    sent = captured["non_stream"]
    assert sent["temperature"] == 1.0
    assert sent["thinking"]["type"] == "enabled"
    assert sent["thinking"]["budget_tokens"] >= 1024
    assert sent["thinking"]["budget_tokens"] <= CONFIG.prompt_budget_tokens
    # Also ensure budget < max_tokens to preserve output room
    assert sent["thinking"]["budget_tokens"] < req.max_tokens


def test_anthropic_thinking_includes_budget_and_temp_stream(monkeypatch, tmp_path):
    captured: dict[str, dict] = {}

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @property
        def text_stream(self):
            yield "Hi"

        def get_final_message(self):
            return type(
                "M",
                (),
                {"content": [type("T", (), {"type": "text", "text": "What now?"})]},
            )

    class FakeClient:
        class messages:
            @staticmethod
            def stream(**kwargs):
                captured["stream"] = kwargs
                return FakeStream()

    monkeypatch.setattr(llm_module, "_ANTHROPIC_CLIENT", FakeClient())

    req = llm_module.QuestionRequest(
        model="anthropic:claude-sonnet-4:20250514:thinking",
        current_transcript="Hello",
        recent_summaries=[],
        opening_question="O",
        question_bank=[],
        language="en",
        conversation_duration="short",
        max_tokens=64,
    )

    chunks: list[str] = []

    def on_delta(t: str):
        chunks.append(t)

    llm_module.stream_followup_question(req, on_delta)

    sent = captured["stream"]
    assert sent["temperature"] == 1.0
    assert sent["thinking"]["type"] == "enabled"
    assert sent["thinking"]["budget_tokens"] >= 1024
    assert sent["thinking"]["budget_tokens"] <= CONFIG.prompt_budget_tokens
    assert sent["thinking"]["budget_tokens"] < req.max_tokens


def test_budget_clamped_below_max_tokens(monkeypatch):
    captured: dict[str, dict] = {}

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                captured["payload"] = kwargs

                class R:
                    content = [type("T", (), {"type": "text", "text": "Ok"})]

                return R()

    monkeypatch.setattr(llm_module, "_ANTHROPIC_CLIENT", FakeClient())

    # Set a huge configured budget via monkeypatching CONFIG on module
    old_budget = CONFIG.prompt_budget_tokens
    try:
        CONFIG.prompt_budget_tokens = 100000  # way larger than max_tokens

        req = llm_module.SummaryRequest(
            transcript_markdown="# Body",
            recent_summaries=[],
            model="anthropic:claude-sonnet-4:20250514:thinking",
            max_tokens=32,
        )
        llm_module.generate_summary(req)

        sent = captured["payload"]
        assert sent["max_tokens"] == 32
        # Now the budget is enforced to be at least 1024 even if max_tokens is small
        assert sent["thinking"]["budget_tokens"] == 1024
    finally:
        CONFIG.prompt_budget_tokens = old_budget


def test_thinking_budget_respects_minimum_1024(monkeypatch):
    """Budget should be coerced to >= 1024 to satisfy Anthropic requirements."""
    captured: dict[str, dict] = {}

    class FakeBadRequest(Exception):
        pass

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                captured["payload"] = kwargs
                thinking = kwargs.get("thinking")
                # Simulate Anthropic server validation: budget must be >= 1024
                if thinking and thinking.get("type") == "enabled":
                    if thinking.get("budget_tokens", 0) < 1024:
                        raise FakeBadRequest(
                            "thinking.enabled.budget_tokens: Input should be greater than or equal to 1024"
                        )

                class R:
                    content = [type("T", (), {"type": "text", "text": "Ok"})]

                return R()

    monkeypatch.setattr(llm_module, "_ANTHROPIC_CLIENT", FakeClient())

    # Force a small max_tokens; implementation should still send >= 1024 budget
    req = llm_module.SummaryRequest(
        transcript_markdown="# Body",
        recent_summaries=[],
        model="anthropic:claude-sonnet-4:20250514:thinking",
        max_tokens=32,
    )

    llm_module.generate_summary(req)

    sent = captured.get("payload", {})
    assert sent.get("thinking", {}).get("budget_tokens", 0) >= 1024
