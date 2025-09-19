from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient

from healthyselfjournal.llm import QuestionResponse
from healthyselfjournal.session import SessionManager
from healthyselfjournal.transcription import BackendSelection, TranscriptionResult
from healthyselfjournal.web.app import WebAppConfig, build_app
from healthyselfjournal.tts import TTSOptions


class StubBackend:
    backend_id = "stub"

    def __init__(self, *_, **__):
        pass

    def transcribe(
        self, wav_path: Path, *, language: str | None = None
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text="stub transcript",
            raw_response={"text": "stub transcript"},
            model="stub-model",
            backend="stub-backend",
        )


@pytest.fixture()
def web_app(monkeypatch, tmp_path: Path):
    # Force deterministic backend selection without touching real services
    monkeypatch.setattr(
        "healthyselfjournal.web.app.resolve_backend_selection",
        lambda *_, **__: BackendSelection(
            backend_id="stub",
            model="stub-model",
            compute=None,
        ),
    )

    # Avoid spinning background threads in tests
    monkeypatch.setattr(
        SessionManager, "schedule_summary_regeneration", lambda self: None
    )

    monkeypatch.setattr(
        "healthyselfjournal.session.create_transcription_backend",
        lambda *args, **kwargs: StubBackend(),
    )

    monkeypatch.setattr(
        "healthyselfjournal.session.generate_followup_question",
        lambda request: QuestionResponse(question="Stub follow-up?", model="stub-llm"),
    )

    config = WebAppConfig(
        sessions_dir=tmp_path,
        static_dir=tmp_path / "static",
    )
    app = build_app(config)
    return app


def test_tts_endpoint_disabled_returns_error(tmp_path: Path, web_app):
    client = TestClient(web_app)
    # Create a session first
    response = client.get("/")
    assert response.status_code == 200
    match = re.search(r"data-session-id=\"([^\"]+)\"", response.text)
    assert match
    session_id = match.group(1)

    # Voice disabled by default → expect error
    result = client.post(f"/session/{session_id}/tts", json={"text": "Hello"})
    assert result.status_code == 400
    data = result.json()
    assert data["status"] == "error"
    assert data["error"] == "voice_disabled"


def test_tts_endpoint_ok_when_enabled(tmp_path: Path, monkeypatch):
    # Enable voice via app config
    config = WebAppConfig(
        sessions_dir=tmp_path,
        static_dir=tmp_path / "static",
        voice_enabled=True,
    )

    # Stub synthesize_text to avoid OpenAI dependency
    def _stub_synth(text: str, opts: TTSOptions) -> bytes:  # type: ignore[override]
        assert text == "Speak me"
        return b"FAKEAUDIOBYTES"

    monkeypatch.setattr("healthyselfjournal.web.app.synthesize_text", _stub_synth)

    app = build_app(config)
    client = TestClient(app)
    # Open session
    response = client.get("/")
    assert response.status_code == 200
    match = re.search(r"data-session-id=\"([^\"]+)\"", response.text)
    assert match
    session_id = match.group(1)

    result = client.post(f"/session/{session_id}/tts", json={"text": "Speak me"})
    assert result.status_code == 200
    assert result.headers.get("content-type", "").startswith("audio/")
    assert result.content == b"FAKEAUDIOBYTES"


def test_upload_creates_session_artifacts(tmp_path: Path, web_app):
    client = TestClient(web_app)

    # Landing page should create a new session and expose metadata
    response = client.get("/")
    assert response.status_code == 200
    match = re.search(r"data-session-id=\"([^\"]+)\"", response.text)
    assert match, "session id not found in HTML"
    session_id = match.group(1)

    upload_path = f"/session/{session_id}/upload"
    payload = {
        "duration_ms": "1500",
        "voiced_ms": "900",
        "question": "Opening question?",
    }
    files = {
        "audio": ("clip.webm", b"faux-data", "audio/webm"),
    }

    result = client.post(upload_path, data=payload, files=files)
    assert result.status_code == 201
    data = result.json()
    assert data["status"] == "ok"
    assert data["session_id"] == session_id
    assert data["next_question"] == "Stub follow-up?"
    assert data["segment_label"].startswith("browser-")

    session_dir = tmp_path / session_id
    markdown_path = tmp_path / f"{session_id}.md"
    audio_path = session_dir / "browser-001.webm"
    stt_json = audio_path.with_suffix(".stt.json")

    assert markdown_path.exists()
    assert audio_path.exists()
    assert stt_json.exists()

    body = markdown_path.read_text(encoding="utf-8")
    assert "stub transcript" in body

    stt_payload = json.loads(stt_json.read_text(encoding="utf-8"))
    assert stt_payload["text"] == "stub transcript"


def test_resume_latest_session_when_enabled(tmp_path: Path, monkeypatch):
    # Force deterministic backend selection
    monkeypatch.setattr(
        "healthyselfjournal.web.app.resolve_backend_selection",
        lambda *_, **__: BackendSelection(
            backend_id="stub",
            model="stub-model",
            compute=None,
        ),
    )

    monkeypatch.setattr(
        SessionManager, "schedule_summary_regeneration", lambda self: None
    )

    class _StubBackend:
        backend_id = "stub"

        def transcribe(self, wav_path: Path, *, language: str | None = None):
            return TranscriptionResult(
                text="stub transcript",
                raw_response={"text": "stub transcript"},
                model="stub-model",
                backend="stub-backend",
            )

    monkeypatch.setattr(
        "healthyselfjournal.session.create_transcription_backend",
        lambda *args, **kwargs: _StubBackend(),
    )

    # First app without resume: create an initial session
    app1 = build_app(
        WebAppConfig(sessions_dir=tmp_path, static_dir=tmp_path / "static")
    )
    client1 = TestClient(app1)
    resp1 = client1.get("/")
    assert resp1.status_code == 200
    sid_match = re.search(r"data-session-id=\"([^\"]+)\"", resp1.text)
    assert sid_match
    first_sid = sid_match.group(1)

    # Create a second app with resume enabled; it should pick up the first session id
    app2 = build_app(
        WebAppConfig(
            sessions_dir=tmp_path,
            static_dir=tmp_path / "static",
            resume=True,
        )
    )
    client2 = TestClient(app2)
    resp2 = client2.get("/")
    assert resp2.status_code == 200
    sid_match2 = re.search(r"data-session-id=\"([^\"]+)\"", resp2.text)
    assert sid_match2
    resumed_sid = sid_match2.group(1)

    assert (
        resumed_sid == first_sid
    ), "Expected web app with --resume to reuse latest session"
